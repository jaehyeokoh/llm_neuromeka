import numpy as np
import torch
import pyrealsense2 as rs
import sys
torch.cuda.empty_cache()
import base64
# GROUND_SAM2_PATH = os.path.join(os.path.dirname(__file__), "Grounded-SAM-2")
# sys.path.append(GROUND_SAM2_PATH)

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2 # GroundedSAM2 모듈
from sam2.sam2_image_predictor import SAM2ImagePredictor # GroundedSAM2 모듈
from sklearn.decomposition import PCA
from typing import Optional, Dict, Any, List , Tuple# Type Hinting 추가
from mcp.server.fastmcp import FastMCP, Context  # MCP 관련 import
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import asyncio 
import cv2
import logging
import re
from typing import Union
# 현재 스크립트 파일 기준 경로 설정
# base_dir = os.path.dirname(os.path.abspath(__file__))
# print(base_dir)

# 각 축(W, L, H)에 대한 크기 보정 오프셋 (단위: 미터)
X_OFFSETS = [-0.016, 0, 0]       # 인덱스별 X 오프셋 인덱스: 카메라 번호(카메라 여러개 사용할것으로 생각해서 만듬)
Y_OFFSETS = [0, 0, 0]       # 인덱스별 Y 오프셋 # indy에서는 이게 남/북 방향임
Z_OFFSETS = [0, 0, 0]   # 인덱스별 Height 오프셋

# 전역 싱글톤 인스턴스들
_model_wrapper_instance = None
_camera_capture_instance = None

# 카메라가 기울어진 각도,  단위: 도(degree)
camera_tilt_theta = 0.0  # 실제 카메라가 아래를 향해 기울어져 있다고 가정

# 연결된 카메라 목록 확인
def list_connected_realsense_devices():
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]

def wrapper_for_indy(results):
    """
    입력: position 필드를 가진 dict들의 리스트
    동작: 각 dict의 
      - position [x,y,z]을 [y, -x, z]로,
      - rotation_degree (°, CW, 원래 x축 기준) 를 새 축계 기준으로 보정
        (rotation_new = rotation_old - 90°) 해서 반환
    """
    wrapped = []
    try:
        for r in results:
            # 원본을 바로 수정하지 않으려면 사본을 만듭니다.
            new_r = r.copy()

            # 1) position 변환
            x, y, z = new_r["position"]
            new_r["position"] = [y, -x, z]

            # 2) rotation_degree 변환
            rot = new_r.get("rotation_degree", 0)
            rot_new = rot - 90
            # -180°~+180° 범위로 정규화 (선택)
            rot_new = (rot_new + 180) % 360 - 180
            new_r["rotation_degree"] = rot_new

            wrapped.append(new_r)
    except:
        wrapped = results
    return wrapped

class GroundedSAM2Wrapper:
    def __init__(
        self,
        dino_model_id="IDEA-Research/grounding-dino-base",
        sam2_config_path="configs/sam2.1/sam2.1_hiera_l.yaml", # 경로 오류 뜨면 Grounded-SAM-2 폴더 내부 위치 넣기
        sam2_ckpt_path = "Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt", # 경로 오류 뜨면 Grounded-SAM-2 폴더 내부 위치 넣기
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 1. Grounding DINO 로딩
        try:
            print("Loading Grounding DINO model...")
            self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_id).to(self.device)
            print("Grounding DINO loaded.")
        except Exception as e:
            print(f"Error loading Grounding DINO: {e}")
            raise

        # 2. SAM2 predictor 로딩
        try:
            print("Loading SAM2 model...")
            sam2_model = build_sam2(sam2_config_path, sam2_ckpt_path, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)
            print("SAM2 loaded.")
        except Exception as e:
            print(f"Error loading SAM2: {e}")
            raise

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.27, # 박스 감지 임계값 -> 낮을수록 더 느슨하게 감지
        text_threshold: float = 0.27, # 텍스트 감지 임계값 -> 낮을수록 더 느슨하게 감지
    ):
        try:
            # BGR → RGB 변환 (이거 안하면 이미지 색상 반전된거로 감지해서 감지 잘 안됨)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)
            image_np = image_rgb
        except Exception as e:
            print(f"Error converting image: {e}")
            raise TypeError("Input image must be a NumPy array.")

        if not text_prompt.strip().endswith("."):
            text_prompt += "."

        inputs = self.dino_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        target_size = [image_pil.size[::-1]]  # [H, W]

        detections = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_size,
        )[0]

        boxes = detections["boxes"]
        scores = detections["scores"]
        phrases = detections["labels"]

        if boxes.size(0) == 0:
            return torch.empty(0, dtype=torch.bool, device=self.device), \
                torch.empty((0, 4), dtype=torch.float, device=self.device), \
                [], \
                torch.empty(0, dtype=torch.float, device=self.device)

        self.sam2_predictor.set_image(image_np)
        boxes_np = boxes.cpu().numpy()

        sam_masks, sam_scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_np,
            multimask_output=False,
        )

        if sam_masks.ndim == 4:
            sam_masks = sam_masks.squeeze(1)

        masks_tensor = torch.from_numpy(sam_masks).to(self.device)

        return masks_tensor, boxes, phrases, scores

# 싱글톤 패턴으로 모델 래퍼 인스턴스 관리
def get_model_wrapper():
    global _model_wrapper_instance
    if _model_wrapper_instance is None:
        print("Creating new GroundedSAM2Wrapper instance...")
        _model_wrapper_instance = GroundedSAM2Wrapper()
        print("GroundedSAM2Wrapper instance created.")
    else:
        print("Using existing GroundedSAM2Wrapper instance.")
    return _model_wrapper_instance

# RealSense 카메라 캡처 클래스
class RealSenseCapture:
    def __init__(self, serial_number=None,depth_w=848, depth_h=480, color_w=848, color_h=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.is_running = False
        self.depth_scale = None
        self.color_intrinsics = None
        self.depth_w, self.depth_h = depth_w, depth_h
        self.color_w, self.color_h = color_w, color_h
        self.fps = fps
        if serial_number:
            self.config.enable_device(serial_number)

        try:
            self.config.enable_stream(rs.stream.depth, self.depth_w, self.depth_h, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.bgr8, self.fps)
        except RuntimeError as e:
            print(f"Error enabling stream: {e}. Check if resolution/fps is supported.")
            self.pipeline = None
            raise

    def start(self):
        if not self.pipeline: return False
        if not self.is_running:
            try:
                print("Starting RealSense pipeline...")
                profile = self.pipeline.start(self.config)
                self.is_running = True
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()

                align_to = rs.stream.color
                self.align = rs.align(align_to)

                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.color_intrinsics = color_stream.get_intrinsics()
                print(f"RealSense started. Depth Scale: {self.depth_scale:.4f}")

                print("Waiting for frames to stabilize...")
                for _ in range(100): # 안정화 시간 필요 -> 이거 안하면 초반에 자기가 색상 조율하느라 초록빛으로 나옴
                    self.pipeline.wait_for_frames()
                print("Stabilization complete.")
                return True
            except RuntimeError as e:
                print(f"Failed to start pipeline: {e}")
                self.is_running = False
                return False
        return True

    def capture_aligned_frames(self, timeout_ms=5000):
        if not self.is_running:
            print("Error: Pipeline not running.")
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if not frames:
                print("Error: Failed to receive frames (timeout).")
                return None, None

            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("Error: Failed to get valid aligned frames.")
                return None, None

            depth_image_raw = np.asanyarray(depth_frame.get_data()) # uint16
            color_image = np.asanyarray(color_frame.get_data()) # BGR

            return color_image, depth_image_raw

        except Exception as e:
            print(f"Error during frame capture: {e}")
            return None, None

    def get_intrinsics(self):
        return self.color_intrinsics

    def get_depth_scale(self):
        return self.depth_scale

    def stop(self):
        if self.is_running:
            print("Stopping RealSense pipeline...")
            self.pipeline.stop()
            self.is_running = False
            print("Pipeline stopped.")

# 싱글톤 패턴으로 카메라 캡처 인스턴스 관리
def get_camera_capture():
    global _camera_capture_instance  # Declare _camera_capture_instance as global at the start of the function
    
    if _camera_capture_instance is None:
        print("Creating new RealSenseCapture instance...")
        # 연결된 카메라 목록에서 첫 번째 선택
        camera_serials = list_connected_realsense_devices()
        if not camera_serials:
            raise RuntimeError("No RealSense cameras connected.")
        selected_serial = camera_serials[0]
        
        _camera_capture_instance = RealSenseCapture(serial_number=selected_serial)
        
        if not _camera_capture_instance.start():
            raise RuntimeError("Failed to start RealSense camera.")
        print("RealSenseCapture instance created and started.")
    else:
        print("Using existing RealSenseCapture instance.")
        
    return _camera_capture_instance




######################################## 포인트 클라우드와 컬러 추가하는 코드 ##################33
# 감지된 물체의 3d 좌표 및 정보 파악
def calculate_object_properties(mask, depth_image_raw, intrinsics, depth_scale):
    # 1) 입력 검증
    if intrinsics is None or depth_scale is None:
        print("Error: Invalid camera intrinsics or depth scale.")
        return None, None, None, None
    if mask.shape != depth_image_raw.shape:
        print("Error: Mask shape mismatch.")
        return None, None, None, None
    if mask.dtype != bool:
        mask = mask > 0

    pts_yx = np.argwhere(mask)
    if pts_yx.size == 0:
        print("Warning: No mask pixels.")
        return None, None, None, None

    # 2) 깊이 → 미터
    depth_m = depth_image_raw.astype(np.float32) * depth_scale

    # 3) 왜곡 보정 포함 deproject → 3D 점군
    pc_list = []
    for y, x in pts_yx:
        d = depth_m[y, x]
        if 0.01 < d < 10.0:
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            pc_list.append([X, Y, Z])

    if len(pc_list) < 10:
        print("Warning: Too few 3D points.")
        return None, None, None, None

    pc = np.array(pc_list, dtype=np.float32)
    center_3d = pc.mean(axis=0).tolist()


    # 4) 절대 높이
    z_vals = pc[:, 2]
    height = float(z_vals.max() - z_vals.min())

    # 5) PCA → OBB 축 길이
    pca = PCA(n_components=3).fit(pc)
    t_pc = pca.transform(pc)
    size_obb = (t_pc.max(axis=0) - t_pc.min(axis=0)).tolist()  # [d0,d1,d2]

    # 6) 높이에 가장 가까운 축 제거 → L, W
    diffs = [abs(d - height) for d in size_obb]
    idx_h = int(np.argmin(diffs))
    lw = [d for i, d in enumerate(size_obb) if i != idx_h]
    L, W = max(lw), min(lw)
    obb_dims = [L, W, height]

    # 7) tilt/rot: 수평면상 가장 긴 축 기준
    axes = pca.components_
    proj = np.hypot(axes[:,0], axes[:,1])
    idx_horiz = int(np.argmax(proj))
    v_h = axes[idx_horiz]
    norm_h = np.linalg.norm(v_h)
    tilt_deg = float(np.degrees(np.arcsin(abs(v_h[2]) / norm_h)))
    rot_deg  = float(np.degrees(np.arctan2(-v_h[1], v_h[0]))) # y 방향 바꿔서 이것도 바꿈

    # 만약 부호 -면 +로 바꿔주는 코드
    if abs(rot_deg) > 90:
        rot_deg = (rot_deg + 180) if rot_deg < 0 else (rot_deg - 180)

    center_3d = [center_3d[0], -center_3d[1], center_3d[2]] # 리얼센스는 y좌표계가 반전임 -> 이거로  수정

    return center_3d, obb_dims, tilt_deg, rot_deg

# Lifespan에서 사용할 컨텍스트 정의
@dataclass
class AppContext:
    model_wrapper: GroundedSAM2Wrapper
    camera_capture: RealSenseCapture

# 수정된 Lifespan 관리자 - 싱글톤 패턴 사용
@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """서버 시작 시 모델과 카메라 로드, 종료 시 카메라 해제"""
    print("Initializing resources for MCP server...")
    
    try:
        # 싱글톤 패턴으로 모델과 카메라 인스턴스 가져오기
        model_wrapper = get_model_wrapper()
        camera_capture = get_camera_capture()

        print("Resources initialized successfully.")
        # 초기화된 객체들을 컨텍스트로 전달
        yield AppContext(model_wrapper=model_wrapper, camera_capture=camera_capture)

    finally:
        print("Shutting down resources...")
        # 싱글톤 인스턴스들은 프로세스 종료까지 유지되므로 여기서는 정리하지 않음
        print("Resources shut down.")

# MCP 서버 인스턴스 생성 및 lifespan 연결
mcp = FastMCP("detect_object_mcp_server", lifespan=app_lifespan)

# 로그 파일 이름, 레벨, 포맷 설정
# log_file_path = os.path.join(base_dir, 'mcp_server.log')
# logging.basicConfig(level=logging.INFO,
#                     filename=log_file_path,
#                     filemode='a',
#                     format='%(asctime)s - %(levelname)s - %(message)s')

# MCP 서버 툴 함수 정의
@mcp.tool()
async def find_object_3d_properties(
    object_names: Union[str, List[str]],
    ctx: Context,
    camera_index: int = 0
) -> List[Dict[str, Any]]:
    """
    Detects 3D properties for one or more object names in a single pass.
    Returns a list of result dicts, each containing:
      - object_name, position, dimensions, area, rotation_degree, score (or error)
    """
    # 1. Normalize input to list of names
    if isinstance(object_names, str):
        names = [n.strip() for n in re.split(r"[\.,]", object_names) if n.strip()]
    else:
        names = object_names


    # 3. Acquire resources
    app_context: AppContext = ctx.request_context.lifespan_context
    wrapper = app_context.model_wrapper
    capture = app_context.camera_capture

    # 4. Capture one frame
    if not capture.is_running:
        raise RuntimeError("Camera is not running. Check initialization.")
    color_image, depth_image_raw = await asyncio.to_thread(capture.capture_aligned_frames)
    if color_image is None or depth_image_raw is None:
        raise RuntimeError("Frame capture failed.")
    intrinsics = capture.get_intrinsics()
    depth_scale = capture.get_depth_scale()


    # 5. 각 name마다 predict 호출
    results: List[Dict[str, Any]] = []
    for name in names:
        # 객체 하나만 쿼리
        text_prompt = f"{name}."

        # 같은 color_image 재사용, 프롬프트만 변경
        masks_tensor, boxes_tensor, phrases_list, scores_tensor = await asyncio.to_thread(
            wrapper.predict,
            color_image,
            text_prompt
        )

        # 6. If no detections, return empty list
        if boxes_tensor.size(0) == 0:
            return []

        # 7. Iterate detections and compute 3D props
        for idx in range(boxes_tensor.size(0)):
            mask_np = masks_tensor[idx].cpu().numpy() > 0.27

            center_3d, obb_dims, tilt_deg, rot_deg= await asyncio.to_thread(
                calculate_object_properties,
                mask_np,
                depth_image_raw,
                intrinsics,
                depth_scale
            )

            if center_3d is None or obb_dims is None:
                results.append({
                    "object_name": name,
                    "error": "3D calculation failed",
                    "point_clouds": []  
                })
                continue

            # apply offsets
            x_off = X_OFFSETS[camera_index]
            y_off = Y_OFFSETS[camera_index]
            z_off = Z_OFFSETS[camera_index]
            x = round(center_3d[0], 3) + x_off # x,y좌표 바꿈
            y = round(center_3d[1], 3) + y_off
            z = round(obb_dims[2], 3) + z_off
            L = round(obb_dims[0], 3)
            W = round(obb_dims[1], 3)
            area = round(L * W, 6)
            rotation = round(rot_deg, 2)

            # 필터링: z>600 또는 L,W 중 하나라도 400 초과면 제외 ################################################## 필터링 코드 추가 7_7
            if z > 0.6 or L > 0.5 or W > 0.5:
                continue

            # 성공한 경우, pt와 clr을 Python 리스트로 변환해서 넣기
            results.append({
                "object_name": name,
                "position": [x, y, z],
                "dimensions": [L, W],
                "area": area,
                "rotation_degree": rotation,
            })

    results = wrapper_for_indy(results) # indy 좌표계에 맞게 변환하는 코드


    # 여기서 딕셔너리가 아니라, 튜플로!
    return results


# 화면 전체 포인트 클라우드 반환 코드
@mcp.tool()
async def generate_point_cloud(ctx: Context,camera_index: int = 0):
        # 3. Acquire resources
    app_context: AppContext = ctx.request_context.lifespan_context
    capture = app_context.camera_capture

    # 4. Capture one frame
    if not capture.is_running:
        raise RuntimeError("Camera is not running. Check initialization.")
    color_image, depth_image_raw = await asyncio.to_thread(capture.capture_aligned_frames)
    if color_image is None or depth_image_raw is None:
        raise RuntimeError("Frame capture failed.")
    intrinsics = capture.get_intrinsics()
    depth_scale = capture.get_depth_scale()

    h, w = depth_image_raw.shape
    depth_m = depth_image_raw.astype(np.float32) * depth_scale

    pc_all_list = []
    color_all_list = []

    for y in range(h):
        for x in range(w):
            d = depth_m[y, x]
            if not (0.01 < d < 10.0):
                continue
            X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            pc_all_list.append([X, Y, Z])
            color = color_image[y, x].astype(np.float32) / 255.0
            color_all_list.append(color)

    pc_all     = np.array(pc_all_list, dtype=np.float32)
    colors_all = np.array(color_all_list, dtype=np.float32)
    points_bytes = pc_all.astype(np.float32).tobytes()
    colors_bytes = colors_all.astype(np.float32).tobytes()

    return {
        "points": base64.b64encode(points_bytes).decode(),
        "colors": base64.b64encode(colors_bytes).decode()
    }

    



@mcp.tool()
async def capture_image_as_jpg(ctx: Context, camera_index: int = 0) -> str:
    """base64로 디코딩한 이미지 반환"""
    try:
        app_context: AppContext = ctx.request_context.lifespan_context
        capture = app_context.camera_capture

        if not capture.is_running:
            raise RuntimeError("Camera is not running. It must be started in lifespan.")

        color_image, _ = await asyncio.to_thread(capture.capture_aligned_frames)
        if color_image is None:
            raise RuntimeError("Failed to capture color frame.")

        success, jpg_bytes = cv2.imencode('.jpg', color_image)
        if not success:
            raise RuntimeError("Failed to encode image to JPEG.")

        # 바이트 배열을 base64로 인코딩하여 문자열로 반환
        jpg_base64 = base64.b64encode(jpg_bytes.tobytes()).decode('utf-8')
        
        return f"data:image/jpg;base64,{jpg_base64}"
        
    except Exception as e:
        logging.error(f"Error capturing JPG from camera {camera_index}: {e}", exc_info=True)
        return ''







# ---- 서버 실행 ----
if __name__ == "__main__":
    print("Starting MCP server...")
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n[INFO] MCP server interrupted by user. Shutting down...")
    finally:
        # Ensure global is declared once and before any operations
        # global _camera_capture_instance  # Declare global once before any usage
        if _camera_capture_instance and _camera_capture_instance.is_running:
            _camera_capture_instance.stop()
        print("[INFO] Server shutdown complete.")
#################################################### 원본#########################################3
# # 감지된 물체의 3d 좌표 및 정보 파악
# def calculate_object_properties(mask, depth_image_raw, intrinsics, depth_scale):
#     # 1) 입력 검증
#     if intrinsics is None or depth_scale is None:
#         print("Error: Invalid camera intrinsics or depth scale.")
#         return None, None, None, None
#     if mask.shape != depth_image_raw.shape:
#         print("Error: Mask shape mismatch.")
#         return None, None, None, None
#     if mask.dtype != bool:
#         mask = mask > 0

#     pts_yx = np.argwhere(mask)
#     if pts_yx.size == 0:
#         print("Warning: No mask pixels.")
#         return None, None, None, None

#     # 2) 깊이 → 미터
#     depth_m = depth_image_raw.astype(np.float32) * depth_scale

#     # 3) 왜곡 보정 포함 deproject → 3D 점군
#     pc_list = []
#     for y, x in pts_yx:
#         d = depth_m[y, x]
#         if 0.01 < d < 10.0:
#             X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
#             pc_list.append([X, Y, Z])

#     if len(pc_list) < 10:
#         print("Warning: Too few 3D points.")
#         return None, None, None, None

#     pc = np.array(pc_list, dtype=np.float32)
#     center_3d = pc.mean(axis=0).tolist()


#     # 4) 절대 높이
#     z_vals = pc[:, 2]
#     height = float(z_vals.max() - z_vals.min())

#     # 5) PCA → OBB 축 길이
#     pca = PCA(n_components=3).fit(pc)
#     t_pc = pca.transform(pc)
#     size_obb = (t_pc.max(axis=0) - t_pc.min(axis=0)).tolist()  # [d0,d1,d2]

#     # 6) 높이에 가장 가까운 축 제거 → L, W
#     diffs = [abs(d - height) for d in size_obb]
#     idx_h = int(np.argmin(diffs))
#     lw = [d for i, d in enumerate(size_obb) if i != idx_h]
#     L, W = max(lw), min(lw)
#     obb_dims = [L, W, height]

#     # 7) tilt/rot: 수평면상 가장 긴 축 기준
#     axes = pca.components_
#     proj = np.hypot(axes[:,0], axes[:,1])
#     idx_horiz = int(np.argmax(proj))
#     v_h = axes[idx_horiz]
#     norm_h = np.linalg.norm(v_h)
#     tilt_deg = float(np.degrees(np.arcsin(abs(v_h[2]) / norm_h)))
#     rot_deg  = float(np.degrees(np.arctan2(-v_h[1], v_h[0]))) # y 방향 바꿔서 이것도 바꿈

#     # 만약 부호 -면 +로 바꿔주는 코드
#     if abs(rot_deg) > 90:
#         rot_deg = (rot_deg + 180) if rot_deg < 0 else (rot_deg - 180)

#     center_3d = [center_3d[0], -center_3d[1], center_3d[2]] # 리얼센스는 y좌표계가 반전임 -> 이거로  수정

#     return center_3d, obb_dims, tilt_deg, rot_deg

############################################################################################


# ######################################## 포인트 클라우드와 컬러 추가하는 코드 ##################33
# def calculate_object_properties(mask, depth_image_raw, color_image_raw, intrinsics, depth_scale):
#     # 1) 입력 검증
#     if intrinsics is None or depth_scale is None:
#         print("Error: Invalid camera intrinsics or depth scale.")
#         return None, None, None, None, None, None
#     if mask.shape != depth_image_raw.shape or mask.shape != color_image_raw.shape[:2]:
#         print("Error: Shape mismatch.")
#         return None, None, None, None, None, None
#     if mask.dtype != bool:
#         mask = mask > 0

#     pts_yx = np.argwhere(mask)
#     if pts_yx.size == 0:
#         print("Warning: No mask pixels.")
#         return None, None, None, None, None, None

#     # 2) 깊이 → 미터
#     depth_m = depth_image_raw.astype(np.float32) * depth_scale

#     pc_list = []
#     color_list = []
#     for y, x in pts_yx:
#         d = depth_m[y, x]
#         if 0.01 < d < 10.0:
#             X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
#             pc_list.append([X, Y, Z])
#             color_list.append(color_image_raw[y, x].astype(np.float32) / 255.0)

#     if len(pc_list) < 10:
#         print("Warning: Too few 3D points.")
#         return None, None, None, None, None, None

#     # 3) ndarray 변환
#     pc      = np.array(pc_list, dtype=np.float32)   # (N,3)
#     colors  = np.array(color_list, dtype=np.float32)  # (N,3)

#     # 4) 중심, 높이, OBB 계산 (기존 코드와 동일)
#     center_3d = pc.mean(axis=0).tolist()
#     z_vals    = pc[:, 2]
#     height    = float(z_vals.max() - z_vals.min())

#     pca     = PCA(n_components=3).fit(pc)
#     t_pc    = pca.transform(pc)
#     size_obb = (t_pc.max(axis=0) - t_pc.min(axis=0)).tolist()
#     diffs    = [abs(d - height) for d in size_obb]
#     idx_h    = int(np.argmin(diffs))
#     lw       = [d for i, d in enumerate(size_obb) if i != idx_h]
#     L, W     = max(lw), min(lw)
#     obb_dims = [L, W, height]

#     axes      = pca.components_
#     proj      = np.hypot(axes[:,0], axes[:,1])
#     idx_horiz = int(np.argmax(proj))
#     v_h       = axes[idx_horiz]
#     norm_h    = np.linalg.norm(v_h)
#     tilt_deg  = float(np.degrees(np.arcsin(abs(v_h[2]) / norm_h)))
#     rot_deg   = float(np.degrees(np.arctan2(-v_h[1], v_h[0])))
#     if abs(rot_deg) > 90:
#         rot_deg = (rot_deg + 180) if rot_deg < 0 else (rot_deg - 180)

#     center_3d = [center_3d[0], -center_3d[1], center_3d[2]]

#     # 5) 반환값에 point cloud 추가
#     return center_3d, obb_dims, tilt_deg, rot_deg, pc, colors
