import numpy as np
import torch
import pyrealsense2 as rs
torch.cuda.empty_cache()
import base64
import zlib


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

from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import open3d as o3d

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
        dino_model_id="IDEA-Research/grounding-dino-tiny",
        sam2_config_path="configs/sam2.1/sam2.1_hiera_t.yaml", # 경로 오류 뜨면 Grounded-SAM-2 폴더 내부 위치 넣기
        sam2_ckpt_path = "Grounded-SAM-2/checkpoints/sam2.1_hiera_tiny.pt", # 경로 오류 뜨면 Grounded-SAM-2 폴더 내부 위치 넣기
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

        with torch.no_grad():
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
        if 0.01 < d < 0.8:
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



def encode_mask(mask: np.ndarray) -> str:
    """
    Compress and encode a binary mask into a base64 string.
    """
    compressed = zlib.compress(mask.astype(np.uint8).tobytes())
    return base64.b64encode(compressed).decode('utf-8')


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
        # masks_tensor, boxes_tensor, phrases_list, scores_tensor = await asyncio.to_thread(
        #     wrapper.predict,
        #     color_image,
        #     text_prompt
        # )
        masks_tensor, boxes_tensor, phrases_list, scores_tensor = await asyncio.to_thread(
            wrapper.predict,
            color_image,
            text_prompt
        )
        if boxes_tensor.size(0) == 0:
            return []
        # 필요한 만큼만 .cpu() 후 GPU 텐서 제거
        masks_tensor_cpu = masks_tensor.detach().cpu()
        boxes_tensor_cpu = boxes_tensor.detach().cpu()
        scores_tensor_cpu = scores_tensor.detach().cpu()

        # GPU 텐서 명시적 제거
        del masks_tensor, boxes_tensor, scores_tensor
        torch.cuda.empty_cache()

        # 이후는 CPU 텐서 기준으로 사용
        for idx in range(boxes_tensor_cpu.size(0)):
            mask_np = masks_tensor_cpu[idx].numpy() > 0.27
        # # 6. If no detections, return empty list
        # if boxes_tensor.size(0) == 0:
        #     return []

        # # 7. Iterate detections and compute 3D props
        # for idx in range(boxes_tensor.size(0)):
        #     mask_np = masks_tensor[idx].cpu().numpy() > 0.27

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
            if z > 0.6 or L > 0.5 or W > 0.15: # 수정 (W : 단축  이 15cm 넘으면 제외)
                continue
            if area >= 0.2 or area  <= 0.001: 
                continue

            results.append({
                "object_name": name,
                "position": [x, y, z],
                "dimensions": [L, W],
                "area": area,
                "rotation_degree": rotation,
                "mask_base64": encode_mask(mask_np),
                "mask_shape": list(mask_np.shape)  # ← 추가
            })
    results = wrapper_for_indy(results) # indy 좌표계에 맞게 변환하는 코드


    # 여기서 딕셔너리가 아니라, 튜플로!
    return results




def decode_mask(mask_base64: str, shape: tuple) -> np.ndarray:
    """
    Decode base64-encoded and zlib-compressed binary mask back to numpy array.
    
    Parameters:
        mask_base64 (str): The base64-encoded mask string.
        shape (tuple): The original (H, W) shape of the mask.
        
    Returns:
        np.ndarray: Decoded binary mask as a boolean numpy array.
    """
    # 1. base64 디코딩
    compressed = base64.b64decode(mask_base64)
    
    # 2. zlib 압축 해제
    decompressed = zlib.decompress(compressed)
    
    # 3. numpy 배열로 변환 및 bool mask로 변경
    mask_flat = np.frombuffer(decompressed, dtype=np.uint8)
    return mask_flat.reshape(shape).astype(bool)




def generate_side_point_cloud(cluster_pts: np.ndarray,
                              num_edge_samples: int = 30,
                              num_height_samples: int = 30,
                              height_range: float = 0.1) -> np.ndarray:
    """
    cluster_pts: (N,3) 해당 클러스터의 원본 포인트
    height_range: 최상단에서 아래로 샘플링할 높이 (m)
    """
    # 1) XY 평면에서 Convex Hull (윗면 윤곽)
    pts2d = cluster_pts[:, :2]
    hull = ConvexHull(pts2d)
    hull_xy = pts2d[hull.vertices]

    # 2) 높이 범위: 물체 최상단(Z_max) ~ Z_max - height_range
    # z_max = cluster_pts[:, 2].max()
    # z_min = z_max - height_range

    # z_min = cluster_pts[:, 2].min()
    zs = cluster_pts[:, 2]
    # 20번째로 작은 Z를 바닥 높이로 사용
    z_min = np.partition(zs, 100)[100]
    z_max =  z_min + height_range

    # 3) 옆면 샘플링
    side_pts = []
    for i in range(len(hull_xy)):
        p1 = hull_xy[i]
        p2 = hull_xy[(i+1)%len(hull_xy)]
        # 변을 따라 샘플링
        for t in np.linspace(0, 1, num_edge_samples):
            x, y = p1*(1-t) + p2*t
            # 높이 방향으로 샘플링
            for z in np.linspace(z_max, z_min, num_height_samples):
                side_pts.append([x, y, z])

    return np.array(side_pts)

def cluster_and_generate_surface(pts: np.ndarray,
                                 cols: np.ndarray,
                                 eps: float = 0.005,
                                 min_samples: int = 10,
                                 min_volume: float = 1e-6,
                                 obj_height: float = 0.1
                                ) -> o3d.geometry.PointCloud:
    """
    마스크된 포인트 클라우드를 받아 클러스터링 후 벽면 보간하고 병합된 포인트 클라우드를 리턴
    """

    # ====== Z 필터링 먼저 ======
    zs = pts[:, 2]
    if len(zs) < 100:
        print("Too few points for Z filtering.")
        return o3d.geometry.PointCloud()
    z_ref = np.partition(zs, 100)[100]
    mask_z = np.abs(zs - z_ref) < 0.01
    pts = pts[mask_z]
    cols = cols[mask_z]
    print(f"[Z-Filter] Kept {len(pts)} points near Z={z_ref:.4f}")

    # DBSCAN 클러스터링 (CPU)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pts)
    max_label = labels.max()
    print(f"[DBSCAN] Detected {max_label + 1} clusters (+ noise)")

    def compute_volume(arr):
        return ConvexHull(arr).volume if arr.shape[0] >= 4 else 0.0

    merged_pts = []
    merged_cols = []
    merged_sides = []

    for lbl in range(max_label + 1):
        mask = labels == lbl
        if not np.any(mask):
            continue

        cluster_pts = pts[mask]
        cluster_cols = cols[mask]
        
        if len(cluster_pts) < 4:
            print("  → Skipped (not enough points after Z filtering)")
            continue
        vol = compute_volume(cluster_pts)
        print(f" Cluster {lbl}: Volume = {vol:.3e} m³")
        if vol < min_volume:
            print("  → Skipped (too small)")
            continue

        # 벽면 보간
        side_pts = generate_side_point_cloud(cluster_pts,
                                             num_edge_samples=30,
                                             num_height_samples=30,
                                             height_range = obj_height)

        merged_pts.append(cluster_pts)
        merged_cols.append(cluster_cols)
        merged_sides.append(side_pts)

    if not merged_pts:
        print("No valid clusters found.")
        return o3d.geometry.PointCloud()

    all_pts = np.vstack(merged_pts)
    all_cols = np.vstack(merged_cols)
    all_sides = np.vstack(merged_sides) if merged_sides else np.empty((0, 3))

    # 병합된 포인트 클라우드 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([all_pts, all_sides]))
    pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([all_cols, np.tile([[0.8, 0.2, 0.2]], (all_sides.shape[0], 1))])
    )

    return pcd




@mcp.tool()
async def generate_completed_point_cloud(
    ctx: Context,
    mask_base64: Optional[str] = None,
    mask_shape: Optional[List[int]] = None,
    camera_index: int = 0,
    eps: float = 0.005,
    min_samples: int = 10,
    min_volume: float = 1e-6,
    obj_height: float = 0.1
):
    """
    GroundedSAM2로 마스킹 된 영역만 받아와서 옆면 보간한 후 모든 맵과 합친 후 밀도 낮춰서 반환
    """
    # 1. 리소스 획득
    app_context: AppContext = ctx.request_context.lifespan_context
    capture = app_context.camera_capture
    color_image, depth_image_raw = await asyncio.to_thread(capture.capture_aligned_frames)
    intrinsics = capture.get_intrinsics()
    depth_scale = capture.get_depth_scale()

    h, w = depth_image_raw.shape
    depth_m = depth_image_raw.astype(np.float32) * depth_scale

    # 2. 마스크 복원
    use_mask = mask_base64 is not None
    mask_np = decode_mask(mask_base64, tuple(mask_shape)) if use_mask else None

    pc_all, color_all = [], []
    pc_masked, color_masked = [], []

    # 3. 전체 포인트 클라우드 생성 & 마스크된 포인트 분리
    for y in range(h):
        for x in range(w):
            d = depth_m[y, x]
            if not (0.01 < d < 0.8): continue

            X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            color = color_image[y, x].astype(np.float32) / 255.0

            pt = [X, Y, Z]
            pc_all.append(pt)
            color_all.append(color)

            if use_mask and mask_np[y, x]:
                pc_masked.append(pt)
                color_masked.append(color)

    # 전체 클라우드
    pc_all_np = np.array(pc_all, dtype=np.float32)
    color_all_np = np.array(color_all, dtype=np.float32)
    # sam으로 마스크 된 클라우드
    pc_masked_np = np.array(pc_masked, dtype=np.float32)
    color_masked_np = np.array(color_masked, dtype=np.float32)
################################# 전체 클라우드는  감지된 물체 중심 기준으로 좌우 30cm만 남김
    x_vals = pc_all_np[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    x_center = (x_min + x_max) / 2
    margin = 0.6  # 60cm

    mask_center = (x_vals >= x_center - margin) & (x_vals <= x_center + margin)
    pc_all_np = pc_all_np[mask_center]
    color_all_np = color_all_np[mask_center]
###############################################################################



    # 4. 클러스터링 + 보간
    print(f"[INFO] Masked points: {len(pc_masked_np)}")
    if len(pc_masked_np) >= min_samples:
        pcd_additional = cluster_and_generate_surface(
            pts=pc_masked_np,
            cols=color_masked_np,
            eps=eps,
            min_samples=min_samples,
            min_volume=min_volume,
            obj_height = obj_height
        )

        # 5. 병합
        if pcd_additional is not None and len(pcd_additional.points) > 0:
            additional_pts = np.asarray(pcd_additional.points)
            additional_cols = np.asarray(pcd_additional.colors)

            pc_merged = np.vstack([pc_all_np, additional_pts])
            col_merged = np.vstack([color_all_np, additional_cols])
        else:
            print("[INFO] No valid clusters passed volume filtering.")
            pc_merged = pc_all_np
            col_merged = color_all_np
    else:
        print("[INFO] Not enough masked points for clustering.")
        pc_merged = pc_all_np
        col_merged = color_all_np

    # 중복 포인트 병합
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_merged)
    pcd.colors = o3d.utility.Vector3dVector(col_merged)
    pcd = pcd.voxel_down_sample(voxel_size=0.003) # x mm당 점 1개씩 반환

    pc_merged = np.asarray(pcd.points)
    col_merged = np.asarray(pcd.colors)
    return {
        "points": base64.b64encode(pc_merged.astype(np.float32).tobytes()).decode(),
        "colors": base64.b64encode(col_merged.astype(np.float32).tobytes()).decode()
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



# @mcp.tool()
# async def generate_completed_point_cloud(
#     ctx: Context,
#     mask_base64_list: Optional[List[str]] = None,
#     mask_shape_list: Optional[List[List[int]]] = None,
#     obj_height_list: Optional[List[float]] = None,
#     camera_index: int = 0,
#     eps: float = 0.005,
#     min_samples: int = 10,
#     min_volume: float = 1e-6,
# ):
#     """
#     GroundedSAM2로 마스킹 된 영역만 받아와서 옆면 보간한 후 모든 맵과 합친 후 밀도 낮춰서 반환
#     """
#     # 단일값도 리스트로 처리 가능하도록 변환
#     if isinstance(mask_base64_list, str):
#         mask_base64_list = [mask_base64_list]
#     if isinstance(mask_shape_list, (list, tuple)) and isinstance(mask_shape_list[0], int):
#         mask_shape_list = [mask_shape_list]
#     if isinstance(obj_height_list, (float, int)):
#         obj_height_list = [obj_height_list]

#     if mask_base64_list and mask_shape_list and obj_height_list:
#         if not (len(mask_base64_list) == len(mask_shape_list) == len(obj_height_list)):
#             raise ValueError("mask_base64_list, mask_shape_list, and obj_height_list must all be the same length.")

#     # 1. 리소스 획득
#     app_context: AppContext = ctx.request_context.lifespan_context
#     capture = app_context.camera_capture
#     color_image, depth_image_raw = await asyncio.to_thread(capture.capture_aligned_frames)
#     intrinsics = capture.get_intrinsics()
#     depth_scale = capture.get_depth_scale()

#     h, w = depth_image_raw.shape
#     depth_m = depth_image_raw.astype(np.float32) * depth_scale

#     # 2. 전체 포인트 클라우드 생성
#     pc_all = []
#     color_all = []
#     for y in range(h):
#         for x in range(w):
#             d = depth_m[y, x]
#             if not (0.01 < d < 0.8): continue
#             X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
#             color = color_image[y, x].astype(np.float32) / 255.0
#             pc_all.append([X, Y, Z])
#             color_all.append(color)

#     pc_all_np = np.array(pc_all, dtype=np.float32)
#     color_all_np = np.array(color_all, dtype=np.float32)

#     # 3. 마스크 기반 포인트 분리
#     masked_pc_list = []
#     masked_color_list = []
#     if mask_base64_list and mask_shape_list and obj_height_list:
#         for i in range(len(mask_base64_list)):
#             mask_np = decode_mask(mask_base64_list[i], tuple(mask_shape_list[i]))
#             pc_masked = []
#             color_masked = []
#             for y in range(h):
#                 for x in range(w):
#                     if not mask_np[y, x]:
#                         continue
#                     d = depth_m[y, x]
#                     if not (0.01 < d < 0.8): continue
#                     X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
#                     color = color_image[y, x].astype(np.float32) / 255.0
#                     pc_masked.append([X, Y, Z])
#                     color_masked.append(color)
#             if len(pc_masked) > 0:
#                 masked_pc_list.append(np.array(pc_masked, dtype=np.float32))
#                 masked_color_list.append(np.array(color_masked, dtype=np.float32))

#     # 4. 보간 실행
#     pc_additional_list = []
#     col_additional_list = []
#     for i in range(len(masked_pc_list)):
#         pc_masked_np = masked_pc_list[i]
#         col_masked_np = masked_color_list[i]
#         print(f"[INFO] Mask {i}: {len(pc_masked_np)} masked points")

#         if len(pc_masked_np) < min_samples:
#             print("  → Skipped (too few points)")
#             continue

#         pcd_add = cluster_and_generate_surface(
#             pts=pc_masked_np,
#             cols=col_masked_np,
#             eps=eps,
#             min_samples=min_samples,
#             min_volume=min_volume,
#             obj_height=obj_height_list[i]
#         )

#         if pcd_add is not None and len(pcd_add.points) > 0:
#             pc_additional_list.append(np.asarray(pcd_add.points))
#             col_additional_list.append(np.asarray(pcd_add.colors))
#         else:
#             print("  → Skipped (no valid cluster)")

#     # 5. 중심 기준 X 필터링
#     x_vals = pc_all_np[:, 0]
#     x_center = (x_vals.min() + x_vals.max()) / 2
#     margin = 0.6
#     mask_center = (x_vals >= x_center - margin) & (x_vals <= x_center + margin)
#     pc_all_np = pc_all_np[mask_center]
#     color_all_np = color_all_np[mask_center]

#     # 6. 병합
#     if pc_additional_list:
#         pc_add_np = np.vstack(pc_additional_list)
#         col_add_np = np.vstack(col_additional_list)
#         pc_merged = np.vstack([pc_all_np, pc_add_np])
#         col_merged = np.vstack([color_all_np, col_add_np])
#     else:
#         print("[INFO] No valid additional surfaces from masks.")
#         pc_merged = pc_all_np
#         col_merged = color_all_np

#     # 7. 다운샘플
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(pc_merged)
#     pcd.colors = o3d.utility.Vector3dVector(col_merged)
#     pcd = pcd.voxel_down_sample(voxel_size=0.003)

#     # 8. base64 인코딩 후 반환
#     pc_merged = np.asarray(pcd.points)
#     col_merged = np.asarray(pcd.colors)
#     return {
#         "points": base64.b64encode(pc_merged.astype(np.float32).tobytes()).decode(),
#         "colors": base64.b64encode(col_merged.astype(np.float32).tobytes()).decode()
#     }