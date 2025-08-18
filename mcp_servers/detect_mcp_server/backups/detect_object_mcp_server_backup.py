import os
import numpy as np
import torch
import pyrealsense2 as rs
import sys

# GROUND_SAM2_PATH = os.path.join(os.path.dirname(__file__), "Grounded-SAM-2")
# sys.path.append(GROUND_SAM2_PATH)

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.build_sam import build_sam2 # GroundedSAM2 모듈
from sam2.sam2_image_predictor import SAM2ImagePredictor # GroundedSAM2 모듈
from sklearn.decomposition import PCA
from typing import Optional, Dict, Any, List # Type Hinting 추가
from mcp.server.fastmcp import FastMCP, Context  # MCP 관련 import
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import asyncio 
import time
import cv2
import logging

# 현재 스크립트 파일 기준 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)

# 각 축(W, L, H)에 대한 크기 보정 오프셋 (단위: 미터)
X_OFFSETS = [0.48, 0.1, -0.05]       # 인덱스별 X 오프셋
Y_OFFSETS = [0.0, 0.6, 0.55]       # 인덱스별 Y 오프셋
Z_OFFSETS = [0, -0.01, -0.03]   # 인덱스별 Height 오프셋

# 전역 싱글톤 인스턴스들
_model_wrapper_instance = None
_camera_capture_instance = None

# 연결된 카메라 목록 확인
def list_connected_realsense_devices():
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]

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
        box_threshold: float = 0.43, # 박스 감지 임계값 -> 낮을수록 더 느슨하게 감지
        text_threshold: float = 0.43, # 텍스트 감지 임계값 -> 낮을수록 더 느슨하게 감지
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
                for _ in range(15): # 안정화 시간 필요
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

# 감지된 물체의 3d 좌표 및 정보 파악
def calculate_object_properties(
    mask: np.ndarray,
    depth_image_raw: np.ndarray,
    intrinsics: rs.intrinsics,
    depth_scale: float
):
    if not intrinsics or depth_scale is None:
        print("Error: Invalid camera intrinsics or depth scale.")
        return None, None
    if mask.shape != depth_image_raw.shape:
        print(f"Error: Mask shape {mask.shape} and depth shape {depth_image_raw.shape} mismatch.")
        return None, None
    if mask.dtype != bool:
        mask = mask > 0

    object_pixels_yx = np.argwhere(mask)
    if object_pixels_yx.size == 0:
        print("Warning: No pixels found in the mask.")
        return None, None

    depth_image_meters = depth_image_raw.astype(np.float32) * depth_scale
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy

    point_cloud = []
    try:
        for y, x in object_pixels_yx:
            depth = depth_image_meters[y, x]
            if depth > 0.01 and depth < 10.0: # 유효 깊이 범위
                Z_pt = depth
                X_pt = (x - cx) * Z_pt / fx
                Y_pt = (y - cy) * Z_pt / fy
                point_cloud.append([X_pt, Y_pt, Z_pt])

        if len(point_cloud) < 10:
            print(f"Warning: Insufficient valid points ({len(point_cloud)}) for 3D calculations.")
            return None, None

        point_cloud_np = np.array(point_cloud, dtype=np.float32)
        center_3d = np.mean(point_cloud_np, axis=0).tolist()

        # PCA로 OBB 크기와 주성분 벡터 계산
        pca = PCA(n_components=3)
        pca.fit(point_cloud_np)

        # OBB 크기 (L, W, H)
        transformed = pca.transform(point_cloud_np)
        min_c = np.min(transformed, axis=0)
        max_c = np.max(transformed, axis=0)
        size_obb = max_c - min_c
        obb_dims = sorted(size_obb.tolist(), reverse=True)

        # 주성분 벡터로 tilt/rotation 계산
        v = pca.components_[0]         # [v_x, v_y, v_z]
        norm = np.linalg.norm(v)
        tilt_rad = np.arccos(abs(v[2]) / norm)
        tilt_deg = np.degrees(tilt_rad)      # 수평면 대비 기울기
        rot_rad = np.arctan2(v[1], v[0])
        rot_deg = np.degrees(rot_rad)        # 평면 내 회전

        # 결과 반환
        return center_3d, obb_dims, float(tilt_deg), float(rot_deg)

    except Exception as e:
        print(f"Error during 3D property calculation: {e}")
        return None, None

# ---- MCP 서버 설정 시작 ----

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
log_file_path = os.path.join(base_dir, 'mcp_server.log')
logging.basicConfig(level=logging.INFO,
                    filename=log_file_path,
                    filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# MCP 서버 툴 함수 정의
@mcp.tool()
async def find_object_3d_properties(object_name: str, ctx: Context, camera_index: int = 0 ) -> Dict[str, Any]:
    logging.info(f"Tool called for '{object_name}' with camera index {camera_index}")

    result = {
        "object_name": None,
        "position": None,
        "dimensions": None,
        "area": None
    }

    app_context: AppContext = ctx.request_context.lifespan_context
    wrapper = app_context.model_wrapper
    capture = app_context.camera_capture

    try:
        if not capture.is_running:
            raise RuntimeError("Camera is not running. Check initialization.")

        color_image, depth_image_raw = await asyncio.to_thread(
            capture.capture_aligned_frames
        )
        if color_image is None or depth_image_raw is None:
            raise RuntimeError("Frame capture failed.")

        intrinsics = capture.get_intrinsics()
        depth_scale = capture.get_depth_scale()

        masks_tensor, boxes_tensor, phrases_list, scores_tensor = await asyncio.to_thread(
            wrapper.predict, color_image, object_name
        )

        if not phrases_list:
            result["error"] = f"Object '{object_name}' not found."
        else:
            best_match_idx = np.argmax(scores_tensor.cpu().numpy())
            selected_phrase = phrases_list[best_match_idx]
            selected_mask_np = masks_tensor[best_match_idx].cpu().numpy() > 0.5

            result["object_name"] = selected_phrase

            center_3d_coords, obb_dims_LWH, tilt_deg, rot_deg = await asyncio.to_thread(
                calculate_object_properties,
                selected_mask_np,
                depth_image_raw,
                intrinsics,
                depth_scale
            )

            if center_3d_coords and obb_dims_LWH:
                x_offset = X_OFFSETS[camera_index]
                y_offset = Y_OFFSETS[camera_index]
                z_offset = Z_OFFSETS[camera_index]

                calculated_x = round(float(center_3d_coords[1]), 3) + x_offset
                calculated_y = round(float(center_3d_coords[0]), 3) + y_offset
                calculated_z = round(float(obb_dims_LWH[2]) + z_offset, 3)

                result["position"] = [calculated_x, calculated_y, calculated_z]
                result["dimensions"] = [round(dim, 3) for dim in obb_dims_LWH[:2]] # L, W (H는 제외)
                
                length = round(obb_dims_LWH[0], 3)  # Rounded L
                width = round(obb_dims_LWH[1], 3)   # Rounded W
                area = round(length * width, 6)  # No rounding for area
                # 2) tilt & rotation 각도 추가
                # result["tilt_degree"]     = round(tilt_deg, 2)   # 수평면 대비 기울기
                result["rotation_degree"] = round(rot_deg, 2)    # 평면 내 회전정도
                result["area"] = area  # Store the area (L * W)

    except Exception as e:
        logging.error(f"Error in tool: {e}", exc_info=True)
        result["error"] = str(e)

    return result

import base64

@mcp.tool()
async def capture_image_as_jpg(ctx: Context, camera_index: int = 0) -> str:
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
