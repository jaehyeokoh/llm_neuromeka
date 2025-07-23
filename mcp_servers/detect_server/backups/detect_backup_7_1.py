import os
import numpy as np
import torch
import pyrealsense2 as rs
import sys
torch.cuda.empty_cache()

GROUND_SAM2_PATH = os.path.join(os.path.dirname(__file__), "Grounded-SAM-2")
sys.path.append(GROUND_SAM2_PATH)

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
import re
from typing import Union
# 현재 스크립트 파일 기준 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
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

# # 감지된 물체의 3d 좌표 및 정보 파악
# def calculate_object_properties(
#     mask: np.ndarray,
#     depth_image_raw: np.ndarray,
#     intrinsics: rs.intrinsics,
#     depth_scale: float
# ):
#     if not intrinsics or depth_scale is None:
#         print("Error: Invalid camera intrinsics or depth scale.")
#         return None, None
#     if mask.shape != depth_image_raw.shape:
#         print(f"Error: Mask shape {mask.shape} and depth shape {depth_image_raw.shape} mismatch.")
#         return None, None
#     if mask.dtype != bool:
#         mask = mask > 0

#     object_pixels_yx = np.argwhere(mask)
#     if object_pixels_yx.size == 0:
#         print("Warning: No pixels found in the mask.")
#         return None, None

#     depth_image_meters = depth_image_raw.astype(np.float32) * depth_scale
#     fx, fy = intrinsics.fx, intrinsics.fy
#     cx, cy = intrinsics.ppx, intrinsics.ppy

#     point_cloud = []
#     # try:
#     #     for y, x in object_pixels_yx:
#     #         depth = depth_image_meters[y, x]
#     #         if depth > 0.01 and depth < 10.0: # 유효 깊이 범위
#     #             Z_pt = depth
#     #             X_pt = (x - cx) * Z_pt / fx
#     #             Y_pt = (y - cy) * Z_pt / fy
#     #             point_cloud.append([X_pt, Y_pt, Z_pt])
#     try:
#         for y, x in object_pixels_yx:
#             d = depth_image_meters[y, x]
#             if 0.01 < d < 20.0:  # 유효 깊이 범위
#                 # 왜곡 보정 포함 deproject
#                 X_pt, Y_pt, Z_pt = rs.rs2_deproject_pixel_to_point(
#                     intrinsics, [x, y], d
#                 )
#                 point_cloud.append([X_pt, Y_pt, Z_pt])

#         if len(point_cloud) < 10:
#             print(f"Warning: Insufficient valid points ({len(point_cloud)}) for 3D calculations.")
#             return None, None

#         point_cloud_np = np.array(point_cloud, dtype=np.float32)

#         # 1) Z값으로 슬라이스: 캔 윗면 근처의 점들만 골라냄
#         z_vals = point_cloud_np[:, 2]
#         z_top = z_vals.max()
#         eps = 0.01  # 두께 5 mm 허용 오차
#         top_slice = point_cloud_np[np.abs(z_vals - z_top) < eps]
#         pca2 = PCA(n_components=2)
#         pts2 = pca2.fit_transform(top_slice[:, :2])
#         # 3) 주성분 축 기준으로 최소·최대값 차이 계산
#         min2, max2 = pts2.min(axis=0), pts2.max(axis=0)
#         diam1, diam2 = (max2 - min2).tolist()  # 단면의 두 축 지름
#         length = float(diam1)
#         width  = float(diam2)

#         center_3d = np.mean(point_cloud_np, axis=0).tolist()

#         # # PCA로 OBB 크기와 주성분 벡터 계산
#         pca = PCA(n_components=3)
#         pca.fit(point_cloud_np)

#         # # OBB 크기 (L, W, H)
#         # transformed = pca.transform(point_cloud_np)
#         # min_c = np.min(transformed, axis=0)
#         # max_c = np.max(transformed, axis=0)
#         # size_obb = max_c - min_c
#         # obb_dims = sorted(size_obb.tolist(), reverse=True) # 이거 이상함 그냥 크기순으로 나열한 거였음
#         # 1) 절대 높이 계산 (카메라 Z축 기준)
#         z_vals = point_cloud_np[:, 2]
#         height = float(z_vals.max() - z_vals.min())

#         # # 2) XY 평면에서만 PCA로 길이·폭 계산
#         # pca_xy = PCA(n_components=2)
#         # xy_transformed = pca_xy.fit_transform(point_cloud_np[:, :2])
#         # min_xy = np.min(xy_transformed, axis=0)
#         # max_xy = np.max(xy_transformed, axis=0)
#         # length, width = (max_xy - min_xy).tolist()
#         # length = float(length)
#         # width  = float(width)

#         # 3) 의미가 명확한 딕셔너리 형태로 저장
#         obb_dims = [length, width, height]
#         # 주성분 벡터로 tilt/rotation 계산
#         v = pca.components_[0]         # [v_x, v_y, v_z]
#         norm = np.linalg.norm(v)
#         tilt_rad = np.arccos(abs(v[2]) / norm)
#         tilt_deg = np.degrees(tilt_rad)      # 수평면 대비 기울기
#         rot_rad = np.arctan2(v[1], v[0])
#         rot_deg = np.degrees(rot_rad)        # 평면 내 회전

#         # 결과 반환
#         return center_3d, obb_dims, float(tilt_deg), float(rot_deg)

#     except Exception as e:
#         print(f"Error during 3D property calculation: {e}")
#         return None, None

# #############################################각도 수정#######################################################
# def calculate_object_properties(
#     mask: np.ndarray,
#     depth_image_raw: np.ndarray,
#     intrinsics: rs.intrinsics,
#     depth_scale: float
# ):
    
#     # 1) 입력 검증
#     if intrinsics is None or depth_scale is None:
#         print("Error: Invalid camera intrinsics or depth scale.")
#         return None, None, None, None
#     if mask.shape != depth_image_raw.shape:
#         print(f"Error: Mask shape {mask.shape} and depth shape {depth_image_raw.shape} mismatch.")
#         return None, None, None, None
#     if mask.dtype != bool:
#         mask = mask > 0

#     # 2) 마스크된 픽셀 좌표
#     object_pixels_yx = np.argwhere(mask)
#     if object_pixels_yx.size == 0:
#         print("Warning: No pixels found in the mask.")
#         return None, None, None, None

#     # 3) 깊이→미터 변환
#     depth_m = depth_image_raw.astype(np.float32) * depth_scale

#     # 4) 왜곡 보정 포함 deproject → 3D 점군 생성
#     point_cloud = []
#     for y, x in object_pixels_yx:
#         d = depth_m[y, x]
#         if 0.01 < d < 20.0:
#             X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
#             point_cloud.append([X, Y, Z])

#     if len(point_cloud) < 10:
#         print(f"Warning: Insufficient valid points ({len(point_cloud)}) for 3D calculations.")
#         return None, None, None, None

#     pc = np.array(point_cloud, dtype=np.float32)
#     center_3d = pc.mean(axis=0).tolist()

#     # 5) 3D PCA로 물체 주축과 단면 평면 법선 추정
#     pca3    = PCA(n_components=3).fit(pc)
#     axes3   = pca3.components_     # [axis0, axis1, axis2(normal)]
#     normal  = axes3[2]
#     centroid = pc.mean(axis=0)

#     # 6) 평면 거리 계산 → “윗면” 점들 슬라이스
#     dists   = np.dot(pc - centroid, normal)
#     eps     = 0.002  # 평면 두께 허용치 (3mm)
#     slice_pts = pc[np.abs(dists) < eps]
#     if len(slice_pts) < 5:
#         print(f"Warning: Too few points in top-slice ({len(slice_pts)})")
#         return center_3d, None, None, None

#     # 7) 평면 내 두 축(가로·세로)으로 투영
#     coords2 = np.dot(slice_pts - centroid, axes3[:2].T)  # M×2

#     # 8) 각 축별 길이 계산
#     width  = float(coords2[:,0].max() - coords2[:,0].min())
#     length = float(coords2[:,1].max() - coords2[:,1].min())

#     # 9) 높이 계산 (Z축 절대 높이)
#     z_vals = pc[:,2]
#     height = float(z_vals.max() - z_vals.min())

#     obb_dims = [length, width, height]

#     # 각도 계산
#     h_proj = np.hypot(axes3[:, 0], axes3[:, 1])  # [√(v0x²+v0y²), √(v1x²+v1y²), √(v2x²+v2y²)]
#     idx_h = int(np.argmax(h_proj))               # 수평면에서 가장 길게 퍼진 축 인덱스
#     v_h = axes3[idx_h]                           # 그 축 벡터
#     norm_h = np.linalg.norm(v_h)

#     # 11) tilt 계산 (0°=완전 수평, 90°=완전 수직)
#     tilt_rad = np.arcsin(abs(v_h[2]) / norm_h)
#     tilt_deg = float(np.degrees(tilt_rad))

#     # 12) rot 계산 (XY 평면에서의 방향)
#     rot_rad = np.arctan2(v_h[1], v_h[0])
#     rot_deg = float(np.degrees(rot_rad))

#     return center_3d, obb_dims, tilt_deg, rot_deg

#################################################### 원본#########################################3
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



# def calculate_object_properties(
#     mask: np.ndarray,
#     depth_image_raw: np.ndarray,
#     intrinsics: rs.intrinsics,
#     depth_scale: float
# ):
#     if not intrinsics or depth_scale is None:
#         print("Error: Invalid camera intrinsics or depth scale.")
#         return None, None
#     if mask.shape != depth_image_raw.shape:
#         print(f"Error: Mask shape {mask.shape} and depth shape {depth_image_raw.shape} mismatch.")
#         return None, None
#     if mask.dtype != bool:
#         mask = mask > 0

#     object_pixels_yx = np.argwhere(mask)
#     if object_pixels_yx.size == 0:
#         print("Warning: No pixels found in the mask.")
#         return None, None

#     # depth → meter
#     depth_image_meters = depth_image_raw.astype(np.float32) * depth_scale
#     fx, fy = intrinsics.fx, intrinsics.fy
#     cx, cy = intrinsics.ppx, intrinsics.ppy

#     # 1) 카메라 좌표계에서 3D 점수집
#     point_cloud = []
#     for y, x in object_pixels_yx:
#         z = depth_image_meters[y, x]
#         if 0.01 < z < 10.0:
#             X = (x - cx) * z / fx
#             Y = (y - cy) * z / fy
#             point_cloud.append([X, Y, z])

#     if len(point_cloud) < 10:
#         print(f"Warning: Insufficient valid points ({len(point_cloud)}) for 3D calculations.")
#         return None, None

#     point_cloud_np = np.array(point_cloud, dtype=np.float32)  # (N,3)

#     # 2) 카메라 틸트 보정 회전행렬 (X축 기준 피치)
#     theta = np.radians(camera_tilt_theta)
#     R_tilt = np.array([
#         [1,           0,            0],
#         [0,  np.cos(theta), -np.sin(theta)],
#         [0,  np.sin(theta),  np.cos(theta)]
#     ], dtype=np.float32)

#     # 3) 월드 기준(틸트 보정) 포인트 클라우드
#     world_pc = (R_tilt @ point_cloud_np.T).T  # (N,3)

#     # 4) 중심점 계산 (world frame)
#     center_world = np.mean(world_pc, axis=0).tolist()

#     # 5) PCA로 OBB 크기 및 주축 계산
#     pca = PCA(n_components=3)
#     pca.fit(world_pc)
#     transformed = pca.transform(world_pc)
#     min_c, max_c = np.min(transformed, axis=0), np.max(transformed, axis=0)
#     size_obb = max_c - min_c
#     obb_dims = sorted(size_obb.tolist(), reverse=True)

#     # 6) world frame 주성분 벡터로 tilt/rotation 계산
#     v = pca.components_[0]
#     norm = np.linalg.norm(v)
#     tilt_rad = np.arccos(abs(v[2]) / norm)
#     tilt_deg = float(np.degrees(tilt_rad))
#     rot_rad = np.arctan2(v[1], v[0])
#     rot_deg = float(np.degrees(rot_rad))

#     return center_world, obb_dims, tilt_deg, rot_deg
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
        scores_np = scores_tensor.cpu().numpy()
        for idx in range(boxes_tensor.size(0)):
            phrase = phrases_list[idx]
            mask_np = masks_tensor[idx].cpu().numpy() > 0.27

            center_3d, obb_dims, tilt_deg, rot_deg = await asyncio.to_thread(
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

            results.append({
                "object_name": name,
                "position": [x, y, z], 
                "dimensions": [L, W],
                "area": area,
                "rotation_degree": rotation,
            })
    results = wrapper_for_indy(results) # indy 좌표계에 맞게 변환하는 코드

    # 8. Return list of detections
    return results






    

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
