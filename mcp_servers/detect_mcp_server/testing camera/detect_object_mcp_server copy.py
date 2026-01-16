import numpy as np
import torch
import pyrealsense2 as rs
torch.cuda.empty_cache()
import zlib
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sklearn.decomposition import PCA
from typing import  Dict, Any, List , Union, Optional# Type Hinting 추가
from mcp.server.fastmcp import FastMCP, Context  # MCP 관련 import
from contextlib import asynccontextmanager, contextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import cv2
import gc
import re
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import open3d as o3d
import pickle
import threading
import asyncio, base64, json




"""
7_29 -> 기존 SAM2에서 Dino : 언어로 객체 탐지, SAM2 : 영역 내 윤곽 탐지 인데 나는 SAM에 그렇게 
좋은 성능을 기대하지 않아서 훨신 빠른 버전인 MobileSAM으로 변경.

7_30 객체 탐지 여부는 오로지 dino만 관리함, sam은 dino의 출력을 다듬는 용도임
"""
from mobile_sam import SamPredictor, sam_model_registry

import logging

# # 루트 로거에 달려 있는 기존 핸들러 모두 제거
# logging.getLogger().handlers.clear()

# # 전파 방지 (중복 로깅 방지)
# logging.getLogger().propagate = False

# # 기본 로그 설정 복구 (필요에 따라 파일 또는 콘솔 설정)
# logging.basicConfig(level=logging.WARNING)  # WARNING 이상 로그만 출력
# # 코드 맨 위에 추가
# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore", category=UserWarning)


# 프로그램 시작 시 한 번만 설정
logging.basicConfig(
    level=logging.INFO,                # 로그 레벨 설정
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 모든 print 비활성화
import builtins
original_print = builtins.print
def safe_print(*args, **kwargs):
    try:
        original_print(*[str(arg) for arg in args], **kwargs)
    except:
        pass
builtins.print = safe_print

# 각 축(W, L, H)에 대한 크기 보정 오프셋 (단위: 미터)
X_OFFSETS = [-0.032, 0, 0]       # 인덱스별 X 오프셋 인덱스: 카메라 번호(카메라 여러개 사용할것으로 생각해서 만듬) -> 그리고 렌즈가 중심이 아니라서 이거 넣음
Y_OFFSETS = [0, 0, 0]       # 인덱스별 Y 오프셋 # indy에서는 이게 남/북 방향임
Z_OFFSETS = [0, 0, 0]   # 인덱스별 Height 오프셋
"""
인디의 손목 카메라 오프셋은 utils.py 내의 tool_to_world의 변수에 0.09로 넣었음
"""
# 전역 싱글톤 인스턴스들
_model_wrapper_instance = None
_camera_capture_instance = None

# 카메라가 기울어진 각도,  단위: 도(degree)
camera_tilt_theta = 0.0  # 실제 카메라가 아래를 향해 기울어져 있다고 가정

# 감지 레벨
score_threshold = 0.4 # 원래 0.5

# 마스크 shape 저장
mask_shape = None

@contextmanager  
def gpu_memory_guard():
    """GPU 메모리 안전 관리 - MCP 서버 환경용"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


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

            wrapped.append(new_r)
    except:
        wrapped = results
    return wrapped

def simple_enhance_clahe(image):
    """
    predict에 입력되는 이미지에 CLAHE로 대비를 늘리고,
    HSV의 S(채도)만 약간 올려 색감을 강화한다.
    반환: BGR uint8 (H, W, 3)
    """
    # 1) CLAHE (L 채널)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])

    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


class GroundedSAM2Wrapper:
    def __init__(
        self,
        dino_model_id="IDEA-Research/grounding-dino-tiny",
        device="cuda",
    ):
        self.device = torch.device(device)
        self._prediction_count = 0
        self._cleanup_interval = 8
        self._last_image_hash = None
        self._cached_image_set = False

        # 병렬 로딩 수행
        dino_thread = threading.Thread(target=self._load_dino, args=(dino_model_id,))
        sam_thread = threading.Thread(target=self._load_sam)

        dino_thread.start()
        sam_thread.start()

        dino_thread.join()
        sam_thread.join()


    def _load_dino(self, dino_model_id):
        try:
            print("Loading Grounding DINO model...")
            self.dino_processor = AutoProcessor.from_pretrained(dino_model_id)
            
            # 직접 vram에 매핑
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                dino_model_id,
                device_map="cuda",  # 또는 self.device
            )
            
            print("Grounding DINO loaded.")
        except Exception as e:
            print(f"Error loading Grounding DINO: {e}")
            raise

    def _load_sam(self):
        try:
            print("Loading MobileSAM model...")
            
            # 여기서는 전통적인 방식만 가능
            sam = sam_model_registry["vit_t"](checkpoint="mobile_sam.pt")
            sam.to(self.device)
            
            # 수동 메모리 정리만 가능
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            self.sam2_predictor = SamPredictor(sam)
            print("MobileSAM loaded.")
        except Exception as e:
            print(f"Error loading MobileSAM: {e}")
            raise

    def predict(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = score_threshold,
        text_threshold: float = score_threshold
    ):
        self._prediction_count += 1
        
        # 전체 predict 함수를 GPU 메모리 가드로 감싸기
        with gpu_memory_guard():
            try:
                # 이미지 전처리 (기존과 동일)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                image_np = image_rgb
                
            except Exception as e:
                logging.log(logging.ERROR, f"[WRAPPER] 이미지 변환 오류: {e}")
                raise TypeError("Input image must be a NumPy array.")

            if not text_prompt.strip().endswith("."):
                text_prompt += "."

            # 변수 초기화 (메모리 추적용)
            inputs = None
            outputs = None
            
            try:
                # GroundingDINO 처리
                inputs = self.dino_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.dino_model(**inputs)
                
            except Exception as e:
                logging.log(logging.ERROR, f"[WRAPPER] GroundingDINO 처리 오류: {e}")
                raise
            finally:
                # GroundingDINO 완료 후 입력 텐서 즉시 정리
                if inputs is not None:
                    del inputs
                    inputs = None

            try:
                # 후처리 (기존과 동일)
                target_size = [image_pil.size[::-1]]  # [H, W]
                
                # inputs.input_ids 대신 안전한 방법 사용
                # (inputs가 이미 삭제되었으므로)
                detections = self.dino_processor.post_process_grounded_object_detection(
                    outputs,
                    None,  # input_ids 없이도 작동하도록 수정 필요시
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_size,
                )[0]

                boxes = detections["boxes"]  
                scores = detections["scores"]
                phrases = detections["labels"]
                
            except Exception as e:
                logging.log(logging.ERROR, f"[WRAPPER] 후처리 오류: {e}")
                raise
            finally:
                # 후처리 완료 후 outputs 정리
                if outputs is not None:
                    del outputs
                    outputs = None

            # 검출된 객체가 없는 경우 (기존과 동일)
            if boxes.size(0) == 0:
                logging.log(logging.INFO, "[WRAPPER] 검출된 객체 없음")
                return torch.empty(0, dtype=torch.bool, device=self.device), \
                       torch.empty((0, 4), dtype=torch.float, device=self.device), \
                       [], \
                       torch.empty(0, dtype=torch.float, device=self.device)

            try:
                # MobileSAM 처리 - 이미지 캐싱 최적화 추가
                image_hash = hash(image_np.tobytes())
                if not self._cached_image_set or self._last_image_hash != image_hash:
                    self.sam2_predictor.set_image(image_np)
                    self._cached_image_set = True
                    self._last_image_hash = image_hash
                
                boxes_np = boxes.cpu().numpy()
                
                # 각 박스별로 개별 처리 (기존 로직 그대로)
                all_masks = []
                all_sam_scores = []
                
                for i, box in enumerate(boxes_np):
                    try:
                        with torch.no_grad():
                            # 단일 박스로 예측 (기존과 동일)
                            single_masks, single_scores, _ = self.sam2_predictor.predict(
                                box=box.reshape(1, -1),  # (1, 4) 형태로 변경
                                multimask_output=False,
                            )
                        
                        # 차원 정리 (기존과 동일)
                        if single_masks.ndim == 4:  # (1, 1, H, W)
                            single_masks = single_masks.squeeze(0).squeeze(0)  # (H, W)
                        elif single_masks.ndim == 3:  # (1, H, W)
                            single_masks = single_masks.squeeze(0)  # (H, W)
                        
                        all_masks.append(single_masks)
                        all_sam_scores.append(single_scores[0] if isinstance(single_scores, np.ndarray) else single_scores)
                        
                    except Exception as box_error:
                        logging.log(logging.ERROR, f"[WRAPPER] 박스 {i} 처리 오류: {box_error}")
                        # 실패한 박스는 빈 마스크로 대체 (기존과 동일)
                        h, w = image_np.shape[:2]
                        empty_mask = np.zeros((h, w), dtype=bool)
                        all_masks.append(empty_mask)
                        all_sam_scores.append(0.0)
                
                # 모든 마스크를 스택으로 결합 (기존과 동일)
                if all_masks:
                    sam_masks = np.stack(all_masks, axis=0)  # (N, H, W)
                    sam_scores = np.array(all_sam_scores)
                else:
                    # 빈 결과 (기존과 동일)
                    logging.log(logging.WARNING, "[WRAPPER] 모든 마스크 처리 실패")
                    return torch.empty(0, dtype=torch.bool, device=self.device), \
                           torch.empty((0, 4), dtype=torch.float, device=self.device), \
                           [], \
                           torch.empty(0, dtype=torch.float, device=self.device)
                           
            except Exception as e:
                logging.log(logging.ERROR, f"[WRAPPER] MobileSAM 처리 오류: {e}")
                raise

            try:
                # 최종 텐서 변환 (기존과 동일)
                masks_tensor = torch.from_numpy(sam_masks).to(self.device)
                
                return masks_tensor, boxes, phrases, scores
                
            except Exception as e:
                logging.log(logging.ERROR, f"[WRAPPER] 최종 텐서 변환 오류: {e}")
                raise
                
            finally:
                # 주기적 강제 정리 (MCP 서버 환경용)
                if self._prediction_count % self._cleanup_interval == 0:
                    import gc
                    gc.collect()
                    logging.debug(f"주기적 메모리 정리 완료 ({self._prediction_count}회차)")


    def predict_dino_only(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.31,
        text_threshold: float = 0.21
    ):
        self._prediction_count += 1
        """
        Grounding DINO만 사용하여 이미지에서 지정된 텍스트 프롬프트에 해당하는 객체를 탐지하고,
        바운딩 박스 좌표, confidence 점수, 라벨(문자열)을 반환합니다.
        SAM(Segment Anything) 처리는 포함되지 않습니다.

        Args:
            image (np.ndarray): BGR 포맷의 입력 이미지 (OpenCV 형식).
            text_prompt (str): 탐지할 객체에 대한 텍스트 프롬프트 (예: "cat, dog").
            box_threshold (float): 바운딩 박스 confidence score 필터링 기준.
            text_threshold (float): 텍스트 매칭 점수 필터링 기준.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, List[str]]:
                - boxes: (N, 4) shape의 바운딩 박스 텐서 (x1, y1, x2, y2)
                - scores: 각 박스의 confidence score 텐서 (N,)
                - phrases: 각 박스에 대응하는 텍스트 라벨 문자열 리스트

        Raises:
            TypeError: 입력 이미지가 numpy array가 아닌 경우.
            Exception: DINO 추론 또는 후처리 중 발생하는 오류.

        Note:
            이 함수는 GPU 메모리 보호 기능을 포함하며, DINO 처리만 수행합니다.
            SAM 기반 마스크는 생성하지 않습니다.
        """
        with gpu_memory_guard():
            try:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
            except Exception as e:
                logging.log(logging.ERROR, f"[DINO_ONLY] 이미지 변환 오류: {e}")
                raise TypeError("Input image must be a NumPy array.")

            if not text_prompt.strip().endswith("."):
                text_prompt += "."

            try:
                inputs = self.dino_processor(
                    images=image_pil,
                    text=text_prompt,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.dino_model(**inputs)
            except Exception as e:
                logging.log(logging.ERROR, f"[DINO_ONLY] DINO 추론 오류: {e}")
                raise

            try:
                target_size = [image_pil.size[::-1]]  # [H, W]
                detections = self.dino_processor.post_process_grounded_object_detection(
                    outputs,
                    None,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=target_size,
                )[0]

                boxes = detections["boxes"]
                scores = detections["scores"]
                phrases = detections["labels"]

            except Exception as e:
                logging.log(logging.ERROR, f"[DINO_ONLY] 후처리 오류: {e}")
                raise

            if boxes.size(0) == 0:
                logging.log(logging.INFO, "[DINO_ONLY] 검출된 박스 없음")
                return torch.empty((0, 4), dtype=torch.float, device=self.device), \
                    torch.empty(0, dtype=torch.float, device=self.device), \
                    []

            return boxes, scores, phrases


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

            # 7_30 추가 (clahe적용해 이미지 선명화)
            color_image = simple_enhance_clahe(color_image)


            # 8_12일 추가 (특정 깊이 이상의 위치에 있는 rgb 값은 0으로 바꿔서 필요없는 부분 (바닥같은거) 날림)
            threshold_m = 1 # 1m 이상으로 먼거는 날림
            mask = (depth_image_raw * self.depth_scale) >= threshold_m
            color_image[mask] = 0
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

def calculate_object_properties(mask, depth_image_raw, intrinsics, depth_scale): # added in 7_31 but PCA remains
    try:
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
        
        # 2) 깊이 → 미터 (한 번에 처리)
        depth_m = depth_image_raw.astype(np.float32) * depth_scale
        
        # 3) 벡터화된 3D 변환
        # 유효한 깊이 값만 필터링
        y_coords, x_coords = pts_yx[:, 0], pts_yx[:, 1]
        depth_vals = depth_m[y_coords, x_coords]
        valid_mask = (depth_vals > 0.01) & (depth_vals < 0.8)
        
        if np.sum(valid_mask) < 4:
            print("Warning: Too few 3D points.")
            return None, None, None, None
        
        # 유효한 점들만 선택
        valid_x = x_coords[valid_mask]
        valid_y = y_coords[valid_mask]
        valid_depths = depth_vals[valid_mask]
        
        # 벡터화된 deproject 계산 
        fx, fy = intrinsics.fx, intrinsics.fy
        ppx, ppy = intrinsics.ppx, intrinsics.ppy
        
        # 3D 좌표 계산 (벡터화)
        X = (valid_x - ppx) * valid_depths / fx
        Y = (valid_y - ppy) * valid_depths / fy
        Z = valid_depths
        
        pc = np.column_stack((X, Y, Z)).astype(np.float32)
        
        # 4) 중심점 계산
        center_3d = pc.mean(axis=0)
        
        # 5) 절대 높이
        height = float(pc[:, 2].max() - pc[:, 2].min())
        
        # 6) PCA → OBB 축 길이 
        pca = PCA(n_components=3, svd_solver='randomized', random_state=42)
        pca.fit(pc)
        
        t_pc = pca.transform(pc)
        size_obb = t_pc.max(axis=0) - t_pc.min(axis=0)
        
        # 7) 높이에 가장 가까운 축 제거 → L, W
        diffs = np.abs(size_obb - height)
        idx_h = np.argmin(diffs)
        
        # 마스킹을 이용한 효율적인 L, W 계산
        mask_lw = np.ones(3, dtype=bool)
        mask_lw[idx_h] = False
        lw = size_obb[mask_lw]
        L, W = float(lw.max()), float(lw.min())
        obb_dims = [L, W, height]
        
        # 8) tilt/rot: 수평면상 가장 긴 축 기준
        axes = pca.components_
        proj = np.hypot(axes[:, 0], axes[:, 1])
        idx_horiz = np.argmax(proj)
        v_h = axes[idx_horiz]
        
        norm_h = np.linalg.norm(v_h)
        tilt_deg = float(np.degrees(np.arcsin(np.abs(v_h[2]) / norm_h)))
        rot_deg = float(np.degrees(np.arctan2(-v_h[1], v_h[0])))
        
        # 각도 정규화
        if np.abs(rot_deg) > 90:
            rot_deg = (rot_deg + 180) if rot_deg < 0 else (rot_deg - 180)
        


        # 좌표계 변환 (y 반전)
        center_3d = [float(center_3d[0]), float(-center_3d[1]), float(center_3d[2])]
        
        return center_3d, obb_dims, tilt_deg, rot_deg
    
    except Exception as e:
        logging.log(logging.ERROR, f"[calculate_object_properties error]{e}")



# Lifespan에서 사용할 컨텍스트 정의
@dataclass
class AppContext:
    model_wrapper: GroundedSAM2Wrapper
    camera_capture: RealSenseCapture



@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """서버 시작 시 모델과 카메라 로드, 종료 시 카메라 해제"""
    print("Initializing resources for MCP server...")

    global _model_wrapper_instance
    global _camera_capture_instance

    def init_model():
        global _model_wrapper_instance
        print("[Init] Creating GroundedSAM2Wrapper...")
        _model_wrapper_instance = GroundedSAM2Wrapper()
        print("[Init] GroundedSAM2Wrapper created.")

    def init_camera():
        global _camera_capture_instance
        print("[Init] Creating RealSenseCapture...")
        camera_serials = list_connected_realsense_devices()
        if not camera_serials:
            raise RuntimeError("No RealSense cameras connected.")
        selected_serial = camera_serials[0]
        _camera_capture_instance = RealSenseCapture(serial_number=selected_serial)
        if not _camera_capture_instance.start():
            raise RuntimeError("Failed to start RealSense camera.")
        print("[Init] RealSenseCapture created and started.")

    try:
        # 병렬 실행
        model_thread = threading.Thread(target=init_model)
        camera_thread = threading.Thread(target=init_camera)

        model_thread.start()
        camera_thread.start()

        model_thread.join()
        camera_thread.join()

        print("Resources initialized successfully.")

        yield AppContext(
            model_wrapper=_model_wrapper_instance,
            camera_capture=_camera_capture_instance
        )

    finally:
        print("Shutting down resources...")
        # (리소스 해제는 현재 설계상 불필요함)
        print("Resources shut down.")


# MCP 서버 인스턴스 생성 및 lifespan 연결
mcp = FastMCP("detect_object_mcp_server", lifespan=app_lifespan)



def encode_mask(mask: np.ndarray) -> str:
    """
    Compress and encode a binary mask into a base64 string.
    """
    compressed = zlib.compress(mask.astype(np.uint8).tobytes())
    return base64.b64encode(compressed).decode('utf-8')


######################################################################################################

# find_object_3d_properties에서 사용
@mcp.tool()
async def find_object_3d_properties(
    object_names: Union[str, List[str]],
    ctx: Context,
    camera_index: int = 0
) -> List[Dict[str, Any]]:
    """
    물체의 위치, L,W, Area, mask 등등을 리턴

    """
    
    # 전체 함수를 GPU 메모리 가드로 감쌈
    with gpu_memory_guard():
        try:
            # 입력 정규화
            if isinstance(object_names, str):
                names = [n.strip() for n in re.split(r"[\.,]", object_names) if n.strip()]
            else:
                names = object_names
            
            # 리소스 획득 (app_lifespan에서 초기화된 것들)
            app_context: AppContext = ctx.request_context.lifespan_context
            wrapper = app_context.model_wrapper  # 이미 GPU에 로드된 모델
            capture = app_context.camera_capture # 이미 시작된 카메라
            
            # 프레임 캡처
            if not capture.is_running:
                raise RuntimeError("Camera is not running. Check initialization.")
            
            color_image, depth_image_raw = await asyncio.to_thread(capture.capture_aligned_frames)
            if color_image is None or depth_image_raw is None:
                raise RuntimeError("Frame capture failed.")
            
            intrinsics = capture.get_intrinsics()
            depth_scale = capture.get_depth_scale()
            
            # 각 객체를 완전히 독립적으로 처리
            all_results: List[Dict[str, Any]] = []
            
            for name_idx, name in enumerate(names):
                # 각 객체 처리도 GPU 가드로 감쌈
                with gpu_memory_guard():
                    try:
                        single_object_results = await process_single_object_mcp_optimized(
                            name, 
                            color_image, 
                            depth_image_raw, 
                            intrinsics, 
                            depth_scale, 
                            wrapper,  # 계속 GPU에 로드된 상태
                            camera_index
                        )
                        
                        all_results.extend(single_object_results)
                        
                    except Exception as e:
                        logging.log(logging.ERROR, f"[find_object_3d_properties error1]{e}")
                        continue
            
            # indy 좌표계 변환 적용
            all_results = wrapper_for_indy(all_results)
            
            return all_results
        
        except Exception as e:
            logging.log(logging.ERROR, f"[find_object_3d_properties error2]{e}")
            return None

async def process_single_object_mcp_optimized(
    object_name: str,
    color_image,
    depth_image_raw,
    intrinsics,
    depth_scale,
    wrapper,
    camera_index: int
) -> List[Dict[str, Any]]:
    """MCP 서버 최적화된 단일 객체 처리"""
    results_for_this_object = []
    
    # 모든 GPU 텐서 변수를 None으로 초기화
    masks_tensor = None
    boxes_tensor = None
    scores_tensor = None
    masks_cpu = None
    scores_cpu = None
    
    try:
        # 1. 객체 검출
        text_prompt = f"{object_name}."
        
        masks_tensor, boxes_tensor, phrases_list, scores_tensor = await asyncio.to_thread(
            wrapper.predict,
            color_image,
            text_prompt
        )
        
        if boxes_tensor.size(0) == 0:
            return results_for_this_object
        
        num_detections = boxes_tensor.size(0)
        logging.info(f"검출 완료: {object_name} - {num_detections}개 객체")
        
        # 2. 즉시 CPU로 이동하고 GPU 해제
        masks_cpu = masks_tensor.detach().cpu().contiguous().clone()
        scores_cpu = scores_tensor.detach().cpu().contiguous().clone()
        
        # GPU 텐서들 즉시 완전 삭제
        del masks_tensor, boxes_tensor, scores_tensor
        masks_tensor = None
        boxes_tensor = None
        scores_tensor = None
        
        # GPU 메모리 강제 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 3. 각 검출을 완전히 독립적으로 처리
        successful_count = 0
        
        for detection_idx in range(num_detections):
            detection_result = None
            try:
                detection_result = await process_detection_mcp_optimized(
                    object_name,
                    detection_idx,
                    masks_cpu,
                    scores_cpu,
                    depth_image_raw,
                    intrinsics,
                    depth_scale,
                    camera_index
                )
                
                if detection_result is not None:
                    results_for_this_object.append(detection_result)
                    successful_count += 1
                    
            except Exception as e:
                logging.warning(f"검출 {detection_idx} 처리 실패: {str(e)[:100]}")
                continue
                
            finally:
                # 각 검출 후 정리
                if detection_result is not None:
                    del detection_result
                detection_result = None
                gc.collect()
        
        # logging.info(f"객체 처리 완료: {object_name} - {successful_count}/{num_detections}개 성공")
        
    except Exception as e:
        logging.error(f"객체 처리 실패: {object_name} - {e}")
        
    finally:
        # 최종 정리
        try:
            if masks_tensor is not None:
                del masks_tensor
            if boxes_tensor is not None:
                del boxes_tensor
            if scores_tensor is not None:
                del scores_tensor
            if masks_cpu is not None:
                del masks_cpu
            if scores_cpu is not None:
                del scores_cpu
        except:
            pass
        gc.collect()
    
    return results_for_this_object


async def process_detection_mcp_optimized(
    object_name: str,
    detection_idx: int,
    masks_cpu_tensor,
    scores_cpu_tensor,
    depth_image_raw,
    intrinsics,
    depth_scale,
    camera_index: int
) -> Dict[str, Any]:
    """MCP 서버 최적화된 단일 검출 처리"""

    global mask_shape

    # 모든 변수를 None으로 초기화
    single_mask_raw = None
    mask_binary = None
    isolated_mask = None
    mask_encoded = None
    
    try:
        # 1. 마스크 추출 - 완전히 새로운 메모리 공간
        single_mask_raw = masks_cpu_tensor[detection_idx].numpy().copy()
        mask_binary = single_mask_raw > score_threshold
        
        # 완전히 독립적인 새 배열 생성 
        isolated_mask = np.empty(mask_binary.shape, dtype=bool)
        isolated_mask[:] = mask_binary[:]
        
        # 스코어 추출
        detection_score = float(scores_cpu_tensor[detection_idx].item())
        
        # 중간 변수들 즉시 삭제
        del single_mask_raw, mask_binary
        single_mask_raw = None
        mask_binary = None
        
        # 2. 3D 속성 계산
        center_3d, obb_dims, tilt_deg, rot_deg = await asyncio.to_thread(
            calculate_object_properties,
            isolated_mask,
            depth_image_raw,
            intrinsics,
            depth_scale
        )
        
        if center_3d is None or obb_dims is None:
            return {
                "object_name": object_name,
                "error": "3D calculation failed",
                "detection_score": detection_score
            }
        
        # 3. 좌표 변환 - 모든 변수를 독립적으로
        x_offset = X_OFFSETS[camera_index]
        y_offset = Y_OFFSETS[camera_index] 
        z_offset = Z_OFFSETS[camera_index]
        
        x_coordinate = round(float(center_3d[0]), 3) + x_offset  # x,y좌표 바꿈 (주석만 있고 실제로는 바꾸지 않음)
        y_coordinate = round(float(center_3d[1]), 3) + y_offset
        z_coordinate = round(float(obb_dims[2]), 3) + z_offset
        length_value = round(float(obb_dims[0]), 3)
        width_value = round(float(obb_dims[1]), 3)
        area_value = round(length_value * width_value, 6)
        rotation_value = round(float(rot_deg), 2)
        
        # 4. 필터링
        if z_coordinate > 0.6 or length_value > 0.6:
            return None
        if area_value >= 0.06 or area_value <= 0.001:
            return None
        
        # 5. 마스크 인코딩 - 완전히 독립적인 변수
        mask_encoded = encode_mask(isolated_mask)
        mask_dimensions = [isolated_mask.shape[0], isolated_mask.shape[1]]
        mask_shape = mask_dimensions
        
        # 6. 최종 결과 딕셔너리 - 모든 값이 독립적
        final_result = {
            "object_name": object_name,
            "position": [x_coordinate, y_coordinate, z_coordinate],
            "dimensions": [length_value, width_value],
            "area": area_value,
            "rotation_degree": rotation_value,
            "mask_base64": mask_encoded,
            "mask_shape": mask_dimensions,
            "score": round(detection_score, 3)
        }
        
        return final_result
        
    except Exception as e:
        logging.log(logging.ERROR, f"[process_detection_mcp_optimized error]{e}")
        return None
    
    finally:
        # 모든 변수 완전 정리 - None 체크 후 삭제
        for var in [single_mask_raw, mask_binary, isolated_mask, mask_encoded]:
            if var is not None:
                del var
                
        # 가비지 컬렉션 (MCP 서버에서는 time.sleep 없이)
        gc.collect()


###################################################### vlm에게 물체에 숫자 라벨과 바운더리 박스만 주게 하는 코드#########################################################


@mcp.tool()
async def capture_dino_labeled_image(ctx: Context, camera_index: int = 0) -> Dict[str, Any]:
    """
    DINO로 객체를 감지하고 숫자 라벨링된 이미지 + 박스 정보를 반환
    
    Returns:
        Dict[str, Any]: {
            "image": "data:image/jpg;base64,...",
            "boxes": [[x1,y1,x2,y2], [x1,y1,x2,y2], ...],  # 라벨 숫자 순서대로
        }
    """
    
    try:
        # 리소스 획득 (기존 패턴과 동일)
        app_context: AppContext = ctx.request_context.lifespan_context
        wrapper = app_context.model_wrapper
        capture = app_context.camera_capture
        
        if not capture.is_running:
            raise RuntimeError("Camera is not running. It must be started in lifespan.")
        
        # 프레임 캡처 (기존 패턴과 동일)
        color_image, *_ = await asyncio.to_thread(capture.capture_aligned_frames)
        if color_image is None:
            raise RuntimeError("Failed to capture color frame.")
        
        # DINO 추론으로 라벨링된 이미지 + 박스 정보 생성
        with gpu_memory_guard():
            annotated_image, boxes_list = await asyncio.to_thread(
                run_dino_inference_with_boxes,
                color_image,
                wrapper
            )
        
        # JPEG로 인코딩 (기존 패턴과 동일)
        success, jpg_bytes = cv2.imencode('.jpg', annotated_image)
        if not success:
            raise RuntimeError("Failed to encode annotated image to JPEG.")
        
        # base64 인코딩 (기존 패턴과 동일)
        jpg_base64 = base64.b64encode(jpg_bytes.tobytes()).decode('utf-8')
        image_data_url = f"data:image/jpg;base64,{jpg_base64}"
        
        # 결과 반환
        return {
            "image": image_data_url,
            "boxes": boxes_list  # 라벨 숫자 순서대로 [x1,y1,x2,y2]
        }
        
    except Exception as e:
        return {
            "image": "",
            "boxes": []
        }


def run_dino_inference_with_boxes(image_np_bgr, wrapper):
    """
    wrapper의 predict_dino_only()를 사용하여 DINO만 실행하고 
    숫자 라벨이 표시된 이미지 + 박스 좌표 리스트를 반환
    
    Returns:
        tuple: (annotated_image, boxes_list)
            - annotated_image: 라벨링된 이미지 (numpy array)
            - boxes_list: [[x1,y1,x2,y2], ...] 라벨 숫자 순서대로
    """
    boxes_tensor = None
    scores_tensor = None
    
    try:
        text_prompt = "object"
        boxes_tensor, scores_tensor, *_ = wrapper.predict_dino_only(
            image_np_bgr,
            text_prompt
        )
        
        if boxes_tensor.size(0) == 0:
            return image_np_bgr.copy(), []
        
        input_boxes = boxes_tensor.detach().cpu().numpy()
        
        del boxes_tensor, scores_tensor
        boxes_tensor = None
        scores_tensor = None
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 박스 좌표를 라벨 순서대로 저장 (소수점 제거)
        boxes_list = []
        for i, box in enumerate(input_boxes):
            x1, y1, x2, y2 = box.astype(int)  # 정수로 변환
            boxes_list.append([x1, y1, x2, y2])  # 라벨 번호 i에 맞는 박스
        
        annotated_frame = image_np_bgr.copy()
        
        if len(input_boxes) > 0:
            # === 색상 팔레트 정의 (BGR) ===
            color_palette = [
                (0, 0, 255),    # 빨강
                (0, 255, 0),    # 초록
                (200, 0, 0),    # 파랑
                (255, 0, 255),  # 분홍
                (0, 165, 255),  # 주황
                (200, 200, 0),  # 청록
            ]
            
            # === 간단한 그룹핑 (IoU 기반) ===
            def compute_iou(box1, box2):
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])
                inter_area = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union_area = area1 + area2 - inter_area
                return inter_area / union_area if union_area > 0 else 0
            
            n = len(input_boxes)
            groups = [-1] * n
            current_group = 0
            
            for i in range(n):
                if groups[i] != -1:
                    continue
                groups[i] = current_group
                for j in range(i + 1, n):
                    if compute_iou(input_boxes[i], input_boxes[j]) >= 0.5:
                        groups[j] = current_group
                current_group += 1
            
            # === 박스 시각화 (크기별 라벨 위치 조정) ===
            for i, box in enumerate(input_boxes):
                x1, y1, x2, y2 = box.astype(int)
                group_id = groups[i]
                color = color_palette[group_id % len(color_palette)]
                
                # 박스 그리기
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)
                
                # 박스 크기 계산
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                
                # 라벨 텍스트 설정
                label = str(i)  # 여기가 중요: 라벨 번호 = 박스 리스트 인덱스
                font_scale = 0.8
                thickness = 2
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                padding = 1
                
                # === 크기별 라벨 위치 결정 ===
                area_threshold = 5500
                
                if box_area > area_threshold:
                    # 큰 박스: 중앙에 라벨 배치
                    label_x = x1 + (box_width - label_size[0]) // 2
                    label_y = y1 + (box_height + label_size[1]) // 2
                    
                    bg_x1 = label_x - padding
                    bg_y1 = label_y - label_size[1] - padding
                    bg_x2 = label_x + label_size[0] + padding
                    bg_y2 = label_y + padding
                    
                    # cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                    cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
                    cv2.putText(annotated_frame, label, (label_x, label_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                    
                else:
                    # 작은 박스: 상단에 라벨 배치
                    cv2.rectangle(annotated_frame,
                                 (x1, y1 - label_size[1] - padding),
                                 (x1 + label_size[0] + padding, y1),
                                 (0, 0, 0), -1)
                    
                    cv2.putText(annotated_frame, label,
                               (x1 + 2, y1 - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        return annotated_frame, boxes_list
        
    except Exception as e:
        logging.error(f"DINO inference failed: {e}")
        return image_np_bgr.copy(), []
        
    finally:
        try:
            if boxes_tensor is not None:
                del boxes_tensor
            if scores_tensor is not None:
                del scores_tensor
        except:
            pass
        gc.collect()




@mcp.tool()
async def match_input_bboxes_with_dino(
    ctx: Context,
    input_boxes: List[List[Union[int, float, str]]],  # [[x1,y1,x2,y2], ...]
    iou_threshold: float = 0.35,       # 일치 기준
    strategy: str = "prefer_dino",     # "prefer_dino" | "prefer_input" | "union_if_low_iou"
    return_image: bool = True
) -> Dict[str, Any]:
    """
    현재 프레임에서 DINO 박스를 얻고, 입력 박스와 IoU로 매칭해 최종 박스 선택.
    반환:
      {
        "boxes": [[x1,y1,x2,y2] or None, ...],  # 입력 순서대로
        "matches": [{"best_iou":0.52,"chosen":"dino|input|union","dino_idx":3}, ...],
        "image": "data:image/jpg;base64,..."    # 옵션(디버그 시각화)
      }
    """

    # ---- 컨텍스트/캡처 ----
    app_context: AppContext = ctx.request_context.lifespan_context
    wrapper = app_context.model_wrapper
    capture = app_context.camera_capture

    color_image, _ = await asyncio.to_thread(capture.capture_aligned_frames)
    if color_image is None:
        return {"boxes": [], "matches": [], "image": ""}

    H, W = color_image.shape[:2]

    # ---- DINO 실행 ----
    with gpu_memory_guard():
        annotated_image, dino_boxes = await asyncio.to_thread(
            run_dino_inference_with_boxes, color_image, wrapper
        )
    # run_dino_inference_with_boxes는 int로 캐스팅해서 리스트 반환함

    # ---- 유틸: 박스 정규화/클리핑 ----
    def _coerce_box(b):
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            return None
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in b]
        except Exception:
            return None
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        x1 = max(0, min(W-1, x1)); x2 = max(0, min(W-1, x2))
        y1 = max(0, min(H-1, y1)); y2 = max(0, min(H-1, y2))
        if x2 <= x1 or y2 <= y1: return None
        return [x1, y1, x2, y2]

    in_boxes = [ _coerce_box(b) for b in (input_boxes or []) ]

    # ---- IoU 등 메트릭 ----
    def iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
        a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
        uni = a1 + a2 - inter
        return inter/uni if uni>0 else 0.0

    # ---- 매칭/선택 ----
    final_boxes = []
    match_info  = []

    for ib in in_boxes:
        if ib is None or not dino_boxes:
            final_boxes.append(ib)
            match_info.append({"best_iou": 0.0, "chosen": "input" if ib else "none", "dino_idx": -1})
            continue

        best_idx, best_i = -1, -1.0
        for j, db in enumerate(dino_boxes):
            v = iou(ib, db)
            if v > best_i:
                best_i, best_idx = v, j

        if best_idx < 0:
            final_boxes.append(ib)
            match_info.append({"best_iou": 0.0, "chosen": "input", "dino_idx": -1})
            continue

        db = dino_boxes[best_idx]

        # 전략
        if best_i >= iou_threshold:
            if strategy == "prefer_dino":
                chosen, fb = "dino", db
            elif strategy == "union_if_low_iou":
                # 충분히 높으면 dino, 애매하면 union
                if best_i < 0.6:
                    x1 = min(ib[0], db[0]); y1 = min(ib[1], db[1])
                    x2 = max(ib[2], db[2]); y2 = max(ib[3], db[3])
                    chosen, fb = "union", [x1,y1,x2,y2]
                else:
                    chosen, fb = "dino", db
            else:  # prefer_input
                chosen, fb = "input", ib
        else:
            if strategy == "prefer_dino":
                chosen, fb = "dino", db
            elif strategy == "union_if_low_iou":
                x1 = min(ib[0], db[0]); y1 = min(ib[1], db[1])
                x2 = max(ib[2], db[2]); y2 = max(ib[3], db[3])
                chosen, fb = "union", [x1,y1,x2,y2]
            else:
                chosen, fb = "input", ib

        # 클리핑
        fb = _coerce_box(fb)
        final_boxes.append(fb)
        match_info.append({"best_iou": round(float(best_i), 4), "chosen": chosen, "dino_idx": int(best_idx)})

    # ---- 디버그 이미지 ----
    out_img_b64 = ""
    if return_image:
        vis = color_image.copy()
        # DINO(초록)
        for b in dino_boxes:
            x1,y1,x2,y2 = [int(v) for v in b]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,255,0), 1)
        # 입력(청록)
        for b in in_boxes:
            if b:
                cv2.rectangle(vis, (b[0],b[1]), (b[2],b[3]), (200,200,0), 1)
        # 최종(빨강)
        for b in final_boxes:
            if b:
                cv2.rectangle(vis, (b[0],b[1]), (b[2],b[3]), (0,0,255), 2)

        ok, jpg_bytes = cv2.imencode(".jpg", vis)
        if ok:
            out_img_b64 = "data:image/jpg;base64," + base64.b64encode(jpg_bytes.tobytes()).decode("utf-8")

    # 메모리 정리
    gc.collect()
    torch.cuda.empty_cache()

    return {"boxes": final_boxes, "matches": match_info, "image": out_img_b64}


@mcp.tool()
async def get_3d_properties_from_boxes(
    boxes_coords: List[List[int]],
    ctx: Context,
    camera_index: int = 0
) -> List[Dict[str, Any]]:
    """
    DINO의 출력된 여러 개의 box를 입력받아, 각 box에 대해 SAM으로 객체의 상세 정보를
    추출해서 리스트 형태로 반환. 
    출력은 입력된 박스 순서와 동일하게 출력.
    """
    # 리소스 및 변수 초기화
    color_image, depth_image_raw, masks, scores, final_mask = None, None, None, None, None
    all_results = []
    global mask_shape

    with gpu_memory_guard():
        try:
            # 1. 리소스 획득 및 프레임 캡처 (한 번만 수행)
            app_context: AppContext = ctx.request_context.lifespan_context
            wrapper = app_context.model_wrapper
            capture = app_context.camera_capture
            color_image, depth_image_raw = await asyncio.to_thread(capture.capture_aligned_frames)
            if color_image is None:
                raise RuntimeError("프레임 캡처 실패")
            intrinsics = capture.get_intrinsics()
            depth_scale = capture.get_depth_scale()

            # MobileSAM에 사용할 이미지를 한 번만 설정하여 효율성 증대
            wrapper.sam2_predictor.set_image(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

            # --- 각 박스에 대한 반복 처리 ---
            for single_box_coords in boxes_coords:
                try:
                    # 2. 현재 박스에 대한 MobileSAM 실행
                    box_np = np.array(single_box_coords, dtype=np.float32).reshape(1, 4)
                    masks, scores, _ = wrapper.sam2_predictor.predict(
                        box=box_np,
                        multimask_output=False,
                    )
                    
                    # 최적화: 필요한 데이터 추출 후 원본 삭제
                    final_mask = masks[0] > score_threshold
                    final_score = round(float(scores[0]), 3)
                    del masks, scores, box_np

                    # 3. 현재 마스크에 대한 3D 속성 계산
                    center_3d, obb_dims, _, rot_deg = await asyncio.to_thread(
                        calculate_object_properties,
                        final_mask,
                        depth_image_raw,
                        intrinsics,
                        depth_scale
                    )
                    
                    if center_3d is None:
                        # 특정 박스 계산 실패 시, 에러 정보를 추가하고 다음 박스로 진행
                        all_results.append({"error": f"Box {single_box_coords}의 3D 속성을 계산할 수 없습니다."})
                        continue

                    # 4. 결과 포맷팅
                    x_offset = X_OFFSETS[camera_index]
                    y_offset = Y_OFFSETS[camera_index]
                    z_offset = Z_OFFSETS[camera_index]

                    mask_shape_arg = [final_mask.shape[0], final_mask.shape[1]]
                    if mask_shape == None:
                        mask_shape = mask_shape_arg
                    
                    z = round(float(obb_dims[2]), 3) + z_offset

                    result = {
                        "position": [
                            round(float(center_3d[0]), 3) + x_offset,
                            round(float(center_3d[1]), 3) + y_offset,
                            z
                        ],
                        "dimensions": [round(float(obb_dims[0]), 3), round(float(obb_dims[1]), 3)],
                        "area": round(float(obb_dims[0]*1000) * float(obb_dims[1]*1000) * (z*1000), 6), # 1보다 작은 수를 곱하면 더 작아지는 오류 수정 -> 가로 * 세로 * 높이임
                        "rotation_degree": round(float(rot_deg), 2),
                        "mask_base64": encode_mask(final_mask),
                        "score": final_score
                    }
                    final_result_for_one_box = wrapper_for_indy([result])[0]
                    
                    # 최종 결과 리스트에 추가
                    all_results.append(final_result_for_one_box)
                
                except Exception as e:
                    # 특정 박스 처리 중 예외 발생 시, 에러 정보를 추가하고 다음으로 진행
                    logging.error(f"Box {single_box_coords} 처리 중 오류: {e}")
                    all_results.append({"error": f"Box {single_box_coords} 처리 중 오류 발생"})
                    continue

            # 모든 박스 처리 후 큰 변수 삭제
            del depth_image_raw

            return all_results

        except Exception as e:
            logging.error(f"[get_3d_properties_from_boxes] 전체 작업 오류: {e}")
            return [{"error": str(e)}] # 전체 작업 실패 시 에러 반환
        finally:
            # 함수 종료 시 확실한 메모리 정리
            if 'color_image' in locals() and color_image is not None:
                del color_image
            if 'depth_image_raw' in locals() and depth_image_raw is not None:
                del depth_image_raw
            if 'masks' in locals() and masks is not None:
                del masks
            if 'scores' in locals() and scores is not None:
                del scores
            if 'final_mask' in locals() and final_mask is not None:
                del final_mask
            gc.collect()


@mcp.tool()
async def bboxes_from_sideclouds(
    ctx: Context,
    side_cloud_items: list,            # [{"id1": <cloud1>}, {"id2": <cloud2>}]
    current_tool_pos: list,            # 로봇 베이스 기준 [x, y, z]
    margin_px: int = 0,
    depth_tol_m: float = 0.02,
) -> dict:
    """
    여러 '절대좌표' 사이드 포인트클라우드를 현재 카메라 포즈에서 재투영해
    id -> 2D 바운딩박스 매핑을 반환.  보이지 않으면 None.
    반환 예: {"id1":[x1,y1,x2,y2], "id2": None, ...}
    """
    import asyncio, base64, json
    import numpy as np

    # ---- 컨텍스트/장치 ----
    app_context: AppContext = ctx.request_context.lifespan_context
    capture = app_context.camera_capture

    # ---- 프레임/파라미터 (1회 캡처) ----
    color_image, depth_raw = await asyncio.to_thread(capture.capture_aligned_frames)
    if color_image is None or depth_raw is None:
        # 입력 키 보존하며 전부 None
        return {list(item.keys())[0] if isinstance(item, dict) and item else f"unknown_{i}": None
                for i, item in enumerate(side_cloud_items)}
    intr = capture.get_intrinsics()
    depth_scale = float(capture.get_depth_scale())
    img_h, img_w = depth_raw.shape
    fx, fy, ppx, ppy = float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy)

    # ---- 좌표 뒤집기 ----
    current_tool_pos = (current_tool_pos or [])[:3]
    if len(current_tool_pos) != 3:
        return {list(item.keys())[0] if isinstance(item, dict) and item else f"unknown_{i}": None
                for i, item in enumerate(side_cloud_items)}
    t_cam_world = np.array([-current_tool_pos[1], -current_tool_pos[0], -current_tool_pos[2]], dtype=np.float32)

    def _decode_cloud(cloud):
        # cloud: base64(float32 Nx3) or JSON string or [[x,y,z],...]
        if isinstance(cloud, str):
            # 1) base64 시도
            try:
                arr = np.frombuffer(base64.b64decode(cloud), dtype=np.float32)
                if arr.size % 3 != 0:
                    raise ValueError("bad size")
                return arr.reshape(-1, 3)
            except Exception:
                # 2) JSON 시도
                try:
                    obj = json.loads(cloud)
                    arr = np.asarray(obj, dtype=np.float32)
                    return arr.reshape(-1, 3)
                except Exception:
                    return None
        elif isinstance(cloud, (list, tuple)):
            try:
                arr = np.asarray(cloud, dtype=np.float32)
                return arr.reshape(-1, 3)
            except Exception:
                return None
        return None

    out = {}  # <-- dict로 반환
    for i, item in enumerate(side_cloud_items):
        # 입력은 단일 키 딕셔너리라고 가정
        if not isinstance(item, dict) or len(item) != 1:
            out[f"unknown_{i}"] = None
            continue
        obj_id = list(item.keys())[0]
        cloud = item[obj_id]

        pts_abs = _decode_cloud(cloud)
        if pts_abs is None or pts_abs.shape[0] < 4:
            out[obj_id] = None
            continue

        # 월드 → 카메라(회전 없음): 빼기
        Pc = pts_abs - t_cam_world.reshape(1, 3)

        # 앞(Z>0)
        Z = Pc[:, 2]
        valid = Z > 1e-4
        if not np.any(valid):
            out[obj_id] = None
            continue
        Pc = Pc[valid]; Z = Pc[:, 2]

        # 투영
        u = (Pc[:, 0] * fx / Z) + ppx
        v = (Pc[:, 1] * fy / Z) + ppy

        in_img = (u >= 0) & (u < (img_w - 1)) & (v >= 0) & (v < (img_h - 1))
        if not np.any(in_img):
            out[obj_id] = None
            continue
        u_i = u[in_img].astype(np.int32)
        v_i = v[in_img].astype(np.int32)
        Zp = Z[in_img]

        # 가림/깊이 일치 필터
        Dm = depth_raw[v_i, u_i].astype(np.float32) * depth_scale
        visible = (Dm == 0) | (np.abs(Dm - Zp) <= depth_tol_m) | (Zp <= Dm + depth_tol_m)
        if np.count_nonzero(visible) < 2:
            out[obj_id] = None
            continue

        u_vis = u[in_img][visible]
        v_vis = v[in_img][visible]

        # 박스 산출(마진+클리핑)
        x1 = int(max(0, np.floor(u_vis.min()) - margin_px))
        y1 = int(max(0, np.floor(v_vis.min()) - margin_px))
        x2 = int(min(img_w - 1, np.ceil(u_vis.max()) + margin_px))
        y2 = int(min(img_h - 1, np.ceil(v_vis.max()) + margin_px))
        out[obj_id] = None if (x2 <= x1 or y2 <= y1) else [x1, y1, x2, y2]

    return out



################################################################################################################



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



    # # 1) XY 평면에서 Convex Hull (윗면 윤곽)
    pts2d = cluster_pts[:, :2]
    pts2d_rounded = np.round(pts2d, decimals=3)
    unique_pts, _ = np.unique(pts2d_rounded, axis=0, return_index=True)

    if len(unique_pts) < 3:
        # 점이 너무 적으면 바운딩 박스 사용
        x_min, x_max = pts2d[:, 0].min(), pts2d[:, 0].max()
        y_min, y_max = pts2d[:, 1].min(), pts2d[:, 1].max()
        hull_xy = np.array([[x_min, y_min], [x_max, y_min], 
                        [x_max, y_max], [x_min, y_max]])
    else:
        try:
            # 중복 제거된 점으로 시도
            hull = ConvexHull(unique_pts, qhull_options="QJ")
            hull_xy = unique_pts[hull.vertices]
        except:
            try:
                # 원본 점으로 재시도
                hull = ConvexHull(pts2d, qhull_options="QJ")
                hull_xy = pts2d[hull.vertices]
            except:
                # 최종 폴백: 바운딩 박스
                x_min, x_max = pts2d[:, 0].min(), pts2d[:, 0].max()
                y_min, y_max = pts2d[:, 1].min(), pts2d[:, 1].max()
                hull_xy = np.array([[x_min, y_min], [x_max, y_min], 
                                [x_max, y_max], [x_min, y_max]])
            

    zs = cluster_pts[:, 2]
    # 100번째로 작은 Z를 바닥 높이로 사용
    k = max(0, min(100, len(zs) - 1))  # 7_30 추가 : 100번째가 없을 때 오류나는거 방지용
    z_min = np.partition(zs, k)[k]
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


    # 윤곽선 정보도 함께 반환
    boundary_limit = 5 # 소숫점 5째 자리까지만 보냄
    boundary_info = {
        "hull_vertices": [[round(p[0], boundary_limit), round(p[1], boundary_limit)] for p in hull_xy],  # 3자리 (mm 단위)
        "z_range": {
            "min": round(float(z_min), boundary_limit), 
            "max": round(float(z_max), boundary_limit)
        }
    }

    return np.array(side_pts), boundary_info


#################################################################################### Qhull 에서 2d 공면 오류 제거용 코드 7_30 추가, cluster_and_generate_surface 에서 사용
def _is_nearly_coplanar(pts: np.ndarray, rel_thresh: float = 1e-3) -> bool:
    """가장 작은 축 범위가 가장 큰 축의 rel_thresh(%) 이하이면 공면으로 간주."""
    if pts.size == 0:
        return True
    ranges = np.ptp(pts, axis=0)  # (dx, dy, dz)
    max_r = float(np.max(ranges))
    if max_r == 0.0:
        return True
    min_r = float(np.min(ranges))
    return (min_r < max_r * rel_thresh)

def _joggle_z(pts: np.ndarray, scale: float | None = None, seed: int | None = 42) -> np.ndarray:
    """z에 미세 난수 주입하여 랭크를 3으로 승격."""
    rng = np.random.default_rng(seed)
    pts = pts.astype(np.float32, copy=True)
    span = float(np.linalg.norm(np.ptp(pts, axis=0)))  # bbox diagonal
    sigma = (1e-4 * span) if scale is None else float(scale)  # ≈ 0.01%
    if sigma == 0.0:
        sigma = 1e-6
    noise = rng.normal(0.0, sigma, size=(pts.shape[0],))
    pts[:, 2] += noise.astype(pts.dtype)
    return pts

def _volume_robust(arr: np.ndarray) -> float:
    """ConvexHull.volume의 견고 래퍼: QbB QJ 시도 → 실패 시 2D area*ε 대체."""
    if arr.shape[0] < 4:
        return 0.0
    try:
        hull = ConvexHull(arr, qhull_options='QbB QJ')
        return float(hull.volume)
    except Exception:
        # 2D 대체: PCA 평면 투영 → 2D ConvexHull 면적 → 얇은 두께 ε 곱
        C = arr - arr.mean(axis=0, keepdims=True)
        try:
            # PCA (SVD)
            _, S, Vt = np.linalg.svd(C, full_matrices=False)
            basis = Vt[:2].T           # (3x2), 평면 기저
            uv = C @ basis             # (N,2)
            hull2d = ConvexHull(uv)    # 2D
            area = float(hull2d.area)  # SciPy: 2D에서 area=면적
        except Exception:
            return 0.0
        eps_thick = max(1e-4, 1e-3 * np.linalg.norm(np.ptp(arr, axis=0)))  # 매우 얇은 셸
        return area * eps_thick
    
#####################################################################################################################


@mcp.tool()
async def generate_side_point_clouds_for_storage(
    ctx: Context,
    objects_data: List[Dict],
    tool_pos: List[float],
    camera_index: int = 0,
    eps: float = 0.005,
    min_samples: int = 10,
    min_volume: float = 1e-6,
):
    """
    물체별 사이드 포인트 클라우드를 생성하여 저장용으로 반환
    
    Args:
        objects_data: 물체 정보 리스트 [{"id": "cup_001", "height": 0.1, "mask_base64": "..."}, ...]
        tool_pos: [x, y, z] 절대좌표계에서의 도구 위치
        
    Returns:
        List[Dict[str, Union[str, None]]]: [{"obj_id": "encoded_data"}, {"obj_id": None}, ...]
    """
    
    # 1. 전역 마스크 크기 가져오기
    global mask_shape
    
    # 2. 입력 검증
    if not objects_data:
        raise ValueError("objects_data cannot be empty")
    
    if len(tool_pos) != 3:
        raise ValueError("tool_pos must be [x, y, z] format")
        
    # 각 객체 데이터 유효성 검사
    for i, obj_data in enumerate(objects_data):
        required_keys = ["id", "height", "mask_base64"]
        for key in required_keys:
            if key not in obj_data:
                raise ValueError(f"objects_data[{i}] missing required key: {key}")
    
    # 2. 카메라 캡처
    app_context: AppContext = ctx.request_context.lifespan_context
    capture = app_context.camera_capture
    color_image, depth_image_raw = await asyncio.to_thread(capture.capture_aligned_frames)
    intrinsics = capture.get_intrinsics()
    depth_scale = capture.get_depth_scale()

    h, w = depth_image_raw.shape
    depth_m = depth_image_raw.astype(np.float32) * depth_scale
    
    # 로봇 좌표계 → 카메라 좌표계 변환
    tool_pos = [
        -tool_pos[1],   # -Y
        -tool_pos[0],  # -X
        -tool_pos[2]    # -Z
    ]

    

    # tool_pos를 numpy 배열로 변환
    tool_offset = np.array(tool_pos, dtype=np.float32)
    
    # 3. 결과 리스트 초기화
    result_list = []
    
    # 4. 각 물체별 독립 처리
    for i, obj_data in enumerate(objects_data):
        obj_id = obj_data["id"]
        mask_base64 = obj_data["mask_base64"]
        obj_height = obj_data["height"]
        
        try:
            print(f"[INFO] Processing object {obj_id} ({i+1}/{len(objects_data)})")
            
            # 4-1. 마스크 디코딩
            mask_np = decode_mask(mask_base64, tuple(mask_shape))
            
            # 4-2. 마스크된 포인트 추출
            pc_masked = []
            color_masked = []
            for y in range(h):
                for x in range(w):
                    if not mask_np[y, x]:
                        continue
                    d = depth_m[y, x]
                    if not (0.01 < d < 0.8):  # 감지 범위: 10cm ~ 80cm
                        continue
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
                    color = color_image[y, x].astype(np.float32) / 255.0
                    pc_masked.append([X, Y, Z])
                    color_masked.append(color)
            
            if len(pc_masked) < min_samples:
                print(f"  → {obj_id}: Too few points ({len(pc_masked)} < {min_samples})")
                result_list.append({obj_id: None})
                continue
            
            pc_masked_np = np.array(pc_masked, dtype=np.float32)
            col_masked_np = np.array(color_masked, dtype=np.float32)
            
            # 4-3. 클러스터링 및 사이드 포인트 클라우드 생성
            side_pcd,_ = cluster_and_generate_surface_for_single_object(
                pts=pc_masked_np,
                cols=col_masked_np,
                eps=eps,
                min_samples=min_samples,
                min_volume=min_volume,
                obj_height=obj_height
            )
            
            if side_pcd is None or len(side_pcd) == 0:
                print(f"  → {obj_id}: Failed to generate side point cloud")
                result_list.append({obj_id: None})
                continue
            
            # 4-4. 절대좌표 변환 (tool_pos 더하기)
            side_points_absolute = side_pcd + tool_offset
            
            # 4-5. 포인트만 인코딩
            encoded_points = base64.b64encode(side_points_absolute.astype(np.float32).tobytes()).decode()
            
            # 4-6. 결과에 추가
            result_list.append({obj_id: encoded_points})
            print(f"  → {obj_id}: Success ({len(side_pcd)} side points)")
            
        except Exception as e:
            print(f"  → {obj_id}: Error - {str(e)}")
            result_list.append({obj_id: None})
            continue
    
    print(f"[INFO] Completed processing {len(objects_data)} objects")
    return result_list


def cluster_and_generate_surface_for_single_object(pts: np.ndarray,
                                                  cols: np.ndarray,
                                                  eps: float = 0.005,
                                                  min_samples: int = 8,
                                                  min_volume: float = 7e-7,
                                                  obj_height: float = 0.1
                                                 ) -> np.ndarray:
    """
    단일 물체에 대한 클러스터링 및 사이드 포인트 클라우드 생성
    기존 cluster_and_generate_surface 로직을 단일 물체용으로 수정
    
    Returns:
        np.ndarray: side_points_array 또는 None
    """
    
    # Z 필터링
    zs = pts[:, 2]
    if len(zs) < 20:
        return None
        
    k = max(0, min(100, len(zs) - 1))
    z_ref = np.partition(zs, k)[k]
    mask_z = np.abs(zs - z_ref) < 0.02
    pts = pts[mask_z]
    cols = cols[mask_z]

    # DBSCAN 클러스터링
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(pts)
    max_label = labels.max()

    if max_label < 0:
        return None

    # 가장 큰 클러스터 선택
    best_cluster_idx = -1
    best_volume = 0
    
    for lbl in range(max_label + 1):
        mask = labels == lbl
        if not np.any(mask):
            continue

        cluster_pts = pts[mask]
        if len(cluster_pts) < 4:
            continue
    
        # 공면성 검사 및 z-jitter 적용
        if _is_nearly_coplanar(cluster_pts, rel_thresh=1e-3):
            cluster_in = _joggle_z(cluster_pts, scale=None, seed=42)
        else:
            cluster_in = cluster_pts

        vol = _volume_robust(cluster_in)
        if vol > best_volume and vol >= min_volume:
            best_volume = vol
            best_cluster_idx = lbl

    if best_cluster_idx < 0:
        return None

    # 최적 클러스터 처리
    mask = labels == best_cluster_idx
    cluster_pts = pts[mask]

    # 사이드 포인트 생성 (boundary_info는 무시)
    side_pts, boundary_info = generate_side_point_cloud(
        cluster_pts,
        num_edge_samples=35,
        num_height_samples=35,
        height_range=obj_height
    )

    return side_pts, boundary_info


@mcp.tool()
async def generate_integrated_point_cloud(
    ctx: Context,
    stored_side_clouds_str: Union[str, List],        # 함수 1 출력 (문자열 또는 리스트)
    target_object_id: str,              # "target_cup_001" 
    target_mask_base64: str,            # 타겟 마스크
    target_obj_height: float,           # 타겟 높이
    current_tool_pos: List[float],      # 현재 도구 위치 [x, y, z]
    tool_pos_feedback: List,
    crop_size_pixels: int = 480,        # 크롭 크기 (픽셀)
    camera_index: int = 0,
    eps: float = 0.005,
    min_samples: int = 10,
    min_volume: float = 1e-6,
):
    """
    저장된 사이드 포인트 클라우드들과 새로 생성한 타겟을 통합하여 최종 포인트 클라우드 생성
    
    Args:
        stored_side_clouds_str: 저장된 사이드 포인트 클라우드들 (문자열 또는 리스트)
        target_object_id: 타겟 물체 ID
        target_mask_base64: 타겟 마스크 (base64)
        target_obj_height: 타겟 물체 높이
        current_tool_pos: 현재 도구 위치 [x, y, z]
        crop_size_pixels: 타겟 마스크 중심 기준 크롭 크기
        
    Returns:
        dict: 최종 통합 포인트 클라우드 (base64 인코딩)
    """
    
    # 전역 마스크 크기 가져오기
    global mask_shape
    
    # 입력 검증
    if len(current_tool_pos) != 3:
        raise ValueError("current_tool_pos must be [x, y, z] format")
    
    # 로봇 좌표계 → 카메라 좌표계 변환
    current_tool_pos = [
        -current_tool_pos[1],   # -Y
        -current_tool_pos[0],  # -X
        -current_tool_pos[2]    # -Z
    ]

    current_tool_offset = np.array(current_tool_pos, dtype=np.float32)
    
    print("[INFO] Starting integrated point cloud generation...")
    
    # 1. 저장된 사이드 포인트 클라우드들 파싱 및 디코딩
    print("[INFO] Step 1: Parsing stored side point clouds...")
    stored_points_list = []
    
    if stored_side_clouds_str:
        try:
            # 입력 타입에 따라 처리 (확장된 버전)
            if isinstance(stored_side_clouds_str, str):
                # 문자열인 경우: 1차 JSON 파싱
                result_list = json.loads(stored_side_clouds_str)
            elif isinstance(stored_side_clouds_str, list):
                # 리스트인 경우: 내부 요소 타입 확인
                if len(stored_side_clouds_str) > 0 and isinstance(stored_side_clouds_str[0], dict):
                    # 이미 파싱된 딕셔너리들의 리스트 (첫 번째 함수 출력)
                    parsed_objects = stored_side_clouds_str
                    print(f"입력 처리 완료: {len(parsed_objects)}개 항목 (이미 파싱됨)")
                else:
                    # 문자열들의 리스트 (기존 방식)
                    result_list = stored_side_clouds_str
            else:
                raise ValueError("stored_side_clouds_str must be string, list of strings, or list of dicts")
            
            # 2차 JSON 파싱 (딕셔너리 리스트가 아닌 경우만)
            if 'result_list' in locals():  # result_list가 정의된 경우만
                print(f"입력 처리 완료: {len(result_list)}개 항목")
                parsed_objects = []
                for item in result_list:
                    if isinstance(item, str):
                        parsed_item = json.loads(item)
                        parsed_objects.append(parsed_item)
                    else:
                        parsed_objects.append(item)
            
            # 타겟 ID 제거 및 디코딩 (공통 처리)
            for obj_dict in parsed_objects:
                for obj_id, encoded_data in obj_dict.items():
                    if obj_id == target_object_id:
                        print(f"[INFO] Skipping stored target: {obj_id}")
                        continue
                    if encoded_data is not None:
                        try:
                            # base64 디코딩
                            points_bytes = base64.b64decode(encoded_data)
                            points_array = np.frombuffer(points_bytes, dtype=np.float32)
                            points_3d = points_array.reshape(-1, 3)
                            stored_points_list.append(points_3d)
                            print(f"[INFO] Loaded stored object {obj_id}: {len(points_3d)} points")
                        except Exception as e:
                            print(f"[WARNING] Failed to decode {obj_id}: {e}")

            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[WARNING] Parsing failed: {e}")
    
    # 2. 현재 카메라 캡처 및 전체 포인트 클라우드 생성 (벡터화 최적화)
    print("[INFO] Step 2: Capturing current camera frame...")
    app_context: AppContext = ctx.request_context.lifespan_context
    capture = app_context.camera_capture
    color_image, depth_image_raw = await asyncio.to_thread(capture.capture_aligned_frames)
    intrinsics = capture.get_intrinsics()
    depth_scale = capture.get_depth_scale()

    h, w = depth_image_raw.shape
    depth_m = depth_image_raw.astype(np.float32) * depth_scale
    
    # ===== 벡터화 최적화: 전체 포인트 클라우드 생성 =====
    print("[INFO] Generating full point cloud (vectorized)...")
    
    # 유효한 depth 마스크 생성
    valid_depth_mask = (depth_m > 0.01) & (depth_m < 0.8)
    valid_y, valid_x = np.where(valid_depth_mask)
    
    if len(valid_y) == 0:
        raise ValueError("No valid points in current frame")
    
    # 배치로 3D 좌표 변환
    valid_depths = depth_m[valid_y, valid_x]
    
    # intrinsics 파라미터 추출
    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy
    
    # 벡터화된 3D 좌표 계산
    X = (valid_x - ppx) * valid_depths / fx
    Y = (valid_y - ppy) * valid_depths / fy
    Z = valid_depths
    
    current_points_np = np.column_stack([X, Y, Z]).astype(np.float32)
    
    # 색상 정보 추출
    current_colors_np = color_image[valid_y, valid_x].astype(np.float32) / 255.0
    
    # 3. 새 타겟 사이드 포인트 클라우드 생성 (벡터화 최적화)
    print("[INFO] Step 3: Generating new target side point cloud...")
    
    # 타겟 마스크 디코딩
    target_mask = decode_mask(target_mask_base64, tuple(mask_shape))
    
    # ===== 벡터화 최적화: 타겟 마스크 포인트 추출 =====
    # 마스크와 depth 조건을 모두 만족하는 픽셀 찾기
    target_valid_mask = target_mask & valid_depth_mask
    target_y, target_x = np.where(target_valid_mask)
    
    if len(target_y) < min_samples:
        print(f"[WARNING] Too few target points: {len(target_y)} < {min_samples}")
        new_target_points = np.empty((0, 3))
        target_boundary_info = None
    else:
        # 타겟 포인트들의 depth 값
        target_depths = depth_m[target_y, target_x]
        
        # 벡터화된 3D 좌표 계산
        target_X = (target_x - ppx) * target_depths / fx
        target_Y = (target_y - ppy) * target_depths / fy
        target_Z = target_depths
        
        target_points_np = np.column_stack([target_X, target_Y, target_Z]).astype(np.float32)
        target_colors_np = color_image[target_y, target_x].astype(np.float32) / 255.0
        
        # 타겟 사이드 포인트 생성
        new_target_side, target_boundary_info = cluster_and_generate_surface_for_single_object(
            pts=target_points_np,
            cols=target_colors_np,
            eps=eps,
            min_samples=min_samples,
            min_volume=min_volume,
            obj_height=target_obj_height
        )
        
        if new_target_side is not None:
            # 절대좌표 변환
            new_target_points = new_target_side + current_tool_pos
            # new_target_points = new_target_side
            print(f"[INFO] New target side points: {len(new_target_points)} (absolute coordinates)")
        else:
            print("[WARNING] Failed to generate target side points")
            new_target_points = np.empty((0, 3))
            target_boundary_info = None
    



    # 4. 모든 포인트 병합 (크롭 전)
    print("[INFO] Step 4: Merging all point clouds...")
    
    # Phase 1: 현재 데이터들 먼저 병합 (상대좌표)
    current_data_points = [current_points_np + current_tool_offset]   # 환경 포인트
    current_data_colors = [current_colors_np]
    
    # 새 타겟 사이드 포인트들 추가 (상대좌표)
    if len(new_target_points) > 0:
        target_colors_all = np.tile([0.2, 0.8, 0.2], (len(new_target_points), 1))  # 초록색
        current_data_points.append(new_target_points)
        current_data_colors.append(target_colors_all)
        print(f"[INFO] Added new target side points: {len(new_target_points)} (relative coordinates)")
    
    # Phase 2: 현재 데이터 병합 후 오프셋 적용
    current_merged_relative = np.vstack(current_data_points)
    current_colors_merged = np.vstack(current_data_colors)
    current_merged_absolute = current_merged_relative  # 한번에 절대좌표 변환
    print(f"[INFO] Current data merged and converted to absolute: {len(current_merged_absolute)} points")
    
    # Phase 3: 저장된 절대좌표 데이터와 최종 병합
    merged_points = [current_merged_absolute]
    merged_colors = [current_colors_merged]
    
    # 저장된 사이드 포인트들 추가 (이미 절대좌표이므로 변환하지 않음)
    stored_points_count = 0
    if stored_points_list:
        stored_points_all = np.vstack(stored_points_list)  # 이미 절대좌표
        stored_colors_all = np.tile([0.8, 0.2, 0.2], (len(stored_points_all), 1))  # 빨간색
        merged_points.append(stored_points_all)
        merged_colors.append(stored_colors_all)
        stored_points_count = len(stored_points_all)
        print(f"[INFO] Added stored side points: {stored_points_count} (already in absolute coordinates)")
    
    # 병합 완료
    merged_points_all = np.vstack(merged_points)
    merged_colors_all = np.vstack(merged_colors)
    print(f"[INFO] Total merged points: {len(merged_points_all)}")

    # 5. 타겟 마스크 중심 기준으로 크롭
    print("[INFO] Step 5: Cropping around target mask center...")
    
    # 마스크 중심점 계산
    mask_y, mask_x = np.where(target_mask)
    if len(mask_y) == 0:
        raise ValueError("Target mask is empty")
    
    center_y = int(np.mean(mask_y))
    center_x = int(np.mean(mask_x))
    
    # 타겟 중심을 3D 좌표로 변환 (크롭 기준점)
    center_depth = depth_m[center_y, center_x]
    if center_depth > 0:
        center_3d_cam_x = (center_x - ppx) * center_depth / fx
        center_3d_cam_y = (center_y - ppy) * center_depth / fy
        center_3d_cam = np.array([center_3d_cam_x, center_3d_cam_y, center_depth])
        center_3d_abs = center_3d_cam + current_tool_offset
    else:
        # depth가 0이면 타겟 포인트들의 평균 사용
        if len(new_target_points) > 0:
            center_3d_abs = np.mean(new_target_points, axis=0)
        else:
            center_3d_abs = current_tool_offset
    
    
    # 크롭 범위 계산 (3D 공간에서)
    # 480 픽셀 → 실제 거리로 변환 (대략적으로 계산)
    pixel_size = 0.001  # 1mm per pixel (대략값, 거리에 따라 다름)
    crop_radius = (crop_size_pixels / 2) * pixel_size
    
    # 3D 크롭 (타겟 중심에서 일정 범위 내의 포인트들만) - 벡터화
    distances = np.linalg.norm(merged_points_all[:, :2] - center_3d_abs[:2], axis=1)
    crop_mask = distances <= crop_radius
    
    cropped_points = merged_points_all[crop_mask]
    cropped_colors = merged_colors_all[crop_mask]
    
    print(f"[INFO] Points after cropping: {len(cropped_points)} (crop radius: {crop_radius:.3f}m)")
    
    if len(cropped_points) == 0:
        print("[WARNING] No points after cropping, using all points")
        cropped_points = merged_points_all
        cropped_colors = merged_colors_all
    
    # 6. 다운샘플링 (voxel_size 조정으로 성능 향상)
    # print("[INFO] Step 6: Downsampling...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cropped_points)
    pcd.colors = o3d.utility.Vector3dVector(cropped_colors)
    pcd = pcd.voxel_down_sample(voxel_size=0.003)  # 0.002 → 0.003으로 증가
    
    final_points = np.asarray(pcd.points)
    final_colors = np.asarray(pcd.colors)
    
    print(f"[INFO] Final point cloud: {len(final_points)} points (after downsampling)")
    
# 7. 출력에 boundary_info 포함 (오프셋 적용)
    result = {
        "points": base64.b64encode(final_points.astype(np.float32).tobytes()).decode(),
        "colors": base64.b64encode(final_colors.astype(np.float32).tobytes()).decode()
    }
    
    # target_boundary_info가 있으면 오프셋 적용 후 추가
    if target_boundary_info is not None:
        # 🆕 boundary_info에 오프셋 적용
        boundary_info_absolute = target_boundary_info.copy()
        
        # hull_vertices에 XY 오프셋 적용
        if 'hull_vertices' in boundary_info_absolute:
            hull_vertices_absolute = []
            for vertex in boundary_info_absolute['hull_vertices']:
                x_rel, y_rel = float(vertex[0]), float(vertex[1])
                # XY 오프셋 적용 (Z는 hull_vertices에 없음)
                x_abs = x_rel + current_tool_offset[0]
                y_abs = y_rel + current_tool_offset[1]
                hull_vertices_absolute.append([
                    round(x_abs, 5),
                    round(y_abs, 5)
                ])
            boundary_info_absolute['hull_vertices'] = hull_vertices_absolute
        
        # z_range에 Z 오프셋 적용
        if 'z_range' in boundary_info_absolute:
            z_min_rel = boundary_info_absolute['z_range']['min']
            z_max_rel = boundary_info_absolute['z_range']['max']
            # Z 오프셋 적용
            z_min_abs = z_min_rel + current_tool_offset[2]
            z_max_abs = z_max_rel + current_tool_offset[2]
            boundary_info_absolute['z_range'] = {
                'min': round(z_min_abs, 5),
                'max': round(z_max_abs, 5)
            }
        
        result["target_boundary_info"] = base64.b64encode(pickle.dumps(boundary_info_absolute)).decode()
    else:
        result["target_boundary_info"] = None
    
    return result



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
        # logging.error(f"Error capturing JPG from camera {camera_index}: {e}", exc_info=True)
        return ''




# # ---- 서버 실행 ----
# if __name__ == "__main__":
#     print("Starting MCP server...")
#     try:
#         mcp.run()
#     except KeyboardInterrupt:
#         print("\n[INFO] MCP server interrupted by user. Shutting down...")
#     finally:
#         # Ensure global is declared once and before any operations
#         # global _camera_capture_instance  # Declare global once before any usage
#         if _camera_capture_instance and _camera_capture_instance.is_running:
#             _camera_capture_instance.stop()
#         print("[INFO] Server shutdown complete.")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import os

def _extract_pc_and_colors_from_mask(mask: np.ndarray,
                                     color_bgr: np.ndarray,
                                     depth_raw: np.ndarray,
                                     intr,
                                     depth_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """SAM 마스크 + RGBD로부터 포인트(XYZ)와 색상(0~1) 추출. (원본 함수들 변경 없음)"""
    if mask.dtype != bool:
        mask = mask > 0
    ys, xs = np.where(mask)
    if len(ys) < 4:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.float32)

    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    z = depth_m[ys, xs]
    valid = (z > 0.01) & (z < 0.8)  # 네 코드 범위에 맞춤
    if np.count_nonzero(valid) < 4:
        return np.zeros((0,3), np.float32), np.zeros((0,3), np.float32)

    xs = xs[valid].astype(np.float32); ys = ys[valid].astype(np.float32); z = z[valid].astype(np.float32)
    fx, fy = float(intr.fx), float(intr.fy)
    cx, cy = float(intr.ppx), float(intr.ppy)
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    pts = np.column_stack([X, Y, z]).astype(np.float32)

    cols = color_bgr[ys.astype(int), xs.astype(int)].astype(np.float32) / 255.0
    return pts, cols

def _save_pc_image(pc: np.ndarray, path: str, title: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    if pc.size:
        ax.scatter(pc[:,0], pc[:,1], pc[:,2], s=1, c='0.35')  # 회색 포인트
        rng = (pc.max(axis=0) - pc.min(axis=0)).astype(float)
        rng[rng == 0] = 1.0
        ax.set_box_aspect(rng)
    # 축/라벨/제목 모두 제거
    ax.set_axis_off()
    fig.tight_layout(pad=0.0)
    fig.savefig(path, dpi=220, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def _save_outline_from_boundary(boundary_info: dict, path: str, pc_xy_for_bg: np.ndarray | None = None):
    """boundary_info['hull_vertices']로 2D 윤곽 저장. 배경 포인트는 연회색, 윤곽선은 강조색."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hull_xy = np.array(boundary_info.get("hull_vertices", []), dtype=np.float32)

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)

    # 배경 포인트(연회색)
    if pc_xy_for_bg is not None and len(pc_xy_for_bg) > 0:
        ax.scatter(pc_xy_for_bg[:,0], pc_xy_for_bg[:,1], s=1, c='0.65')

    # 윤곽선(강조색)
    if hull_xy.shape[0] >= 3:
        closed = np.vstack([hull_xy, hull_xy[0]])
        ax.plot(closed[:,0], closed[:,1], linewidth=3, c='crimson')  # 포인트와 다른 색

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # 축 전부 제거
    fig.tight_layout(pad=0.0)
    fig.savefig(path, dpi=220, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def _save_outline_xy(pc: np.ndarray, path: str, boundary_info: dict | None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if boundary_info and "hull_vertices" in boundary_info:
        hull_xy = np.array(boundary_info["hull_vertices"], dtype=np.float32)
    else:
        # 윤곽만 필요하므로 샘플 수 작게 호출
        _, bi = generate_side_point_cloud(pc, num_edge_samples=4, num_height_samples=2, height_range=0.01)
        hull_xy = np.array(bi["hull_vertices"], dtype=np.float32) if bi else None

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)

    # 배경 포인트(연회색)
    if pc.size:
        pts2d = pc[:, :2]
        ax.scatter(pts2d[:,0], pts2d[:,1], s=1, c='0.65')

    # 윤곽선(강조색)
    if hull_xy is not None and len(hull_xy) >= 3:
        closed = np.vstack([hull_xy, hull_xy[0]])
        ax.plot(closed[:,0], closed[:,1], linewidth=3, c='crimson')

    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # 축 전부 제거
    fig.tight_layout(pad=0.0)
    fig.savefig(path, dpi=220, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def _o3d_from_numpy(points: np.ndarray,
                    colors: np.ndarray | None = None,
                    default_color=(0.5, 0.5, 0.5),
                    voxel: float | None = None):
    import open3d as o3d
    if points is None or len(points) == 0:
        return None
    pc = o3d.geometry.PointCloud()
    pts64 = points.astype(np.float64, copy=False)
    pc.points = o3d.utility.Vector3dVector(pts64)
    if colors is not None and len(colors) == len(points):
        cols = np.clip(colors.astype(np.float64, copy=False), 0.0, 1.0)
        pc.colors = o3d.utility.Vector3dVector(cols)
    else:
        col = np.array(default_color, dtype=np.float64)[None, :]
        pc.colors = o3d.utility.Vector3dVector(np.repeat(col, len(points), axis=0))
    if voxel and voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    return pc

def show_clouds_o3d(raw_pc: np.ndarray,
                    side_pc: np.ndarray | None = None,
                    raw_cols: np.ndarray | None = None,
                    boundary_info: dict | None = None,
                    side_color=(0.9, 0.1, 0.1),
                    voxel_raw: float = 0.0,
                    voxel_side: float = 0.0,
                    background_color=(1.0, 1.0, 1.0),   # ← 기본: 흰색
                    window_name="Point Clouds"):
    import open3d as o3d

    def _o3d_from_numpy(points, colors=None, default_color=(0.6,0.6,0.6), voxel=0.0):
        if points is None or len(points) == 0:
            return None
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points.astype(np.float64, copy=False))
        if colors is not None and len(colors) == len(points):
            pc.colors = o3d.utility.Vector3dVector(np.clip(colors.astype(np.float64, copy=False), 0.0, 1.0))
        else:
            col = np.array(default_color, dtype=np.float64)[None, :]
            pc.colors = o3d.utility.Vector3dVector(np.repeat(col, len(points), axis=0))
        if voxel and voxel > 0:
            pc = pc.voxel_down_sample(voxel)
        return pc

    geoms = []
    pc_raw = _o3d_from_numpy(raw_pc, raw_cols, default_color=(0.6,0.6,0.6), voxel=voxel_raw)
    if pc_raw is not None: geoms.append(pc_raw)

    if side_pc is not None and len(side_pc) > 0:
        pc_side = _o3d_from_numpy(side_pc, None, default_color=side_color, voxel=voxel_side)
        geoms.append(pc_side)

    if boundary_info and "hull_vertices" in boundary_info:
        hv = np.asarray(boundary_info["hull_vertices"], dtype=float)
        if hv.shape[0] >= 3:
            import open3d as o3d
            z_top = float(boundary_info.get("z_range", {}).get(
                "max", (np.median(raw_pc[:,2]) if raw_pc is not None and len(raw_pc)>0 else 0.0)
            ))
            pts3d = np.column_stack([hv, np.full((hv.shape[0],), z_top, dtype=float)])
            lines = np.array([[i, (i+1) % len(pts3d)] for i in range(len(pts3d))], dtype=np.int32)
            line = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts3d),
                                        lines=o3d.utility.Vector2iVector(lines))
            line.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.2]]), (len(lines), 1)))
            geoms.append(line)

    if not geoms:
        print("[WARN] Nothing to visualize.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1120, height=840, visible=True)
    for g in geoms: vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.5
    opt.background_color = np.array(background_color, dtype=float)  # ← 흰색 등 원하는 색
    vis.run()    # 창 닫을 때까지 대기 (순차 표시에 필요)
    vis.destroy_window()



def run_demo_visualize_noargs():
    """
    인자 없이 실행. (원본 함수 수정 없음)
    파이프라인: DINO→SAM→마스크 → 원본PC 저장 → (클러스터 함수 호출) boundary_info로 윤곽 저장 → side PC 저장
    """
    TEXT_PROMPT = "can"   # ← 네가 원하는 프롬프트로 이 상수만 바꾸면 됨
    OUTDIR = "demo_outputs"
    OBJ_HEIGHT = 0.13

    # 모델/카메라
    wrapper = get_model_wrapper()
    capture = get_camera_capture()

    color_image, depth_image_raw = capture.capture_aligned_frames()
    if color_image is None or depth_image_raw is None:
        raise RuntimeError("Failed to capture frames.")
    intr = capture.get_intrinsics()
    depth_scale = capture.get_depth_scale()

    # DINO+SAM (원래 predict: DINO 후 SAM 마스크 반환)
    masks_tensor, boxes_tensor, _, scores_tensor = wrapper.predict(color_image, f"{TEXT_PROMPT}.")
    if boxes_tensor.size(0) == 0:
        raise RuntimeError("No detections from DINO+SAM.")
    best = int(torch.argmax(scores_tensor).item())
    mask_np = (masks_tensor[best].detach().cpu().numpy() > 0.2)

    # (1) 원본 포인트클라우드 (마스크 기반)
    raw_pc, raw_cols = _extract_pc_and_colors_from_mask(mask_np, color_image, depth_image_raw, intr, depth_scale)
    if raw_pc.shape[0] < 4:
        raise RuntimeError("Too few 3D points.")
    _save_pc_image(raw_pc, os.path.join(OUTDIR, "1_raw_point_cloud.png"), "Raw Point Cloud")

    # (2)(3) 네 “클러스터+Z밴드” 경로로 윤곽/사이드 생성  ← ★ 원본 함수 그대로 사용
    side_pc, boundary_info = cluster_and_generate_surface_for_single_object(
        pts=raw_pc,
        cols=raw_cols,
        eps=0.005,
        min_samples=8,
        min_volume=7e-7,
        obj_height=OBJ_HEIGHT
    )

    if side_pc is None or boundary_info is None:
        raise RuntimeError("Failed to generate side point cloud / boundary.")

    # 윤곽선 저장 (boundary_info 사용; 배경으로는 원본 XY 산점도 깔기)
    _save_outline_from_boundary(boundary_info, os.path.join(OUTDIR, "2_outline_convexhull.png"),
                                pc_xy_for_bg=raw_pc[:, :2])

    # (3) 사이드 포인트클라우드 저장
    _save_pc_image(side_pc, os.path.join(OUTDIR, "3_side_point_cloud.png"), "Side Point Cloud")

    print(f"[INFO] Saved images to: {OUTDIR}")

    # (선택) Open3D로 순차 표시
    # 1) side 하기 전 (원본 포인트만)
    show_clouds_o3d(raw_pc=raw_pc,
                    side_pc=None,
                    raw_cols=raw_cols,
                    boundary_info=None,
                    background_color=(1.0, 1.0, 1.0),   # 흰 배경
                    window_name="Before SIDE (raw only)")

    # 2) side 한 후 (원본 + side + 윤곽선)
    show_clouds_o3d(raw_pc=raw_pc,
                    side_pc=side_pc,
                    raw_cols=raw_cols,
                    boundary_info=boundary_info,
                    background_color=(1.0, 1.0, 1.0),   # 흰 배경
                    window_name="After SIDE (raw + side + outline)")







if __name__ == "__main__":
    run_demo_visualize_noargs()
















# def test_dino_labeled_image():
#     """
#     DINO 라벨링 함수 테스트
#     """
    
#     # 모델 및 카메라 초기화
#     print("모델 초기화 중...")
#     wrapper = GroundedSAM2Wrapper()
    
#     print("카메라 초기화 중...")
#     capture = RealSenseCapture()  # 실제 카메라 클래스로 수정
#     if not capture.start():
#         raise RuntimeError("카메라 시작 실패")
    
#     try:
#         # 프레임 캡처
#         print("프레임 캡처 중...")
#         color_image, *_ = capture.capture_aligned_frames()
#         if color_image is None:
#             raise RuntimeError("프레임 캡처 실패")
        
#         # DINO 라벨링된 이미지 생성
#         print("DINO 추론 중...")
#         annotated_image = run_dino_inference_simple(color_image, wrapper)
        
#         # 결과 저장
#         print("결과 저장 중...")
        
#         # 1. 현재 디렉토리에 저장
#         cv2.imwrite('dino_labeled_result.jpg', annotated_image)
        
#         # 2. 이 대화에서 접근 가능한 디렉토리에도 복사 저장
#         # (실제 경로는 환경에 따라 조정 필요)
#         import shutil
#         import os
        
#         # 여러 가능한 경로들 시도
#         possible_paths = [
#             './uploads/',
#             './images/',
#             './debug/',
#             './',  # 현재 디렉토리
#         ]
        
#         saved_paths = []
#         for path in possible_paths:
#             try:
#                 if not os.path.exists(path):
#                     os.makedirs(path, exist_ok=True)
                
#                 dest_file = os.path.join(path, 'dino_labeled_debug.jpg')
#                 cv2.imwrite(dest_file, annotated_image)
#                 saved_paths.append(dest_file)
#                 print(f"이미지 저장됨: {dest_file}")
                
#             except Exception as e:
#                 print(f"경로 {path}에 저장 실패: {e}")
        
#         # 원본 이미지도 저장 (비교용)
#         try:
#             cv2.imwrite('original_image_debug.jpg', color_image)
#             for path in possible_paths:
#                 try:
#                     dest_file = os.path.join(path, 'original_image_debug.jpg')
#                     cv2.imwrite(dest_file, color_image)
#                     print(f"원본 이미지 저장됨: {dest_file}")
#                     break
#                 except:
#                     continue
#         except Exception as e:
#             print(f"원본 이미지 저장 실패: {e}")
        
#         # base64 인코딩도 테스트
#         success, jpg_bytes = cv2.imencode('.jpg', annotated_image)
#         if success:
#             jpg_base64 = base64.b64encode(jpg_bytes.tobytes()).decode('utf-8')
#             result_data_url = f"data:image/jpg;base64,{jpg_base64}"
#             print(f"Base64 길이: {len(result_data_url)} 문자")
            
#             # base64를 파일로도 저장 (디버깅용)
#             with open('dino_labeled_base64.txt', 'w') as f:
#                 f.write(result_data_url)
        
#         print("✅ 테스트 완료! 결과 파일:")
#         print("  - dino_labeled_result.jpg (현재 디렉토리)")
#         if saved_paths:
#             for path in saved_paths:
#                 print(f"  - {path} (디버그용)")
#         print("  - dino_labeled_base64.txt")
#         print("\n📸 이제 저장된 이미지를 업로드해서 분석받을 수 있습니다!")
#         print("   파일명: dino_labeled_debug.jpg 또는 dino_labeled_result.jpg")
        
#     except Exception as e:
#         print(f"❌ 테스트 실패: {e}")
#         import traceback
#         traceback.print_exc()
        
#     finally:
#         print("리소스 해제 중...")
#         capture.stop()


# def run_dino_inference_simple(image_np_bgr, wrapper):
#     """
#     wrapper의 predict_dino_only()를 사용하여 DINO만 실행하고 숫자 라벨만 표시한 이미지 반환
#     SAM 없이 DINO만 사용하므로 효율적!
#     """
    
#     # GPU 텐서 변수 초기화
#     boxes_tensor = None
#     scores_tensor = None
    
#     try:
#         # DINO만 실행 (SAM 없음) - 효율적!
#         text_prompt = "object"
        
#         boxes_tensor, scores_tensor, phrases_list = wrapper.predict_dino_only(
#             image_np_bgr,
#             text_prompt
#         )
        
#         print(f"검출된 객체 수: {boxes_tensor.size(0)}")
        
#         if boxes_tensor.size(0) == 0:
#             print("검출된 객체가 없습니다.")
#             return image_np_bgr.copy()
        
#         # 박스를 CPU로 이동
#         input_boxes = boxes_tensor.detach().cpu().numpy()
#         class_names = phrases_list
        
#         print(f"검출된 객체들: {class_names}")
        
#         # GPU 텐서들 즉시 삭제
#         del boxes_tensor, scores_tensor
#         boxes_tensor = None
#         scores_tensor = None
        
#         # GPU 메모리 강제 정리
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
        
#         # 시각화 (단일 색상으로 VLM 편향 제거)
#         annotated_frame = image_np_bgr.copy()
        
#         if len(input_boxes) > 0:
#             # 단일 색상: 밝은 초록
#             box_color = (0, 0, 255)  # BGR 형식
            
#             # 각 박스와 라벨 그리기
#             for i, box in enumerate(input_boxes):
#                 x1, y1, x2, y2 = box.astype(int)
                
#                 print(f"박스 {i}: ({x1}, {y1}, {x2}, {y2}) - {class_names[i]}")
                
#                 # 박스 그리기
#                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 1)
                
#                 # 숫자 라벨 그리기 (적당한 크기로 조정)
#                 label = str(i)
#                 font_scale = 0.7  # 1.2 → 0.8로 축소
#                 thickness = 2     # 3 → 2로 축소
#                 label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
#                 # 라벨 배경 크기도 축소
#                 padding = 4  # 15 → 8로 축소
#                 cv2.rectangle(annotated_frame, 
#                              (x1, y1 - label_size[1] - padding), 
#                              (x1 + label_size[0] + padding, y1), 
#                              (0, 0, 0), -1)  # 검은색 배경
                
#                 # 라벨 텍스트 그리기 (위치도 조정)
#                 cv2.putText(annotated_frame, label, 
#                            (x1 + 2, y1 - 2),  # 7,7 → 4,4로 축소
#                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
#         return annotated_frame
        
#     except Exception as e:
#         logging.error(f"DINO inference failed: {e}")
#         print(f"추론 실패: {e}")
#         return image_np_bgr.copy()
        
#     finally:
#         # 최종 정리
#         try:
#             if boxes_tensor is not None:
#                 del boxes_tensor
#             if scores_tensor is not None:
#                 del scores_tensor
#         except:
#             pass
#         gc.collect()


# if __name__ == "__main__":
#     test_dino_labeled_image()