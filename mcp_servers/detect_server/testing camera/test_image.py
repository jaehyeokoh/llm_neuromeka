# %% [markdown]
# 환경 설정 & 모델 로드

# %%

import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from PIL import Image
import sys

print("Current working dir:", os.getcwd())

GROUND_SAM2_PATH = os.path.join(os.getcwd(), "Grounded-SAM-2")

sys.path.append(GROUND_SAM2_PATH)
# transformers 기반 (huggingface) GroundingDINO
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# 사용자 정의 color map (예시)
CUSTOM_COLORS = [
    "#4365F0",  # blue
    "#F04343",  # red
    "#43F059",  # green
]

# 1) 디바이스 설정 (GPU 가능 시 GPU, 아니면 CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2) bfloat16 / TF32 최적화 (Ampere 이후 GPU)
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
if DEVICE == "cuda":
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

# 3) SAM2 모델 설정 (가장 가벼운 Tiny)
SAM2_CHECKPOINT = os.path.join(GROUND_SAM2_PATH, "checkpoints", "sam2.1_hiera_tiny.pt")

SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"

# 실제로는 build_sam2, SAM2ImagePredictor import 후 사용
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

print("Loading SAM2 model...")
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 loaded.")

# 4) Grounding DINO (Tiny)
GROUNDING_MODEL = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(GROUNDING_MODEL)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_MODEL).to(DEVICE)
print("Grounding DINO (Tiny) loaded on", DEVICE)

print("\n✅ All models are loaded and ready!")

# %% [markdown]
# 추론 함수 정의

# %%
def run_inference(
    # img_path: str,
    image,
    text_prompt: str = "box, cup.",
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    show_window: bool = True
):
    """
    img_path     : 추론할 이미지 경로
    text_prompt  : "car. tire."처럼 Grounding DINO에 사용할 텍스트 프롬프트 (소문자+마침표 권장)
    box_threshold: 박스 임곗값
    text_threshold: 텍스트 임곗값
    show_window  : True면 opencv 창에 결과 출력 (cv2.imshow)
    dump_json    : True면 결과를 JSON 파일로 저장
    output_dir   : JSON 저장 시 디렉토리

    반환값: annotated_frame (최종 시각화 이미지, numpy array)
    """

    # 1. 이미지 로드 (PIL, NumPy)
    # image_pil = Image.open(img_path).convert("RGB")
    image_pil = image
    image_np = np.array(image_pil)  # SAM2 predictor용

    # 2. SAM2 predictor 준비
    sam2_predictor.set_image(image_np)

    # 3. Grounding DINO 추론 (박스 / 라벨 / 스코어)
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]]
    )

    # # 결과가 비어있을 수 있으니 방어적 체크
    # if len(results) == 0 or "boxes" not in results[0]:
    #     print("No bounding boxes found by Grounding DINO!")
    #     return None

    input_boxes = results[0]["boxes"].cpu().numpy()   # shape (n,4)
    class_names = results[0]["labels"]
    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_ids = np.arange(len(class_names))
    print(input_boxes)

    # 4. SAM2로 마스크 예측
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,  # True면 여러 후보 마스크를 만듦
    )

    # shape 변환
    if masks.ndim == 4:
        # (n,1,H,W) -> (n,H,W)
        masks = masks.squeeze(1)

    # 5. 시각화
    img_bgr = image
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    labels_for_display = [f"{cls} {c:.2f}" for cls, c in zip(class_names, confidences)]

    # color
    from supervision.draw.color import ColorPalette
    color_palette = ColorPalette.from_hex(CUSTOM_COLORS)


    # box
    box_annotator = sv.BoxAnnotator(color=color_palette)
    annotated_frame = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)

    # label
    label_annotator = sv.LabelAnnotator(color=color_palette)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels_for_display
    )

    # mask
    mask_annotator = sv.MaskAnnotator(color=color_palette)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

    # 6. 결과 창에 띄우기
    if show_window:
        window_name = "Grounded SAM2 Demo"
        cv2.imshow(window_name, annotated_frame)

        while True:
            # 1) 50ms 대기
            key = cv2.waitKey(50)

            # 2) 사용자가 'ESC' 키 누른 경우 (optional)
            if key == 27:  # ESC ASCII code
                break

            # 3) 창이 닫혔는지 확인
            #    getWindowProperty가 음수가 되면 창이 닫혔다고 판단
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()


print("✅ Inference function `run_inference` is ready!")

# %%
import numpy as np
from PIL import Image
import cv2 # OpenCV 추가

def run_inference(
    image, # 입력은 NumPy 배열 (BGR)
    text_prompt: str = "box. cup",
    box_threshold: float = 0.4,
    text_threshold: float = 0.3,
    show_window: bool = True
):
    """
    # ... (docstring은 동일) ...
    """

    # 1. 이미지 로드 및 변환 (NumPy BGR -> PIL RGB)
    image_np_bgr = image  # 입력된 NumPy 배열 (BGR 형식임을 명시)

    # BGR NumPy 배열을 RGB NumPy 배열로 변환
    image_np_rgb = cv2.cvtColor(image_np_bgr, cv2.COLOR_BGR2RGB)

    # RGB NumPy 배열을 PIL 이미지 객체로 변환
    image_pil = Image.fromarray(image_np_rgb)

    # SAM2 predictor는 NumPy 배열을 사용하므로 원본 BGR 배열이나 RGB 배열을 사용
    # (어떤 형식을 선호하는지는 SAM2 모델에 따라 다를 수 있으나, 보통 NumPy면 괜찮음)
    image_np_for_sam = image_np_bgr # 또는 image_np_rgb

    # 2. SAM2 predictor 준비
    sam2_predictor.set_image(image_np_for_sam) # NumPy 배열 전달

    # 3. Grounding DINO 추론 (PIL 이미지 사용)
    inputs = processor(images=image_pil, text=text_prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    # target_sizes 계산 시 PIL 이미지의 .size 사용 (이제 정상 작동)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]] # PIL 객체의 size 속성 사용
    )

    # ... (Grounding DINO 추론 및 results 처리 후) ...

    if not isinstance(results, list) or len(results) == 0 or not isinstance(results[0], dict) or "boxes" not in results[0]:
        print("Grounding DINO 결과가 비어있거나 형식이 올바르지 않습니다.")
        annotated_frame = image_np_bgr.copy() # 주석 없는 원본 프레임
        if show_window:
            try:
                cv2.imshow("Grounded SAM2 Demo", annotated_frame)
                while True:
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27: break
                    if cv2.getWindowProperty("Grounded SAM2 Demo", cv2.WND_PROP_VISIBLE) < 1: break
                cv2.destroyAllWindows()
            except cv2.error: pass # 창이 이미 닫힌 경우 무시
        return annotated_frame # 결과 반환

    input_boxes = results[0]["boxes"].cpu().numpy()
    class_names = results[0]["labels"]
    confidences = results[0]["scores"].cpu().numpy().tolist()
    class_ids = np.arange(len(class_names))

    print(f"Detected boxes shape: {input_boxes.shape}") # 디버깅: 감지된 박스 개수 확인

    # --- 핵심 수정 부분 ---
    # 감지된 박스가 있을 때만 SAM2 predict 및 마스크 주석 처리 수행
    if input_boxes.shape[0] > 0:
        # 4. SAM2로 마스크 예측
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        if masks.ndim == 4:
            masks = masks.squeeze(1)

        # 마스크 유효성 검사 추가
        valid_masks = masks is not None and masks.size > 0 and masks.shape[0] == input_boxes.shape[0]
        if not valid_masks:
             print("Warning: SAM2 generated invalid or mismatched masks.")
             masks = None # 마스크를 사용하지 않도록 설정
        else:
             masks = masks.astype(bool) # bool 타입 변환은 유효할 때만

    else:
        print("No objects detected by Grounding DINO, skipping SAM2 prediction.")
        masks = None # 감지된 박스가 없으므로 마스크 없음

    # 5. 시각화
    img_bgr_annotated = image_np_bgr.copy()

    # Detections 객체 생성 (마스크 유무에 따라 다르게)
    if masks is not None and input_boxes.shape[0] > 0:
         detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks, # 이미 bool 타입이거나 None
            class_id=class_ids
        )
    elif input_boxes.shape[0] > 0: # 박스는 있지만 마스크는 없는 경우
        detections = sv.Detections(
            xyxy=input_boxes,
            class_id=class_ids
        )
    else: # 박스도 없는 경우 (이론상 위에서 처리되지만 안전하게)
        detections = sv.Detections.empty()


    annotated_frame = img_bgr_annotated # 일단 원본 복사본으로 시작

    # 감지된 것이 있을 때만 주석 추가
    if detections.is_empty() == False :
        labels_for_display = [f"{cls} {c:.2f}" for cls, c in zip(class_names, confidences)]

        from supervision.draw.color import ColorPalette
        color_palette = ColorPalette.from_hex(CUSTOM_COLORS)

        # Box Annotator
        box_annotator = sv.BoxAnnotator(color=color_palette)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

        # Label Annotator
        label_annotator = sv.LabelAnnotator(color=color_palette)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels_for_display
        )

        # Mask Annotator (마스크가 있을 때만)
        if masks is not None:
            mask_annotator = sv.MaskAnnotator(color=color_palette)
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)


    # 6. 결과 창에 띄우기
    if show_window:
        window_name = "Grounded SAM2 Demo"
        cv2.imshow(window_name, annotated_frame) # 최종 결과 표시

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27: break
            try:
                 if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1: break
            except cv2.error: break

        cv2.destroyAllWindows()

    return annotated_frame

# %%
import pyrealsense2 as rs
import numpy as np
import cv2
import time

# --- RealSense 카메라 제어 클래스 (이전 답변의 개선된 버전) ---
class RealSenseCapture:
    def __init__(self, depth_w=848, depth_h=480, color_w=848, color_h=480, fps=30):
        """카메라 설정 초기화"""
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.align = None
        self.is_running = False
        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None # 컬러 기준으로 정렬 및 언프로젝션 시 사용

        self.depth_w, self.depth_h = depth_w, depth_h
        self.color_w, self.color_h = color_w, color_h
        self.fps = fps

        print("스트림 설정 시도...")
        try:
            self.config.enable_stream(rs.stream.depth, self.depth_w, self.depth_h, rs.format.z16, self.fps)
            self.config.enable_stream(rs.stream.color, self.color_w, self.color_h, rs.format.bgr8, self.fps)

            print("스트림 설정 완료.")
        except RuntimeError as e:
            print(f"지원하지 않는 스트림 설정입니다: {e}")
            self.pipeline = None # 파이프라인 사용 불가 표시
            raise

    def start(self):
        """파이프라인 시작 및 정보 가져오기"""
        if not self.pipeline: return False # 초기화 실패 시 시작 불가
        if not self.is_running:
            print("파이프라인 시작 중...")
            try:
                profile = self.pipeline.start(self.config)
                self.is_running = True
                print("파이프라인 시작 완료.")

                # 깊이 스케일 가져오기
                depth_sensor = profile.get_device().first_depth_sensor()
                self.depth_scale = depth_sensor.get_depth_scale()
                print(f"Depth Scale: {self.depth_scale}")

                # 정렬 객체 생성 (컬러 스트림 기준)
                align_to = rs.stream.color
                self.align = rs.align(align_to)

                # 내부 파라미터(Intrinsics) 가져오기 (정렬 기준인 컬러 스트림 사용)
                color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                self.color_intrinsics = color_stream.get_intrinsics()
                print(f"Color Intrinsics: fx={self.color_intrinsics.fx}, fy={self.color_intrinsics.fy}, "
                      f"cx={self.color_intrinsics.ppx}, cy={self.color_intrinsics.ppy}")

                # 초기 안정화
                print("초기 안정화 대기 중...")
                for _ in range(5):
                    self.pipeline.wait_for_frames()
                print("초기 안정화 완료.")
                return True

            except RuntimeError as e:
                print(f"파이프라인 시작 실패: {e}")
                self.is_running = False
                return False
        else:
            print("파이프라인이 이미 실행 중입니다.")
            return True

    def capture_aligned_frames(self, timeout_ms=2000):
        """정렬된 컬러 및 깊이 프레임 캡처 (NumPy 배열 반환)"""

        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()

        # 필터 옵션 설정 (선택 사항, 필요에 따라 값 조정)
        spatial.set_option(rs.option.filter_magnitude, 2) # 필터 강도 (1-5)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5) # 스무딩 강도 (0.25-1)
        spatial.set_option(rs.option.filter_smooth_delta, 20) # 엣지 보존 강도 (1-50)
        # spatial.set_option(rs.option.holes_fill, 0) # 홀 채우기 (0: 없음, 1: 인접 픽셀, 2: 가장 먼 곳 ...)

        temporal.set_option(rs.option.filter_smooth_alpha, 0.4) # 과거 프레임 가중치 (0-1)
        temporal.set_option(rs.option.filter_smooth_delta, 1) # 변화량 임계값 (1-100)
        if not self.is_running:
            print("오류: 파이프라인 미실행.")
            return None, None

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            if not frames:
                print("오류: 프레임 수신 실패 (타임아웃).")
                return None, None

            aligned_frames = self.align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            depth_frame = spatial.process(depth_frame)
            depth_frame = temporal.process(depth_frame)

            if not depth_frame or not color_frame:
                print("오류: 유효한 프레임 획득 실패.")
                return None, None

            # 중요: 깊이 프레임은 원본(uint16) 그대로 반환, 스케일은 별도 관리
            depth_image_raw = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return color_image, depth_image_raw # 깊이 원본 반환

        except Exception as e:
            print(f"프레임 캡처 중 오류: {e}")
            return None, None

    def get_intrinsics(self):
        """컬러 카메라 내부 파라미터 반환"""
        return self.color_intrinsics

    def get_depth_scale(self):
        """깊이 스케일 반환"""
        return self.depth_scale

    def stop(self):
        """파이프라인 중지"""
        if self.is_running:
            print("파이프라인 중지...")
            self.pipeline.stop()
            self.is_running = False
            print("파이프라인 중지 완료.")

    def __enter__(self):
        if not self.start():
             raise RuntimeError("RealSense 카메라 시작 실패")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

# %%
capture= RealSenseCapture()
capture.start()

# %% [markdown]
# 여러 이미지/프롬프트 테스트

# %%
# 여기에 마음대로 호출해서 실험할 수 있습니다!
# 예: truck 이미지로 시연
# demo_image_path = "notebooks/images/cars.jpg"
color_image, depth_image_raw = capture.capture_aligned_frames()

print(color_image)



# demo_prompt = "car. person. phone. tire. white cat. grey cat. other cat. brown cat. brown dog. white dog. brown. black. white"
# demo_prompt = "blue box. calculator. black box. white circle. white box"
demo_prompt = "white box."

run_inference(
    color_image,
    text_prompt=demo_prompt,
    box_threshold=0.4,
    text_threshold=0.4,
    show_window=True   # 창에 표시
)

# result_img2 = run_inference(
#     img_path="notebooks/images/cnd.jpg",
#     text_prompt=demo_prompt,
#     box_threshold=0.3,
#     text_threshold=0.2,
#     show_window=True   # 창에 표시
# )



