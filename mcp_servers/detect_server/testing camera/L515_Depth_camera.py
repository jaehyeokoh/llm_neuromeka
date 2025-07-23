import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import os
GROUND_SAM2_PATH = os.path.join(os.path.dirname(__file__), "Grounded-SAM-2")
sys.path.append(GROUND_SAM2_PATH)
# --- 설정 ---
DEPTH_WIDTH = 848  # L515 기본 깊이 해상도
DEPTH_HEIGHT = 480
COLOR_WIDTH = 848   # L515 기본 컬러 해상도 (Full HD)
COLOR_HEIGHT = 480
FPS = 30

# --- RealSense 파이프라인 초기화 ---
pipeline = rs.pipeline()
config = rs.config()

# 중앙 영역 설정
REGION_SIZE = 40 # 평균 계산할 영역의 한 변 크기 (픽셀 단위, 정사각형 가정
# 특정 장치 시리얼 번호로 연결 (선택 사항, 여러 대 연결 시 유용)
# serial_number = "여기에_L515_시리얼번호_입력"
# config.enable_device(serial_number)

# 스트림 설정: 깊이(Z16 format), 컬러(BGR8 format)
# L515는 특정 해상도/포맷 조합만 지원할 수 있습니다. 필요시 SDK 문서 참조.
try:
    config.enable_stream(rs.stream.depth, DEPTH_WIDTH, DEPTH_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT, rs.format.bgr8, FPS) # OpenCV 호환성을 위해 BGR8 사용
except RuntimeError as e:
    print(f"지원하지 않는 스트림 설정입니다: {e}")
    print("카메라가 연결되었는지, 다른 프로그램에서 사용 중이지 않은지 확인하세요.")
    exit()

# --- 스트리밍 시작 ---
print("파이프라인 시작 중...")
try:
    profile = pipeline.start(config)
    print("파이프라인 시작 완료.")

    # 깊이 센서의 스케일 가져오기 (미터 단위 변환에 사용)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth Scale is: {depth_scale:.4f} (1 unit = {depth_scale*1000:.2f} mm)")

    # 컬러 스트림을 기준으로 깊이 프레임을 정렬하기 위한 객체 생성
    align_to = rs.stream.color
    align = rs.align(align_to)

    # 깊이 데이터를 시각화하기 위한 컬러맵 객체 생성
    colorizer = rs.colorizer()

  # --- 후처리 필터 생성 ---
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()

    # 필터 옵션 설정 (선택 사항, 필요에 따라 값 조정)
    spatial.set_option(rs.option.filter_magnitude, 2) # 필터 강도 (1-5)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5) # 스무딩 강도 (0.25-1)
    spatial.set_option(rs.option.filter_smooth_delta, 20) # 엣지 보존 강도 (1-50)
    # spatial.set_option(rs.option.holes_fill, 0) # 홀 채우기 (0: 없음, 1: 인접 픽셀, 2: 가장 먼 곳 ...)

    temporal.set_option(rs.option.filter_smooth_alpha, 0.4) # 과거 프레임 가중치 (0-1)
    temporal.set_option(rs.option.filter_smooth_delta, 1) # 변화량 임계값 (1-100)
  

    # --- 메인 루프: 프레임 가져오기 및 표시 ---
    while True:
        # 프레임 세트 대기 (깊이 + 컬러)
        frames = pipeline.wait_for_frames()

        # 깊이 프레임을 컬러 프레임 좌표계에 정렬
        aligned_frames = align.process(frames)

        # 정렬된 프레임 가져오기
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # 프레임이 유효하지 않으면 건너뛰기
        if not depth_frame or not color_frame:
            print("프레임 누락...")
            continue

        # --- 데이터 변환 ---
        # 깊이 및 컬러 이미지를 NumPy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data()) # BGR 포맷

        # --- 시각화 ---
        # 깊이 이미지에 컬러맵 적용 (시각적 확인용)


        # 정렬된 깊이 프레임에서 값을 가져와야 컬러 프레임 중앙과 일치

        # 중요: 필터는 rs.frame 객체에 적용
        filtered_depth_frame = depth_frame
        filtered_depth_frame = spatial.process(filtered_depth_frame)
        filtered_depth_frame = temporal.process(filtered_depth_frame)

        depth_colormap = np.asanyarray(colorizer.colorize(filtered_depth_frame).get_data())
        depth_image_raw = np.asanyarray(filtered_depth_frame.get_data())
       # --- 깊이 평균 계산 ---
        # 중요: 정렬된 프레임은 컬러 해상도 기준이므로 컬러 이미지 크기 사용
        h, w = color_image.shape[:2]
        center_x, center_y = w // 3, h // 3


        # 중앙 영역 좌표 계산 (이미지 경계 고려)
        half_region = REGION_SIZE // 2
        x1 = max(0, center_x - half_region)
        y1 = max(0, center_y - half_region)
        x2 = min(w, center_x + half_region)
        y2 = min(h, center_y + half_region)

        average_depth_meters = 0.0 # 기본값 초기화
        try:
            # 깊이 이미지에서 해당 영역 추출
            depth_roi = depth_image_raw[y1:y2, x1:x2]

            # 유효한 깊이 값만 필터링 (0보다 큰 값)
            valid_depths = depth_roi[depth_roi > 0]

            if valid_depths.size > 0:
                # 유효한 깊이 값들의 평균 계산 (원본 단위)
                average_raw_depth = np.mean(valid_depths)
                # 미터 단위로 변환
                average_depth_meters = average_raw_depth * depth_scale
            else:
                # print("경고: 중앙 영역에 유효한 깊이 값이 없습니다.") # 너무 자주 출력될 수 있음
                pass

        except Exception as e:
            print(f"깊이 평균 계산 중 오류: {e}")


         # 컬러 이미지에 사각형 그리기
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2) # 녹색 사각형

        # 컬러 이미지에 평균 깊이 텍스트 표시
        text = f"Avg Depth: {average_depth_meters:.3f} m"
        # 텍스트 위치는 사각형 아래쪽으로 조정
        text_x = x1
        text_y = y2 + 20 # 사각형 아래 20픽셀
        # 텍스트가 화면 밖으로 나가지 않도록 조정 (선택 사항)
        if text_y > h - 10: text_y = y1 - 10

        cv2.putText(color_image, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
                # --- 화면 정중앙에 점 그리기 ---
        # 실제 이미지 중앙 좌표 계산
        true_center_x = w // 2
        true_center_y = h // 2

        # 점(작은 원) 그리기 설정
        dot_radius = 3  # 점의 반지름 (픽셀 단위)
        dot_color = (0, 0, 255) # 점의 색상 (BGR 형식, 여기서는 빨간색)
        dot_thickness = -1 # 원 내부를 채움 (-1)

        # 컬러 이미지에 점 그리기
        cv2.circle(color_image, (true_center_x, true_center_y), dot_radius, dot_color, dot_thickness)

        # OpenCV를 사용하여 이미지 표시
        cv2.namedWindow('RealSense L515 - Color', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense L515 - Color', color_image)

        cv2.namedWindow('RealSense L515 - Depth', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense L515 - Depth', depth_colormap)

        # --- 종료 조건 ---
        key = cv2.waitKey(1)
        # 'q' 키 또는 'ESC' 키를 누르면 종료
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

except Exception as e:
    print(f"오류 발생: {e}")

finally:
    # --- 정리 ---
    print("파이프라인 중지 중...")
    pipeline.stop()
    print("파이프라인 중지 완료.")
