import json
import numpy as np
def tool_to_world(toolpos, x_obj, y_obj, psi_obj, offset_x=0.09, offset_y=0.0):
    """
    로봇 베이스(월드) 기준에서 엔드툴 위치 및 yaw(psi_t)를 입력받아,
    카메라 오프셋(offset_x, offset_y)을 내부적으로 적용한 후,
    카메라 기준으로 본 물체 위치/방향(x_obj, y_obj, psi_obj)을
    다시 월드 기준으로 변환합니다.

    Args:
        x_t (float): 베이스 기준 엔드툴의 X 위치 (m)
        y_t (float): 베이스 기준 엔드툴의 Y 위치 (m)
        psi_t (float): 베이스 기준 엔드툴의 yaw 각도 (°)
        x_obj, y_obj (float): 카메라 기준계에서 본 물체의 X/Y 위치 (m)
        psi_obj (float): 카메라 기준계에서 본 물체의 yaw 각도 (°)
        offset_x (float): 툴 기준에서 카메라의 X 오프셋 (m), 기본값 0.09
        offset_y (float): 툴 기준에서 카메라의 Y 오프셋 (m), 기본값 0.0

    Returns:
        x_w, y_w, psi_w (tuple):
            x_w (float): 월드 기준계에서 물체의 X 위치 (m)
            y_w (float): 월드 기준계에서 물체의 Y 위치 (m)
            psi_w (float): 월드 기준계에서 물체의 yaw (°)
    """
    x_t = toolpos[0]
    y_t = toolpos[1]
    psi_t = toolpos[5]
    # 1. 툴->카메라 월드 위치 계산
    c_t, s_t = np.cos(np.deg2rad(psi_t)), np.sin(np.deg2rad(psi_t))
    # 카메라 월드 위치
    x_c = x_t + c_t * offset_x - s_t * offset_y
    y_c = y_t + s_t * offset_x + c_t * offset_y
    psi_c = psi_t  # 카메라 yaw 오프셋 없다고 가정

    # 2. 카메라->월드 물체 위치 계산
    x_w = x_c + c_t * x_obj - s_t * y_obj
    y_w = y_c + s_t * x_obj + c_t * y_obj
    psi_w = psi_c + psi_obj

    return x_w, y_w, psi_w
def update_world_coordinates(objects, toolpos, offset_x=0.09, offset_y=0.0):
    """
    주어진 객체 리스트에서 각 객체의 로컬 좌표(x, y)와 회전 각도(psi)를 뽑아
    tool_to_world 함수를 통해 월드 좌표로 변환한 후,
    원본 객체 딕셔너리에 'world_x', 'world_y', 'world_w' 키로 추가/치환합니다.

    Args:
        objects (list): JSON 문자열 또는 dict 형태의 객체 리스트
        toolpos (tuple): 툴의 월드 좌표 (x, y, z)
        offset_x (float): x축 오프셋
        offset_y (float): y축 오프셋

    Returns:
        list: 변환된 월드 좌표가 추가된 객체 리스트
    """
    updated = []
    for entry in objects:
        # JSON 문자열이면 파싱
        obj = json.loads(entry) if isinstance(entry, str) else entry.copy()

        # 로컬 좌표와 회전 추출
        x_obj, y_obj = obj["position"][0], obj["position"][1]
        psi_obj = obj.get("rotation_degree")

        # 월드 좌표 계산 (tool_to_world 함수 필요)
        x_w, y_w, w = tool_to_world(toolpos, x_obj, y_obj, psi_obj, offset_x, offset_y)
        print(x_w,y_w,w)
        # 원본 딕셔너리에 치환 혹은 추가
        obj["position"][0] = x_w
        obj["position"][1] = y_w
        obj["rotation_degree"] = w
        updated.append(obj)
    return updated

# 예시 사용법:
objs = ['{\n  "object_name": "black mat",\n  "position": [\n    0.019,\n    0.025,\n    0.011\n  ],\n  "dimensions": [\n    0.32,\n    0.247\n  ],\n  "area": 0.07904,\n  "rotation_degree": -86.89\n}', '{\n  "object_name": "silver table",\n  "position": [\n    0.007,\n    -0.188,\n    0.018\n  ],\n  "dimensions": [\n    0.321,\n    0.177\n  ],\n  "area": 0.056817,\n  "rotation_degree": -88.41\n}']
tool_position = [0.4, 0.15, 0.28, 0.0, 180.0, 0.0001]
transformed = update_world_coordinates(objs, tool_position)
print(transformed)
