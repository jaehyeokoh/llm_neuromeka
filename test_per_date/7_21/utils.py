import re
import ast
import inspect
import json
import io
import openai
import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment
from uuid import uuid4
from langchain_core.messages import AIMessage
from typing import Union, List, Dict, Any, Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
########################### 초기 설정값 ############################




########################## Tool_message 관련 코드 ##########################
def find_object_3d_properties_message(quoted_name):
    detect_msg = AIMessage(
    content="",
    tool_calls=[{
        "name": "find_object_3d_properties",
        "args": {"object_names": quoted_name},
        "id": str(uuid4()),
        "type": "tool_call"
        }]
    )
    return detect_msg

def generate_point_cloud_message(mask_base64,mask_shape, obj_height ):
    msg = AIMessage(
    content="",
    tool_calls=[{
        "name": "generate_completed_point_cloud",
        "args": {"mask_base64_list":mask_base64, "mask_shape_list":mask_shape,"obj_height_list":obj_height},
        "id": str(uuid4()),
        "type": "tool_call"
        }]
    )
    return msg

def view_properties_message(quoted_name):
    """사진속 물체 탐지 결과 확인 코드"""
    detect_msg = AIMessage(
    content="",
    tool_calls=[{
        "name": "debug_detect_visualization",
        "args": {"object_names": quoted_name},
        "id": str(uuid4()),
        "type": "tool_call"
        }]
    )
    return detect_msg

def capture_image_as_jpg_message(index):
    detect_msg = AIMessage(content="", tool_calls=[{
    "name": "capture_image_as_jpg",
    "args": {"camera_index": index},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return detect_msg

def moveL_message(pos):
    msg = AIMessage(content="", tool_calls=[{
    "name": "robot_move",
    "args": {"x": pos[0], "y": pos[1], "z": pos[2], "w": pos[3]},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def moveL_rpy_message(pos):
    msg = AIMessage(content="", tool_calls=[{
    "name": "robot_move",
    "args": {"x": pos[0], "y": pos[1], "z": pos[2], "r": pos[3], "p":pos[4], "w": pos[5], "bypass_singular": True},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def gripper_message(action):
    if action == "Grab":
        name = "robot_grab"
    elif action == "Release":
        name = "robot_release"
    msg = AIMessage(content="", tool_calls=[{
    "name": name,
    "args": {},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg


def gripper_rotate_message():
    msg = AIMessage(content="", tool_calls=[{
    "name": "robot_grip_rotate",
    "args": {},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def check_IK_message(pos):
    msg = AIMessage(content="", tool_calls=[{
    "name": "check_IK",
    "args": {"pos": pos},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def robot_get_tcp_pose_message():
    msg = AIMessage(content="", tool_calls=[{
    "name": "robot_get_tcp_pose",
    "args": {},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def robot_move_home_message():
    msg = AIMessage(content="", tool_calls=[{
    "name": "move_home",
    "args": {},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def anygrasp_message(data):
    msg = AIMessage(
    content="",
    tool_calls=[{
        "name": "anygrasp_analyze",
        "args": {"data": data},
        "id": str(uuid4()),
        "type": "tool_call"
        }]
    )
    return msg

########################## MCP 설정 관련 코드 ##########################
def load_mcp_config():
    """./mcp_config.json 파일에서 MCP 설정을 로드합니다."""
    try:
        with open(f"./mcp_config.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"MCP 설정 파일 로드 중 오류 발생: {str(e)}")
        return None

def create_server_config():
    """로드된 MCP 설정에서 서버 구성 딕셔너리를 생성합니다."""
    config = load_mcp_config()
    server_config = {}
    if config and "mcpServers" in config:
        for server_name, server_config_data in config["mcpServers"].items():
            if "command" in server_config_data:
                server_config[server_name] = {
                    "command": server_config_data.get("command"),
                    "args": server_config_data.get("args", []), 
                    "transport": "stdio",
                }
            elif "url" in server_config_data:
                server_config[server_name] = {
                    "url": server_config_data.get("url"),
                    "transport": "sse",
                }
    return server_config


########################## JSON 추출 및 파싱 관련 함수 ##########################
def extract_json(raw: str):
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    # 리스트 혹은 객체 시작 위치 탐색
    start = raw.find("{")
    start_list = raw.find("[")
    if (start_list != -1 and (start_list < start or start == -1)):
        start = start_list

    # 끝 위치 탐색
    end_obj = raw.rfind("}")
    end_list = raw.rfind("]")
    end = max(end_obj, end_list)

    if start == -1 or end == -1 or start > end:
        raise ValueError("JSON 블록을 찾을 수 없습니다.")

    json_str = raw[start:end+1]
    return json.loads(json_str)

def convert_list_to_dict(obj_list): 
    if isinstance(obj_list, dict):
        return obj_list  # 이미 dict이면 그대로 반환
    elif isinstance(obj_list, list):
        # ▶ ID를 키로 사용하도록 변경
        return { item['id']: item for item in obj_list }
    else:
        raise TypeError(f"Expected list or dict, got {type(obj_list)}: {obj_list}")


################# node 만들때 오류나서 만든 함수 ##################
def wrap_async(fn, **fixed_kwargs):
    """
    어떤 함수든 (llm, state, config) 이든 (state, config, tool_node) 이든
    알아서 state, config를 넣고 나머지는 fixed_kwargs로 넘긴다.
    """
    sig = inspect.signature(fn)

    async def wrapper(state, config):
        # 함수 인자 이름 가져오기
        param_names = list(sig.parameters.keys())

        # 준비할 인자 리스트
        args = []
        kwargs = {}

        for name in param_names:
            if name == "state":
                args.append(state)
            elif name == "config":
                kwargs["config"] = config
            elif name in fixed_kwargs:
                args.append(fixed_kwargs[name])
            else:
                # 없는 인자는 기본값을 쓰게 냅둬
                pass

        return await fn(*args, **kwargs)

    return wrapper


########################## GPT 출력 (action_sypervisor) 포맷팅 관련 함수 ##########################

def remove_duplicates_by_id(objects: List[Dict]) -> List[Dict]:
    """
    id같으면 하나 없애는 코드
    """
    try:
        seen = set()
        unique_objects = []
        for obj in objects:
            obj_id = obj.get('id')
            if obj_id not in seen:
                seen.add(obj_id)
                unique_objects.append(obj)
        return unique_objects
    except:
        return []
# # 음성 파일을 텍스트로 변환하는 함수
# def record_audio(duration=5, fs=16000):
#     print(f"음성 입력을 {duration}초 동안 기다립니다...")
#     audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
#     return audio_data.flatten()

# # 음성 데이터를 텍스트로 변환하는 함수 (업데이트된 OpenAI API 사용)
# def transcribe_audio(audio_data):
#     try:
#         # 오디오 데이터를 BytesIO 객체로 변환
#         audio_file = io.BytesIO()
#         write(audio_file, 16000, audio_data)
#         audio_file.seek(0)

#         # Whisper 모델을 사용하여 텍스트로 변환
#         transcript = client.audio.transcriptions.create(
#         model='whisper-1', 
#         file=audio_file, 
#         response_format='text' 
#         )

#         return response['text']
#     except Exception as e:
#         print(f"음성 인식 오류: {e}")
#         return "음성 인식에 실패했습니다."


############################################ 객체 위치 피드백 관련 함수 ######################################3
import json
import re
from math import sqrt, inf

def ensure_list_of_dicts(
    raw: Union[str, Dict[str, Any], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    raw가 JSON 문자열이든, dict이든, list[dict]이든,
    항상 List[Dict] 형태로 반환합니다.
    """
    # 1) 문자열이면 JSON 파싱
    if isinstance(raw, str):
        parsed = json.loads(raw)
    else:
        parsed = raw

    # 2) dict → [dict], list[dict]은 그대로, 그 외는 에러
    if isinstance(parsed, dict):
        return [parsed]
    elif isinstance(parsed, list):
        # (optional) 각 아이템이 dict인지도 검사할 수 있음
        return parsed
    else:
        raise ValueError(f"expecting dict or list[dict], got {type(parsed)}")
    
def parse_to_obj(raw):
    """
    raw가 dict/list면 그대로 반환,
    str이면 extract_json(raw)로 JSON 블록을 뽑아 파싱 후 반환.
    """
    if not isinstance(raw, str):
        return raw
    return extract_json(raw)

def match_or_create_object(prev_list, new_det, tool_pos,distance_threshold=0.04):
    """
    distance_threshold보다 가까운 같은 이름의 객체는 같은 객체로 판단하고 무시하는 코드
    prev_list: [{"id", "name", "position":{x,y,z}, ...}, ...]
    new_det: dict or JSON-string of {"object_name", "position":[x,y,z], ...}
    """
    # 0) parse
    new_det = parse_to_obj(new_det)

    if isinstance(new_det, str):
        print("[MATCH] new_det is str → attempting json.loads")
        try:
            new_det = json.loads(new_det.strip())
        except Exception as e:
            print(f"[MATCH][ERROR] Failed to parse new_det JSON: {e}")
            raise

    # 1) validation
    if not isinstance(new_det, dict):
        raise TypeError(f"[MATCH][ERROR] new_det must be dict after parsing, got {type(new_det)}")
    required_keys = ("object_name", "position", "dimensions", "area", "rotation_degree")
    for key in required_keys:
        if key not in new_det:
            raise KeyError(f"[MATCH][ERROR] Missing key '{key}' in new_det")

    name = new_det["object_name"]
    pos = new_det["position"]
    dims = new_det["dimensions"]

    # 2) unpack
    x_new, y_new, z_new = pos
    # 3) dimensions checked above

    # 4) filter candidates
    candidates = [o for o in prev_list if o.get("name") == name]

    # distance function
    def dist(o):
        p = o.get("position")
        if isinstance(p, dict):
            x0, y0, z0 = p["x"], p["y"], p["z"]
        else:
            x0, y0, z0 = p
        d = sqrt((x0 - x_new)**2 + (y0 - y_new)**2 + (z0 - z_new)**2)
        return d

    # 5) matching
    if candidates:
        best = min(candidates, key=dist)
        best_dist = dist(best)
        if best_dist <= distance_threshold:
            return None

    # 6) new ID
    prefix = name.replace(" ", "_")
    suffixes = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for o in prev_list:
        m = pattern.match(o.get("id", ""))
        if m:
            suffixes.append(int(m.group(1)))
    next_idx = max(suffixes, default=-1) + 1
    new_id = f"{prefix}_{next_idx}"
    print(f"[MATCH] creating new object id={new_id}")

    x_new,y_new, w = tool_to_world(tool_pos, round(x_new, 3),round(y_new, 3),new_det["rotation_degree"]) # 월드 좌표계로 전환
    # 7) create
    new_obj = {
        "id": new_id,
        "name": name,
        # "position": {"x": round(x_new, 3), "y": round(y_new, 3), "z": round(z_new, 3)},
        "position": {"x": round(x_new, 3), "y": round(y_new, 3), "z": round(z_new, 3)},
        "dimensions": {"L": dims[0], "W": dims[1]},
        "area": new_det["area"],
        "rotation_degree": w
    }

    prev_list.append(new_obj)
    return new_obj


def find_closest_detection(raw: str, old_pos: dict):
    """
    이전에 감지된 객체 위치에서 가장 가까운 객체를 반환 -> 다른 물건 옮길때 건드려서 위치 이동했을 수도 있으니까
    Debug-enabled version:
    raw: '["{...}", "{...}", "{...}"]' 형태의 문자열
    old_pos: {'x':float, 'y':float, 'z':float}
    """
    # 1) raw가 JSON 문자열이라면 우선 파싱
    parsed = json.loads(raw)  # parsed: List[str]

    # 2) parsed가 리스트면, 내부 문자열 하나하나를 dict로 파싱
    if isinstance(parsed, list):
        dets = []
        for idx, s in enumerate(parsed):
            try:
                d = json.loads(s)  # "{…}" → {…}
                dets.append(d)
            except json.JSONDecodeError as e:
                print(f"[WARN] Failed to parse element {idx}: {e}")
        # 이제 dets는 List[Dict] 형태
    elif isinstance(parsed, dict):
        dets = [parsed]
    else:
        # safety fallback
        return None

    if not dets:
        print("[ERROR] No valid detection dicts parsed.")
        return None

    # 3) 거리 계산
    x0, y0, z0 = old_pos['x'], old_pos['y'], old_pos['z']
    best, best_dist = None, float('inf')
    for d in dets:
        pos = d.get("position")
        if not (isinstance(pos, (list,tuple)) and len(pos)==3):
            continue
        dist = sum((p - q)**2 for p, q in zip(pos, (x0, y0, z0)))
        if dist < best_dist:
            best_dist, best = dist, d

    if best is None:
        return None

    return {
        "object_name": best["object_name"],
        "position": [round(c,3) for c in best["position"]],
        "dimensions": best["dimensions"],
        "area": best["area"],
        "rotation_degree": best["rotation_degree"],
    }


##########################################################
# 파지 경로 찾는 코드 3 (Grasp Approach Direction Finder)
# 목표: 장애물 사이에서 로봇 그리퍼가 캔을 잡기 위한 가장 안전한 방향을 찾는다.
# 입력: target_object (dict), obstacles (list of dicts)
# 출력: 방향 벡터 (dx, dy) 및 상태 메시지
##########################################################

import json
import math

# --- [1] 객체 입력 포맷을 내부 처리 포맷으로 정규화 ---
def normalize_object(obj, threshold = 0.02):
    if obj["position"][2] <= threshold:
        return
    if obj["area"] >= 0.042:
        return
    return {
        "id": obj.get("object_name", "unknown"),                 # 객체 고유 이름
        "name": obj.get("object_name", "unknown"),               # 표시용 이름
        "pos": tuple(obj["position"][:2]),                       # (x, y) 좌표만 사용 (z 제외)
        "dim": tuple(obj["dimensions"]),                         # (width, height)
        "angle_cw": obj["rotation_degree"]                       # 회전 각도 (시계 방향)
    }

# --- [2] 최적 파지 방향 계산 ---
# def compute_best_approach_direction(target_object, obstacles, offset=0.001):

#     # [2-1] 회전된 사각형의 네 모서리 좌표를 계산
#     def rect_corners(obj):
#         cx, cy = obj['pos']
#         w, h = obj['dim']
#         theta = math.radians(obj['angle_cw'])  # 시계→수학 좌표계로 변환
#         corners = []
#         for dxs in (-1, 1):
#             for dys in (-1, 1):
#                 dx, dy = dxs * w/2, dys * h/2
#                 x = cx + dx * math.cos(theta) - dy * math.sin(theta)
#                 y = cy + dx * math.sin(theta) + dy * math.cos(theta)
#                 corners.append((x, y))
#         return corners

#     # [2-2] 레이와 선분의 교차점 여부 계산
#     def intersect_ray_seg(origin, direction, seg_start, seg_end):
#         (ox, oy), (dx, dy) = origin, direction
#         x1, y1 = seg_start; x2, y2 = seg_end
#         vx, vy = x2 - x1, y2 - y1
#         det = dx * vy - dy * vx
#         if abs(det) < 1e-6:
#             return None  # 평행 → 교차 없음
#         t = ((x1 - ox) * vy - (y1 - oy) * vx) / det  # ray 방향으로의 거리
#         u = ((x1 - ox) * dy - (y1 - oy) * dx) / det  # 선분 내 위치 (0 ≤ u ≤ 1)
#         if 0 <= t <= 0.2 and 0 <= u <= 1: # 감지 선분의 길이를 20cm로 제한
#             return t
#         return None

#     # [2-3] 캔 중심 좌표
#     tx, ty = target_object['pos']

#     # [2-4] 각도 0~359도에서 장애물 충돌 여부 검사
#     free = [False] * 360
#     for deg in range(360):
#         rad = math.radians(deg)
#         dir_vec = (math.cos(rad), math.sin(rad))
#         origin = (tx + offset * dir_vec[0], ty + offset * dir_vec[1])  # 캔에서 offset만큼 바깥에서 시작

#         hit = False
#         for obj in obstacles:
#             corners = rect_corners(obj)
#             for i in range(4):
#                 s, e = corners[i], corners[(i+1)%4]
#                 if intersect_ray_seg(origin, dir_vec, s, e):  # ray가 사각형과 교차?
#                     hit = True
#                     break
#             if hit:
#                 break
#         free[deg] = not hit  # 충돌 없으면 FREE

#     # [2-5] 연속된 FREE 구간 중 가장 긴 구간 탐색 (360도 → 래핑 허용)
#     free2 = free + free  # 래핑 처리를 위한 2배 배열
#     best_len = 0
#     best_start = None
#     cur_len = 0
#     cur_start = None
#     for i, val in enumerate(free2):
#         if val:
#             if cur_len == 0:
#                 cur_start = i
#             cur_len += 1
#         else:
#             if cur_len > best_len:
#                 best_len = cur_len
#                 best_start = cur_start
#             cur_len = 0
#     if cur_len > best_len:  # 마지막까지 FREE로 끝난 경우
#         best_len = cur_len
#         best_start = cur_start

#     # [2-6] FREE 공간이 전혀 없을 경우 예외 처리
#     if best_start is None:
#         return None, None, "[ERROR] No free direction found"
    
#     ######################################################################## 7_10 추가#############33
#     # 실제 최대 각도로 클램프
#     actual_len = min(best_len, 360)

#     # 최소 허용 각도(여기선 5°) 검사
#     if actual_len < 5:
#         return None, None, "[ERROR] No direction with at least 5° free space"
#     ###################################################################################################

#     # ────────────────────────────────────────────────
#     # [NEW] 180° 근방 스캔: 정확 180°가 아니어도 ±snap_range 안에 Free 각도가 있으면 바로 반환
#     snap_range = 20  
#     for d in range(0, snap_range+1):
#         for candidate in (270 - d, 270 + d): # -> 270도면 0,1방향 벡터 나옴
#             deg = candidate % 360
#             if free[deg]:
#                 # candidate가 free라면 유효 segment에 속하는지(optional) 확인 후 즉시 반환
#                 rad = math.radians(deg)
#                 dx, dy = -math.cos(rad), -math.sin(rad)
#                 return dx, dy, f"[OK] Snapped near 270° = {deg}° (±{snap_range}°)"
#     # ────────────────────────────────────────────────


#     # [2-7] 최적 방향의 중앙 각도 계산
#     best_start = best_start % 360
#     best_mid = (best_start + best_len // 2) % 360
#     best_rad = math.radians(best_mid)

#     # [2-8] 접근 방향 벡터 (빈 공간 → 캔 중심 방향)
#     dx = -math.cos(best_rad)
#     dy = -math.sin(best_rad)

#     return dx, dy, f"[OK] Best direction = {best_mid}°, vector = ({dx:.4f}, {dy:.4f})"

def compute_best_approach_direction(target_object, obstacles, offset=0.01):
    # rect_corners: 직사각형 객체의 모서리 좌표 리스트 생성 함수
    def rect_corners(obj):
        cx, cy = obj['pos']
        w, h = obj['dim']
        theta = math.radians(obj['angle_cw'])  # 회전 각도를 라디안으로 변환
        corners = []
        # 네 개 코너 계산: dxs, dys 값 조합
        for dxs in (-1, 1):
            for dys in (-1, 1):
                dx, dy = dxs * w/2, dys * h/2
                # 회전 변환 적용: (cx + dx*cos - dy*sin, cy + dx*sin + dy*cos)
                x = cx + dx * math.cos(theta) - dy * math.sin(theta)
                y = cy + dx * math.sin(theta) + dy * math.cos(theta)
                corners.append((x, y))
        return corners  # [(x1,y1), ..., (x4,y4)]

    # intersect_ray_seg: 광선(origin, direction)과 선분(seg_start->seg_end)의 교차 여부 및 거리 t 반환
    def intersect_ray_seg(origin, direction, seg_start, seg_end):
        (ox, oy), (dx, dy) = origin, direction
        x1, y1 = seg_start; x2, y2 = seg_end
        vx, vy = x2 - x1, y2 - y1  # 선분 벡터
        # 행렬식 계산: det = dx*vy - dy*vx
        det = dx * vy - dy * vx
        if abs(det) < 1e-6:
            return None  # 평행
        # t: 레이 파라미터, u: 선분 파라미터
        t = ((x1 - ox) * vy - (y1 - oy) * vx) / det
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / det
        # t >= 0.01 (origin 근처 false intersection 제거), 0<=u<=1 범위 내이면 교차
        if 1e-2 <=t <= 0.25 and 0 <= u <= 1:
            return t
        return None

    # 목표 객체 위치
    tx, ty = target_object['pos']
    # free[d]: 방향(degree) d에 대해 장애물과의 충돌 여부 False=충돌 없음
    free = [False] * 360
    # ray 데이터 저장용 (시각화를 위해)
    ray_data = []
    
    # 0~359도로 스캔
    for deg in range(360):
        rad = math.radians(deg)
        dir_vec = (math.cos(rad), math.sin(rad))  # 단위 방향 벡터
        # 광선 출발점: 목표 객체 표면에서 약간 떨어진 위치
        origin = (tx + offset * dir_vec[0], ty + offset * dir_vec[1])
        hit = False
        closest_t = float('inf')
        hit_point = None
        
        # 모든 장애물에 대해 intersection 검사
        for obj in obstacles:
            corners = rect_corners(obj)
            # 각 모서리 선분 간 교차 여부 확인
            for i in range(4):
                s, e = corners[i], corners[(i+1)%4]
                t = intersect_ray_seg(origin, dir_vec, s, e)
                if t is not None and t < closest_t:
                    closest_t = t
                    hit = True
                    hit_point = (origin[0] + t * dir_vec[0], origin[1] + t * dir_vec[1])
        
        free[deg] = not hit  # 충돌 없으면 True
        
        # ray 데이터 저장 (시각화용)
        if hit:
            ray_data.append({
                'degree': deg,
                'origin': origin,
                'direction': dir_vec,
                'hit': True,
                'hit_point': hit_point,
                'distance': closest_t
            })
        else:
            # 충돌이 없으면 임의의 긴 거리로 설정
            ray_length = 1.0
            end_point = (origin[0] + ray_length * dir_vec[0], origin[1] + ray_length * dir_vec[1])
            ray_data.append({
                'degree': deg,
                'origin': origin,
                'direction': dir_vec,
                'hit': False,
                'end_point': end_point,
                'distance': ray_length
            })

    # free 배열을 두 번 이어서 원형 구간 탐색 용이하게 만듦
    free2 = free + free
    best_len = 0  # 최적 연속 True 구간 길이
    best_start = None
    cur_len = 0
    cur_start = None
    # 연속된 True 구간 찾기: 최대 길이, 시작 인덱스 기록
    for i, val in enumerate(free2):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_len = 0
    # 마지막 구간 처리
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start

    # 교차 구간이 없으면 None 반환
    if best_start is None:
        return None, None, "[ERROR] No free direction found"

    # 실제 0~359 범위로 매핑
    best_start = best_start % 360
    # 중간 방향 계산
    best_mid = (best_start + best_len // 2) % 360
    best_rad = math.radians(best_mid)

    # 반환 벡터: 시선이 아니라 접근 방향(화살표 반전)
    dx = -math.cos(best_rad)
    dy = -math.sin(best_rad)

    return dx, dy, f"[OK] Best direction = {best_mid}°, vector = ({dx:.4f}, {dy:.4f})"

# 150° 이상인 모든 구간을 찾는 헬퍼 ######################################################### 7_9 추가함
import math

def find_wide_open_spans(target_object, obstacles, offset=0.001, threshold=150):
    """
    타겟 주변에 돌아가면서 선분을 쏴서 threshold(기본 150도) 이상으로 빈 공간이 남는 모든 구간을 찾아 반환합니다.
    반환 리스트 항목: {'start_deg', 'span_deg', 'mid_deg', 'vector'}
    """
    # 사각형의 네 모서리 좌표 계산
    def rect_corners(obj):
        cx, cy = obj['pos']
        w, h = obj['dim']
        theta = math.radians(obj['angle_cw'])
        corners = []
        for dxs in (-1, 1):
            for dys in (-1, 1):
                dx, dy = dxs * w/2, dys * h/2
                x = cx + dx * math.cos(theta) - dy * math.sin(theta)
                y = cy + dx * math.sin(theta) + dy * math.cos(theta)
                corners.append((x, y))
        return corners

    # 레이와 선분 교차 테스트
    def intersect_ray_seg(origin, direction, s, e):
        (ox, oy), (dx, dy) = origin, direction
        x1, y1 = s; x2, y2 = e
        vx, vy = x2 - x1, y2 - y1
        det = dx * vy - dy * vx
        if abs(det) < 1e-6:
            return False
        t = ((x1 - ox) * vy - (y1 - oy) * vx) / det
        u = ((x1 - ox) * dy - (y1 - oy) * dx) / det
        return (t >= 0 and 0 <= u <= 1)

    # 중심 좌표
    tx, ty = target_object['pos']

    # 각도별 충돌 검사
    free = []
    for deg in range(360):
        rad = math.radians(deg)
        dir_vec = (math.cos(rad), math.sin(rad))
        origin = (tx + offset * dir_vec[0], ty + offset * dir_vec[1])

        hit = False
        for obj in obstacles:
            corners = rect_corners(obj)
            for i in range(4):
                if intersect_ray_seg(origin, dir_vec, corners[i], corners[(i+1) % 4]):
                    hit = True
                    break
            if hit:
                break
        free.append(not hit)

    # 래핑 처리 및 threshold 이상 구간 수집
    free2 = free + free
    wide_dirs = []
    cur_start = None
    cur_len = 0

    for i, is_free in enumerate(free2):
        if is_free:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len >= threshold:
                start = cur_start % 360
                length = cur_len
                mid = (start + length // 2) % 360
                rad_mid = math.radians(mid)
                vec = (-math.cos(rad_mid), -math.sin(rad_mid))
                wide_dirs.append({
                    'start_deg': start,
                    'span_deg': length,
                    'mid_deg': mid,
                    'vector': vec
                })
            cur_len = 0

    # 마지막 구간 체크
    if cur_len >= threshold:
        start = cur_start % 360
        length = cur_len
        mid = (start + length // 2) % 360
        rad_mid = math.radians(mid)
        vec = (-math.cos(rad_mid), -math.sin(rad_mid))
        wide_dirs.append({
            'start_deg': start,
            'span_deg': length,
            'mid_deg': mid,
            'vector': vec
        })

    # 중복 제거 (start_deg 기준)
    unique = {}
    for d in wide_dirs:
        key = (d['start_deg'], d['span_deg'])
        unique[key] = d
    return list(unique.values())





def compute_waypoints(dx, dy, target, center_radius, base_offset=0.05):
    """
    그리퍼 접근을 위한 시작점과 목표점 그리고 yaw(°) 계산.

    매개변수:
        dx (float): 정규화된 방향 벡터의 X 성분.
        dy (float): 정규화된 방향 벡터의 Y 성분.
        target (tuple): 목표 객체의 중심 좌표 (x, y, z).
        center_radius (float): 객체 중심으로부터 그리퍼가 접촉할 지점까지의 반경(예: 객체 반지름).
        base_offset (float): 객체 표면에서 추가로 떨어질 그립하기 시작할 거리 (기본값 0.05 m).

    반환값:
        dict: {
            'start_point': (x, y, z),    # 접근 시작점 좌표
            'target_point': (x, y, z),   # 객체 지름 고려 된 목표 객체 좌표
            'yaw_deg': yaw 각도 (도 단위)
        }
    """
    x_t, y_t, z_t = target

    if z_t >= 0.12: # 높이가 크면 
        z_t = z_t / 2 + 0.01 # 중간에서 위쪽 잡기
    elif z_t >= 0.1:
        z_t -=0.1
    
    # 총 오프셋: 객체 반경 + 추가 베이스 오프셋
    total_offset = center_radius + base_offset

    # yaw 계산 (도 단위)
    yaw_deg = math.degrees(math.atan2(dy, dx))

    # 접근 시작점: 중심에서 방향벡터 반대 방향으로 total_offset 만큼 떨어진 위치
    start_point = [
        round(x_t - total_offset * dx, 4),
        round(y_t - total_offset * dy, 4),
        round(z_t, 4)
    ]

    # 접촉점: 객체 표면상의 실제 접촉점
    contact_point = [
        round(x_t - center_radius * dx, 4),
        round(y_t - center_radius * dy, 4),
        round(z_t, 4)
    ]

    # # 목표점: 객체 중심
    # target_point = (
    #     round(x_t, 4),
    #     round(y_t, 4),
    #     round(z_t, 4)
    # )

    return {
        'start_point': start_point,
        'target_point': contact_point,
        'yaw_deg': round(yaw_deg, 2)
    }

###################################################################### 잡을 때 rpy 구하는 코드 ############################


def direction_from_angles(azimuth_deg, elevation_deg):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    return np.array([np.cos(el)*np.cos(az),
                     np.cos(el)*np.sin(az),
                     np.sin(el)])

def compute_roll_to_maintain_tool_y_up(D):
    # minimal rot aligning +Z to D
    z = np.array([0,0,1], float)
    target = D/np.linalg.norm(D)
    axis = np.cross(z, target)
    if np.linalg.norm(axis)<1e-8:
        rot1 = R.from_quat([0,0,0,1])
    else:
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z, target), -1,1))
        rot1 = R.from_rotvec(axis*angle)
    # after that rotation, tool Y in world:
    new_y = rot1.apply([0,1,0])
    # project both new_y and world up onto plane normal to D
    proj = lambda v: v - np.dot(v, target)*target
    p_new_y = proj(new_y); p_new_y /= np.linalg.norm(p_new_y)
    p_up    = proj([0,0,1]);   p_up    /= np.linalg.norm(p_up)
    # signed angle between them around axis=D
    dot = np.clip(np.dot(p_new_y, p_up), -1,1)
    ang = np.arccos(dot)
    sign = np.sign(np.dot(np.cross(p_new_y, p_up), target))
    return sign*ang

def get_alignment_and_compensation(dx,dy, elevation_deg=-10.0):
    """
    입력:
      - current_quat: [x, y, z, w] 현재 툴의 방향(quaternion)
      - dx, dy 원하는 XY 평면 방향 벡터
      - elevation_deg: 수평면 대비 위쪽(+) 또는 아래쪽(-)으로 기울일 각도(°)
    반환:
      - rpy_deg: 최종 자세의 롤, 피치, 요 각도(°) 튜플
      - feedback_deg: 마지막 관절(joint6) 보상 각도(°)
    """
    # 1) build tilted direction
    az = np.rad2deg(np.arctan2(dy, dx))
    D_tilt = direction_from_angles(az, elevation_deg)

    # 2) minimal rotation: +Z -> D_tilt
    z = np.array([0,0,1], float)
    axis = np.cross(z, D_tilt)
    if np.linalg.norm(axis)<1e-8:
        rot1 = R.from_quat([0,0,0,1])
    else:
        axis /= np.linalg.norm(axis)
        angle = np.arccos(np.clip(np.dot(z, D_tilt)/np.linalg.norm(D_tilt), -1,1))
        rot1 = R.from_rotvec(axis*angle)
    
    # 3) compensation yaw around D_tilt
    comp_rad = compute_roll_to_maintain_tool_y_up(D_tilt)
    rot2 = R.from_rotvec(comp_rad * (D_tilt/np.linalg.norm(D_tilt)))

    # 4) combine rotations and extract Euler
    rot_final = rot2 * rot1
    rpy = rot_final.as_euler('xyz', degrees=True)
    feedback_deg = np.rad2deg(comp_rad)
    return rpy, feedback_deg



#################################### 카메라 관련 코드 ##################################################

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
    # psi_t = toolpos[5]
    psi_t = 0
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

def update_world_coordinates(toolpos, objects):
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
        x_w, y_w, w = tool_to_world(toolpos, x_obj, y_obj, psi_obj)
        print(x_w,y_w,w)
        # 원본 딕셔너리에 치환 혹은 추가
        obj["position"][0] = x_w
        obj["position"][1] = y_w
        obj["rotation_degree"] = w
        updated.append(obj)
    return updated