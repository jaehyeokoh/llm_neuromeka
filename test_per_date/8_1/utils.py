import re
import ast
import inspect
import json
from uuid import uuid4
from langchain_core.messages import AIMessage
from typing import Union, List, Dict, Any, Optional
import numpy as np
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
    "args": {"target_pose": pos},
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

def anygrasp_message(data,target_boundary_info, debug):
    msg = AIMessage(
    content="",
    tool_calls=[{
        "name": "anygrasp_analyze",
        "args": {"data": data, "target_boundary_info":target_boundary_info, "debug": debug},
        "id": str(uuid4()),
        "type": "tool_call"
        }]
    )
    return msg

def wait_idle_message():
    msg = AIMessage(content="", tool_calls=[{
    "name": "wait_idle",
    "args": {},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def monitoring_and_recover_message():
    msg = AIMessage(content="", tool_calls=[{
    "name": "monitoring_and_recover",
    "args": {},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def check_collision_message(start, end, obstacles):
    msg = AIMessage(content="", tool_calls=[{
    "name": "check_collision",
    "args": {"start_obj":start,"end":end},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def save_obstacles2assemble_server(obstacles):
    msg = AIMessage(content="", tool_calls=[{
    "name": "save_obstacles",
    "args": {"obstacles":obstacles},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
    return msg

def get_obstacles2assemble_server():
    msg = AIMessage(content="", tool_calls=[{
    "name": "get_obstacles",
    "args": {},
    "id": str(uuid4()),
    "type": "tool_call"
    }])
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

def format_actions_for_llm(action_list: list[str]):
    """
    문자열 리스트 (action 들) 를 받아서 번호 매겨 출력용 리스트로 변환
    """
    return [f"{i + 1}. {action}" for i, action in enumerate(action_list)]

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


def parse_plan_editor_output(text: str):
    """
    LLM divided_input editor 출력 파싱 함수.
    - reasoning: Step-by-step 문자열 리스트
    - steps: [{"action": "...", "objects": ["..."]}] 형태의 리스트
    """
    result = {
        "reasoning": [],
        "steps": []
    }
    
    # 구간 분리: reasoning / answer
    parts = text.strip().split("###")
    if len(parts) != 2:
        raise ValueError("Invalid format: could not split reasoning and answer by ###")
    
    # 1. reasoning 파싱
    reasoning_block = parts[0]
    reasoning_lines = reasoning_block.splitlines()
    for line in reasoning_lines:
        if line.strip().lower().startswith("step"):
            result["reasoning"].append(line.strip())
    
    # 2. steps 파싱
    answer_block = parts[1]
    lines = answer_block.splitlines()
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if line.lower().startswith("steps:"):
            current_section = "steps"
        elif current_section == "steps":
            match = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
            if match:
                full_step = match.group(2).strip()
                # -- 구분자로 action과 objects 분리
                if " -- " in full_step:
                    action, objects_str = full_step.split(" -- ", 1)
                    objects = [obj.strip() for obj in objects_str.split(',') if obj.strip()]
                else:
                    action = full_step
                    objects = []
                
                result["steps"].append({
                    "action": action.strip(),
                    "objects": objects
                })
    
    return result

def parse_llm_non_json2json_output_for_vlm(text: str) -> dict:
    """
    Parses LLM output of the form:
    
    reasoning:
    <multiple lines>
    ###
    answer:
    <comma-separated list of objects> or "None"
    """

    # 강제 문자열 분리
    try:
        reasoning_block, output_block = text.strip().split("###")
    except ValueError:
        raise ValueError("Output missing '###' separator")

    # reasoning 처리
    reasoning_lines = []
    for line in reasoning_block.strip().splitlines():
        if line.lower().startswith("reasoning:"):
            continue
        cleaned = line.strip("- ").strip()
        if cleaned:
            reasoning_lines.append(cleaned)

    # output 처리
    output_line = output_block.strip()
    if not output_line.lower().startswith("answer:"):
        raise ValueError("Output block must start with 'answer:'")

    output_content = output_line[len("answer:"):].strip()

    if output_content.lower() == "none":
        answer = []
    else:
        answer = [obj.strip() for obj in output_content.split(",") if obj.strip()]

    return {
        "reasoning": reasoning_lines,
        "answer": answer
    }

def parse_llm_non_json2json_output_for_input_matching(text: str) -> dict:
    """
    Parses LLM output of the form:
    reasoning:
    <reasoning lines>
    ###
    answer:
    matched_ids: [1, 2]
    missing: ["cup", "glass"]
    """
    try:
        reasoning_block, output_block = text.strip().split("###")
    except ValueError:
        raise ValueError("Output missing '###' separator")

    # reasoning 처리
    reasoning_lines = []
    for line in reasoning_block.strip().splitlines():
        if line.lower().startswith("reasoning:"):
            continue
        cleaned = line.strip("- ").strip()
        if cleaned:
            reasoning_lines.append(cleaned)

    # output 처리
    output_lines = [
        line.strip() for line in output_block.strip().splitlines()
        if line.strip() and not line.lower().startswith("answer:")
    ]

    output_dict = {
        "matched_ids": [],
        "missing": []
    }

    for line in output_lines:
        match = re.match(r"(matched_ids|missing)\s*:\s*(.+)", line)
        if match:
            key = match.group(1)
            try:
                value = ast.literal_eval(match.group(2))
                if isinstance(value, list):
                    output_dict[key] = value
            except Exception:
                raise ValueError(f"Could not parse list value for '{key}': {match.group(2)}")

    return {
        "reasoning": reasoning_lines,
        "answer": output_dict
    }



def parse_simple_plan_editor_output(text: str) -> dict:
    """
    Parses output of Plan Editor Agent in the following hybrid format:

    reasoning:
    <step-by-step lines>
    ###
    answer:
    missing: ["object1", "object2"]

    또는

    answer:
    1. Step one...
    2. Step two...
    ...
    """
    try:
        reasoning_block, output_block = text.strip().split("###")
    except ValueError:
        raise ValueError("Output missing '###' separator")

    # reasoning 처리
    reasoning_lines = []
    for line in reasoning_block.strip().splitlines():
        if line.strip().lower().startswith("reasoning:"):
            continue
        cleaned = line.strip("- ").strip()
        if cleaned:
            reasoning_lines.append(cleaned)

    # answer 블록 처리
    output_lines = [
        line.strip() for line in output_block.strip().splitlines()
        if line.strip() and not line.lower().startswith("answer:")
    ]

    # case 1: missing line 존재
    for line in output_lines:
        if line.lower().startswith("missing:"):
            try:
                missing_list = ast.literal_eval(line[len("missing:"):].strip())
                if not isinstance(missing_list, list):
                    raise ValueError
                return {
                    "reasoning": reasoning_lines,
                    "answer_type": "missing",
                    "answer": {"missing": missing_list}
                }
            except Exception:
                raise ValueError(f"Could not parse missing list: {line}")

    # case 2: numbered step 목록
    step_lines = []
    for line in output_lines:
        if re.match(r"^\d+\.\s", line):
            step_lines.append(line)

    if step_lines:
        return {
            "reasoning": reasoning_lines,
            "answer_type": "steps",
            "answer": {"steps": step_lines}
        }

    # 아무것도 매칭 안 됨
    raise ValueError("Answer block must contain either 'missing:' or numbered steps.")

def split_plan_by_grip_release(lines):
    """
    Plan을 'grip' ~ 'release'를 기준으로 동작 단위 블록으로 분리.
    - grip 이전 step도 해당 grip-release 블록에 포함
    - 각 줄의 숫자 prefix 제거
    """
    import re

    blocks = []
    current_block = []
    pending_pre_grip = []
    inside_action = False

    for line in lines:
        # 숫자 prefix 제거
        line = re.sub(r'^\d+\.\s*', '', line)
        lower_line = line.lower()

        if not inside_action and "grip" not in lower_line:
            pending_pre_grip.append(line)
            continue

        if "grip" in lower_line:
            # grip 만나면 pre-grip 준비단계 포함해서 새 block 시작
            current_block = pending_pre_grip + [line]
            pending_pre_grip = []
            inside_action = True

        elif "release" in lower_line:
            current_block.append(line)
            blocks.append(current_block)
            current_block = []
            inside_action = False

        else:
            if inside_action:
                current_block.append(line)
            else:
                pending_pre_grip.append(line)

    # 마지막 블록 누락 방지
    if current_block:
        blocks.append(current_block)

    return blocks

######################################################### Hard agent 파서
def parse_agent_assignment_output(text: str) -> dict:
    """
    Hard Agent의 assign 프롬포트 출력을 파싱함
    Parses LLM output of the form:
    reasoning:
    Step 1: ...
    Step 2: ...
    ###
    answer:
    <Agent_Name>
    """
    try:
        reasoning_block, answer_block = text.strip().split("###")
    except ValueError:
        raise ValueError("Output missing '###' separator")

    # reasoning 추출
    reasoning_lines = []
    for line in reasoning_block.strip().splitlines():
        if line.strip().lower().startswith("reasoning:"):
            continue
        cleaned = line.strip()
        if cleaned:
            reasoning_lines.append(cleaned)

    # answer 추출
    answer_lines = answer_block.strip().splitlines()
    agent_name = None
    for line in answer_lines:
        if line.strip().lower().startswith("answer:"):
            continue
        cleaned = line.strip()
        if cleaned:
            agent_name = cleaned
            break

    if not agent_name:
        raise ValueError("No agent name found in answer block")

    return {
        "reasoning": reasoning_lines,
        "answer": agent_name
    }


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


def match_or_create_object(prev_list, new_det, tool_pos, distance_threshold=0.04):
    """
    개선된 객체 매칭 함수
    - 위치가 기존 객체와 겹치면 새 데이터를 삭제 (None 반환)
    - position_only_check=True: 위치만으로 중복 판단 (이름 무시)
    - position_only_check=False: 기존 방식 (같은 이름 + 위치)
    """
    # 0) parse (기존과 동일)
    new_det = parse_to_obj(new_det)
    if isinstance(new_det, str):

        try:
            new_det = json.loads(new_det.strip())
        except Exception as e:
            print(f"[MATCH][ERROR] Failed to parse new_det JSON: {e}")
            raise

    # 1) validation (기존과 동일)
    if not isinstance(new_det, dict):
        raise TypeError(f"[MATCH][ERROR] new_det must be dict after parsing, got {type(new_det)}")
    required_keys = ("object_name", "position", "area")
    for key in required_keys:
        if key not in new_det:
            raise KeyError(f"[MATCH][ERROR] Missing key '{key}' in new_det")
    
    name = new_det["object_name"]
    pos = new_det["position"]
    
    # 2) unpack
    x_new, y_new, z_new = pos
    x_new, y_new, *_ = tool_to_world(tool_pos, round(x_new, 3), round(y_new, 3))
    # distance function
    def dist(o):
        p = o.get("position")
        if isinstance(p, dict):
            x0, y0, z0 = p["x"], p["y"], p["z"]
        else:
            x0, y0, z0 = p
        d = sqrt((x0 - x_new)**2 + (y0 - y_new)**2 + (z0 - z_new)**2)
        return d
    
    # 4) filter candidates - 개선된 부분
    # 위치만으로 중복 검사 (이름 무시)
    candidates = prev_list  # 모든 객체 검사
    print(f"[MATCH] Checking position-based duplicates for all {len(candidates)} objects")

    
    # 5) matching - 가장 가까운 객체 찾기
    if candidates:
        best = min(candidates, key=dist)
        best_dist = dist(best)
        
        if best_dist <= distance_threshold:

            
            return None  # 중복이므로 새 객체 생성하지 않음 (기존 객체 유지)
    
    # 6) new ID 생성 (기존과 동일)
    prefix = name.replace(" ", "_")
    new_id = generate_new_id(prev_list, prefix)
    
    print(f"[MATCH] Creating new object id={new_id}")

    
    # 7) create (기존과 동일)
    new_obj = {
        "id": new_id,
        "name": name,
        "position": {"x": round(x_new, 3), "y": round(y_new, 3), "z": round(z_new, 3)},
        "area": new_det["area"],
        "mask_base64": new_det["mask_base64"], # 7_30 추가
        "mask_shape": new_det["mask_shape"] # 7_30 추가
    }
    
    return new_obj


def generate_new_id(prev_list, prefix):
    """match_or_create_object에서 사용되는 새 ID 생성 함수"""
    suffixes = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for o in prev_list:
        m = pattern.match(o.get("id", ""))
        if m:
            suffixes.append(int(m.group(1)))
    next_idx = max(suffixes, default=-1) + 1
    return f"{prefix}_{next_idx}"


def find_closest_detection(raw: str, old_pos: dict):
    """
    이전에 감지된 객체 위치에서 가장 가까운 객체를 반환 -> 다른 물건 옮길때 건드려서 위치 이동했을 수도 있으니까
    Debug-enabled version:
    raw: '["{...}", "{...}", "{...}"]' 형태의 문자열
    old_pos: {'x':float, 'y':float, 'z':float}
    """
    # 1) JSON 문자열이라면 파싱
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception as e:
            print("[ERROR] Failed to parse JSON:", e)
            return None
    else:
        parsed = raw  # 이미 파싱된 상태라고 가정

    # 2) 리스트 안 요소가 str이면 dict로 파싱, 아니면 그대로
    dets = []
    if isinstance(parsed, list):
        for idx, s in enumerate(parsed):
            if isinstance(s, str):
                try:
                    d = json.loads(s)
                    dets.append(d)
                except json.JSONDecodeError as e:
                    print(f"[WARN] Failed to parse element {idx}: {e}")
            elif isinstance(s, dict):
                dets.append(s)
            else:
                print(f"[WARN] Unexpected item type at index {idx}: {type(s)}")
    elif isinstance(parsed, dict):
        dets = [parsed]
    else:
        print("[ERROR] Parsed input is not a list or dict.")
        return None

    if not dets:
        print("[ERROR] No valid detection dicts parsed.")
        return None


    # 3) 거리 계산
    x0, y0, z0 = old_pos['x'], old_pos['y'], old_pos['z']
    best, best_dist = None, 0.1
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
        "area": best["area"],
        "mask_base64": best["mask_base64"],
        "mask_shape" : best["mask_shape"]

    }


##########################################################
# 파지 경로 찾는 코드 3 (Grasp Approach Direction Finder)
# 목표: 장애물 사이에서 로봇 그리퍼가 캔을 잡기 위한 가장 안전한 방향을 찾는다.
# 입력: target_object (dict), obstacles (list of dicts)
# 출력: 방향 벡터 (dx, dy) 및 상태 메시지
##########################################################





# 150° 이상인 모든 구간을 찾는 헬퍼 ######################################################### 7_9 추가함
import math





#################################### 카메라 관련 코드 ##################################################

def tool_to_world(toolpos, x_obj, y_obj, psi_obj=0, offset_x=0.09, offset_y=0.0):
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
        # psi_obj = obj.get("rotation_degree")

        # 월드 좌표 계산 (tool_to_world 함수 필요)
        x_w, y_w, w = tool_to_world(toolpos, x_obj, y_obj)
        print(x_w,y_w,w)
        # 원본 딕셔너리에 치환 혹은 추가
        obj["position"][0] = x_w
        obj["position"][1] = y_w
        # obj["rotation_degree"] = w
        updated.append(obj)
    return updated





def remove_target_from_detections(detections: list, target_obj: dict, tolerance: float = 1e-3):
    def parse(x):
        if isinstance(x, str):
            x = json.loads(x)
        if isinstance(x, (list, tuple)):  # [name,[x,y,z]] or [name,x,y,z]
            if x and isinstance(x[0], str):
                pos = x[1] if len(x) > 1 and isinstance(x[1], (list, tuple)) else x[1:4]
                x = {"object_name": x[0], "position": pos}
        if not isinstance(x, dict):
            return None, None
        name = str(x.get("object_name", "")).strip().lower()
        pos = [round(float(p), 5) for p in x.get("position", [])]
        if len(pos) != 3:
            return None, None
        return name, pos

    def is_same_object(d):
        try:
            n1, p1 = parse(d)
            n2, p2 = parse(target_obj)
            if not (n1 and n2):
                return False
            if n1 != n2:
                return False
            dist_sq = sum((a - b) ** 2 for a, b in zip(p1, p2))
            return dist_sq < tolerance ** 2
        except Exception:
            return False

    return [d for d in detections if not is_same_object(d)]


################# 받은 plan에서 물체가 놓일 장소에 따라 오프셋 적용 ##########################
def apply_plan_offset(direction, pos, offset_distance=0.15):
    """
    Apply position offset based on direction/location
    
    Args:
        direction (str): Direction or location (e.g., 'right', 'left', 'above', 'to', 'grasp', etc.)
        pos (list): Current position [x, y, z, roll, pitch, yaw]
        offset_distance (float): Distance to offset (default: 0.1 meters)
    
    Returns:
        list: Modified position [x, y, z, roll, pitch, yaw]
    """
    direction = direction.strip().lower()
    edited_pos = pos.copy()  # Make a copy to avoid modifying original
    
    # Apply offset based on direction/location
    if direction == 'right':
        edited_pos[1] -= offset_distance  # Move in -Y direction (right in robot frame)
        edited_pos[2] = 0.01
    elif direction == 'left':
        edited_pos[1] += offset_distance  # Move in +Y direction (left in robot frame)
        edited_pos[2] = 0.01
    elif direction == 'front' or direction == "in front":
        edited_pos[0] += offset_distance  # Move in +X direction (forward)
        edited_pos[2] = 0.01
    elif direction == 'behind':
        edited_pos[0] -= offset_distance  # Move in -X direction (backward)
        edited_pos[2] = 0.01
    # elif direction == 'up':
    #     edited_pos[2] += offset_distance  # Move in +Z direction (up)
    # elif direction == 'down':
    #     edited_pos[2] -= offset_distance  # Move in -Z direction (down)
    # elif direction == 'above':
    #     edited_pos[2] += offset_distance  # Move above current position
    # elif direction == 'below':
    #     edited_pos[2] -= offset_distance  # Move below current position
    
    return edited_pos

def detect_duplicate_targets_on_release(parsed):
    """
    supervisor가 출력한 액션에서 release에서 잡는 물체와 놓는 위치가 같은경우 그건 잘못된 거니 무시하게 하는 코드
    입력 : [{'agent': 'move', 'action': 'to', 'target': 'white_box_0'},{'agent': 'release', 'action': 'to', 'target': 'white_box_0, white_box_0'}] 이런 형태
    출력 : True (증복됨), False(증복 안됨)
    """
    for entry in parsed:
        if entry.get('agent') == 'release':
            targets = [t.strip() for t in entry.get('target', '').split(',')]
            if len(targets) == 2 and targets[0] == targets[1]:
                return True
    return False



############################################################ assemble mcp 서버의 함수를 굳이 딜레이 없이 사용하기 위해 복붙한거 ##########
def _as_float(x: Any) -> float:
    return float(x if not (isinstance(x, (list, tuple)) and len(x) == 1) else x[0])

def _xy_z_from_position(pos: Any):
    """position은 [x, y, z] 또는 {'x':..,'y':..,'z':..}만 허용."""
    if isinstance(pos, (list, tuple)):
        if len(pos) != 3:
            raise ValueError(f"position must be [x,y,z], got {pos!r}")
        return _as_float(pos[0]), _as_float(pos[1]), _as_float(pos[2])
    # dict도 허용 (혹시 올 수 있으니)
    x = _as_float(pos["x"]); y = _as_float(pos["y"]); z = _as_float(pos["z"])
    return x, y, z

def area_to_radius(area: float) -> float:
    a = _as_float(area)
    if a < 0: raise ValueError("area must be non-negative")
    return math.sqrt(a / math.pi)

def _segment_circle_intersect(p0, p1, c, r, eps: float = 1e-9) -> bool:
    x0,y0 = p0; x1,y1 = p1; cx,cy = c
    vx,vy = x1-x0, y1-y0
    wx,wy = cx-x0, cy-y0
    vv = vx*vx + vy*vy
    if vv <= eps:
        dx,dy = x0-cx, y0-cy
        return (dx*dx + dy*dy) <= (r*r + eps)
    t = (wx*vx + wy*vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx,qy = x0 + t*vx, y0 + t*vy
    dx,dy = qx-cx, qy-cy
    return (dx*dx + dy*dy) <= (r*r + eps)

# --- obstacles 정규화: 항상 [x,y,z,area] ---
def _extract_obstacles(obstacles: List[Dict[str, Any]]) -> List[List[float]]:
    """
    각 항목: {'position':[x,y,z], 'area': ... , ...}  (다른 키는 무시)
    반환: [[x,y,z,area], ...]
    잘못된 항목은 스킵.
    """
    out: List[List[float]] = []

    for obj in obstacles:
        try:
            x,y,z = _xy_z_from_position(obj["position"])
            area  = _as_float(obj["area"])
            out.append([x,y,z,area])
        except Exception:
            continue
    return out

# ---- 최종: start는 start_obj.position ----
def check_collision(
    start: Union[List[float], Dict[str, Any]],
    end: Union[List[float], Dict[str, Any]],
    obstacles: List[Dict[str, Any]],
):
    """
    description:
      Check if the line from start_obj.position to end collides with any obstacle
      If collided and z exists, return min z lift as z_margin; else z_margin = 0.0.

    input:
      - end_pos: [x, y] -> the end position
      - start_obj: { 'object_name': ... , 'position': {x, y, z},'area':..., } -> the start position of object to grasp and move
      - obstacles: [{ 'object_name': ... , 'position': {x, y, z},'area':..,}, ...] -> the obstacles information you provided.
    output:
        {"collided": bool, "z_margin": float} -> true means collided and z_margin means the margin to add to lift height before move to avoid collision
    """
    z_tol: float = 0.04
    # clearance: float = 0.07   # z 여유 -> 물건 잡고 미리 7cm 올린다고 가정 후 이걸 반영한거임
    z_tol = _as_float(z_tol)
    # clearance = _as_float(clearance)


    EPS_POS2 = 0.01   # xy 거리 제곱 허용
    EPS_Z    = 0.01
    EPS_AREA = 0.01
    try:
        sx, sy, sz = _xy_z_from_position(start)
        # start_area = float(start["area"])
        start_area = 0.01
        r_start = area_to_radius(start_area)
        p0 = (sx, sy); p1 = (float(end[0]), float(end[1]))

        collided_any = False
        max_margin = 0.0
        if obstacles == []:
            return [False, 0.0]
        obs = enumerate(_extract_obstacles(obstacles))

        for i, (ox, oy, oz, area) in obs:
            # 0) self-filter
            if ((ox - sx)**2 + (oy - sy)**2 <= EPS_POS2):
                # _LOG.debug("[obs %d] skipped: same as start object", i)
                continue

            # 1) z-필터
            if oz < (sz - z_tol):
                # _LOG.debug("[obs %d] ignored by z-filter (oz=%.6f < %.6f)", i, oz, sz - z_tol - clearance)
                continue

            # 2) 2D 충돌 판정
            r_eff = area_to_radius(area) + r_start
            hit = _segment_circle_intersect(p0, p1, (ox, oy), r_eff)
            # _LOG.debug("[obs %d] pos=(%.6f,%.6f,%.6f) area=%.6f r_eff=%.6f hit=%s",
                    # i, ox, oy, oz, area, r_eff, hit)
            if hit:
                collided_any = True
                margin = max(0.0, (oz + z_tol) - sz + 0.05)
                if margin > max_margin:
                    max_margin = margin

        # 루프 종료 후 최종 반환
        if collided_any:
            return [True, max_margin]
        else:
            return [False, 0.0]
    except Exception as e:
        print("[collision detect non mcp ] error occured :",e)
        return [False, f"error occured{e}"]
    


# 증복 물체들 제거하는 코드

def is_close_position(p1, p2, threshold=0.01):
    return all(abs(a - b) <= threshold for a, b in zip(p1, p2))

def deduplicate_by_position(objects, threshold=0.007):
    """
    위치를 보고 동일 위치의 물건이면 가장 스코어 높은 것만 반환함
    """
    filtered = []
    for obj in objects:
        matched = False
        for i, existing in enumerate(filtered):
            if is_close_position(obj["position"], existing["position"], threshold):
                if obj["score"] > existing["score"]:
                    filtered[i] = obj  # 더 높은 score로 대체
                matched = True
                break
        if not matched:
            filtered.append(obj)
    return filtered