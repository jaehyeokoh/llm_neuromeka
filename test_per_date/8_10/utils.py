"""General-purpose utilities for MCP config, JSON extraction, detection matching,
camera/world transforms, collision checks, and simple geometry helpers.

- 행동 불변(Behavior preserved): 로직/출력/예외 메시지 변경 없음
- 정리만 수행: 포맷/간단 독스트링/가독성 향상, 불필요한 공백 제거
"""

# from __future__ import annotations

import base64
import inspect
import json
import math
import re
from math import inf, sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ============================== 초기 설정값 ==============================


# ============================ MCP 설정 관련 코드 ============================
def load_mcp_config(config_name) -> Optional[Dict[str, Any]]:
    """`./config_name.json`에서 MCP 설정을 로드."""
    try:
        with open(f"./{config_name}.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"MCP 설정 파일 로드 중 오류 발생: {str(e)}")
        return None


def create_server_config(config_name = "mcp_config") -> Dict[str, Dict[str, Any]]:
    """로드된 MCP 설정으로 서버 구성 딕셔너리 생성."""
    config = load_mcp_config(config_name)
    server_config: Dict[str, Dict[str, Any]] = {}
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


# ========================= JSON 추출 및 파싱 관련 함수 =========================
def extract_json(raw: str) -> Any:
    """문자열에서 최외곽 JSON 블록(객체/리스트)을 찾아 `json.loads`로 파싱."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    # 시작 위치
    start_obj = raw.find("{")
    start_list = raw.find("[")
    start = start_obj
    if start_list != -1 and (start_list < start or start == -1):
        start = start_list

    # 끝 위치
    end_obj = raw.rfind("}")
    end_list = raw.rfind("]")
    end = max(end_obj, end_list)

    if start == -1 or end == -1 or start > end:
        raise ValueError("JSON 블록을 찾을 수 없습니다.")

    json_str = raw[start : end + 1]
    return json.loads(json_str)


def convert_list_to_dict(obj_list: Union[List[Dict[str, Any]], Dict[str, Any]]) -> Dict[str, Any]:
    """리스트를 `{id: item}` 딕셔너리로 변환. 이미 dict면 그대로 반환."""
    if isinstance(obj_list, dict):
        return obj_list
    if isinstance(obj_list, list):
        return {item["id"]: item for item in obj_list}
    raise TypeError(f"Expected list or dict, got {type(obj_list)}: {obj_list}")


def format_actions_for_llm(action_list: List[str]) -> List[str]:
    """문자열 액션 리스트를 1-based 넘버링 포맷으로 변환."""
    return [f"{i + 1}. {action}" for i, action in enumerate(action_list)]


# ======================= node 생성 헬퍼 (비동기 래퍼) =======================
def wrap_async(fn, **fixed_kwargs):
    """
    임의의 async 함수 `fn`에 대해 (state, config)를 자동 주입하고,
    나머지 인자는 fixed_kwargs로 전달하는 래퍼를 생성.
    """
    sig = inspect.signature(fn)

    async def wrapper(state, config):
        param_names = list(sig.parameters.keys())
        args: List[Any] = []
        kwargs: Dict[str, Any] = {}

        for name in param_names:
            if name == "state":
                args.append(state)
            elif name == "config":
                kwargs["config"] = config
            elif name in fixed_kwargs:
                args.append(fixed_kwargs[name])
            else:
                # 없는 인자는 기본값 사용
                pass

        return await fn(*args, **kwargs)

    return wrapper


# ================= GPT 출력(action_supervisor) 포맷팅 관련 함수 =================
def remove_duplicates_by_id(objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """리스트 내에서 같은 id를 가진 객체를 하나로 정리."""
    try:
        seen = set()
        unique_objects: List[Dict[str, Any]] = []
        for obj in objects:
            obj_id = obj.get("id")
            if obj_id not in seen:
                seen.add(obj_id)
                unique_objects.append(obj)
        return unique_objects
    except Exception:
        return []


# =========================== 객체 위치 피드백 관련 함수 ===========================
def parse_to_obj(raw: Union[str, Dict[str, Any], List[Any]]) -> Union[Dict[str, Any], List[Any]]:
    """`raw`가 dict/list면 그대로, 문자열이면 `extract_json`으로 파싱 후 반환."""
    if not isinstance(raw, str):
        return raw
    return extract_json(raw)


def match_or_create_object(
    prev_list: List[Dict[str, Any]],
    new_det: Union[str, Dict[str, Any]],
    tool_pos: Optional[List[float]],
    distance_threshold: float = 0.04,
) -> Dict[str, Any]:
    """
    개선된 객체 매칭:
    - 기존 위치와 충분히 가까우면 기존 ID를 쓰는 새 객체 반환
    - 아니면 새 ID 생성
    """
    # 0) parse
    new_det = parse_to_obj(new_det)
    if isinstance(new_det, str):
        try:
            new_det = json.loads(new_det.strip())
        except Exception as e:
            print(f"[MATCH][ERROR] Failed to parse new_det JSON: {e}")
            raise

    # 1) validation
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
    if tool_pos is not None:
        x_new, y_new, *_ = tool_to_world(tool_pos, round(x_new, 3), round(y_new, 3))

    # distance function
    def dist(o: Dict[str, Any]) -> float:
        p = o.get("position")
        if isinstance(p, dict):
            x0, y0, z0 = p[0], p[1], p[2]
        else:
            x0, y0, z0 = p
        return sqrt((x0 - x_new) ** 2 + (y0 - y_new) ** 2 + (z0 - z_new) ** 2)

    # 4) 모든 후보를 위치 기반으로 검사
    candidates = prev_list
    print(f"[MATCH] Checking position-based duplicates for all {len(candidates)} objects")

    # 5) 가장 가까운 객체가 임계값 이하면 기존 ID로 새 객체 구성
    if candidates:
        best = min(candidates, key=dist)
        best_dist = dist(best)
        if best_dist <= distance_threshold:
            new_obj = {
                "id": best["id"],  # 기존 객체 ID 사용
                "object_name": name,
                "position": [round(x_new, 3), round(y_new, 3), round(z_new, 3)],
                "area": new_det["area"],
                "mask_base64": new_det["mask_base64"],
                "dimensions": new_det["dimensions"],
                "rotation_degree": new_det["rotation_degree"],
                "score": new_det["score"],
            }
            return new_obj

    # 6) new ID 생성
    prefix = name.replace(" ", "_")
    new_id = generate_new_id(prev_list, prefix)
    print(f"[MATCH] Creating new object id={new_id}")

    # 7) create
    return {
        "id": new_id,
        "object_name": name,
        "position": [round(x_new, 3), round(y_new, 3), round(z_new, 3)],
        "area": new_det["area"],
        "mask_base64": new_det["mask_base64"],  # 7_30 추가
        "dimensions": new_det["dimensions"],
        "rotation_degree": new_det["rotation_degree"],
        "score": new_det["score"],
    }


def generate_new_id(prev_list: List[Dict[str, Any]], prefix: str) -> str:
    """`match_or_create_object`에서 사용하는 새 ID 생성."""
    suffixes: List[int] = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for o in prev_list:
        m = pattern.match(o.get("id", ""))
        if m:
            suffixes.append(int(m.group(1)))
    next_idx = max(suffixes, default=-1) + 1
    return f"{prefix}_{next_idx}"


def find_closest_detection(raw: Union[str, List[Any], Dict[str, Any]], old_pos: Dict[str, float], id: str) -> Optional[Dict[str, Any]]:
    """
    이전 감지 위치 근방에서 가장 가까운 감지 결과 선택.
    - raw: '["{...}", "{...}", ...]' 형태 문자열 또는 이미 파싱된 리스트/딕셔너리
    - old_pos: {'x': float, 'y': float, 'z': float} (주의: 내부에서는 인덱싱 사용)
    """
    # 1) JSON 파싱
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception as e:
            print("[ERROR] Failed to parse JSON:", e)
            return None
    else:
        parsed = raw

    # 2) 요소 파싱
    dets: List[Dict[str, Any]] = []
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
    x0, y0, z0 = old_pos[0], old_pos[1], old_pos[2]
    best, best_dist = None, 0.1
    for d in dets:
        pos = d.get("position")
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3):
            continue
        dist = sum((p - q) ** 2 for p, q in zip(pos, (x0, y0, z0)))
        if dist < best_dist:
            best_dist, best = dist, d

    if best is None:
        return None

    return {
        "id": id,
        "object_name": best["object_name"],
        "position": [round(c, 3) for c in best["position"]],
        "area": best["area"],
        "mask_base64": best["mask_base64"],
    }


# =============================== 카메라 관련 코드 ===============================
def tool_to_world(
    toolpos: List[float],
    x_obj: float,
    y_obj: float,
    psi_obj: float = 0.0,
    offset_x: float = 0.09,
    offset_y: float = 0.0,
) -> Tuple[float, float, float]:
    """
    로봇 베이스(월드) 기준 엔드툴 위치(toolpos)와 카메라 오프셋을 반영해,
    카메라 좌표계의 (x_obj, y_obj, psi_obj)를 월드 좌표계로 변환.
    """
    x_t = toolpos[0]
    y_t = toolpos[1]
    # psi_t = toolpos[5]
    psi_t = 0

    # 1) 툴→카메라 월드 위치
    c_t, s_t = np.cos(np.deg2rad(psi_t)), np.sin(np.deg2rad(psi_t))
    x_c = x_t + c_t * offset_x - s_t * offset_y
    y_c = y_t + s_t * offset_x + c_t * offset_y
    psi_c = psi_t  # yaw 오프셋 없음 가정

    # 2) 카메라→월드 물체 위치
    x_w = x_c + c_t * x_obj - s_t * y_obj
    y_w = y_c + s_t * x_obj + c_t * y_obj
    psi_w = psi_c + psi_obj
    return x_w, y_w, psi_w


def update_world_coordinates(toolpos: Tuple[float, float, float], objects: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    각 객체의 로컬 좌표를 월드 좌표로 변환하여 position에 반영.
    - toolpos: (x, y, z)
    """
    updated: List[Dict[str, Any]] = []
    for entry in objects:
        obj = json.loads(entry) if isinstance(entry, str) else entry.copy()
        x_obj, y_obj = obj["position"][0], obj["position"][1]
        x_w, y_w, w = tool_to_world(toolpos, x_obj, y_obj)
        print(x_w, y_w, w)
        obj["position"][0] = x_w
        obj["position"][1] = y_w
        updated.append(obj)
    return updated


def remove_target_from_detections(detections: List[Any], target_obj: Any, tolerance: float = 1e-3) -> List[Any]:
    """탐지 리스트에서 target 객체와 (이름+위치)가 같은 항목 제거."""

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
            return dist_sq < tolerance**2
        except Exception:
            return False

    return [d for d in detections if not is_same_object(d)]


# ============== 받은 plan에서 위치/방향에 따른 오프셋 적용 ==============
def apply_plan_offset(direction: str, pos: List[float], offset_distance: float = 0.15, grasp_z: float = 0.07) -> List[float]:
    """
    direction(좌/우/앞/뒤)에 따라 pos를 이동. z는 grasp 높이에 맞춰 보정.
    - 반환: 수정된 [x, y, z, roll, pitch, yaw]
    """
    direction = direction.strip().lower()
    edited_pos = pos.copy()
    if grasp_z >= 0.11:
        grasp_z += 0.03
    height = grasp_z + 0.05

    if direction == "right":
        edited_pos[1] -= offset_distance
        edited_pos[2] = height
    elif direction == "left":
        edited_pos[1] += offset_distance
        edited_pos[2] = height
    elif direction == "front" or direction == "in front":
        edited_pos[0] += offset_distance
        edited_pos[2] = height
    elif direction == "behind":
        edited_pos[0] -= offset_distance
        edited_pos[2] = height
    return edited_pos


def detect_duplicate_targets_on_release(parsed: List[Dict[str, Any]]) -> bool:
    """
    release 액션에서 잡는 물체와 놓는 위치가 동일하면 True.
    입력 예: [{'agent':'release','action':'to','target':'white_box_0, white_box_0'}]
    """
    for entry in parsed:
        if entry.get("agent") == "release":
            targets = [t.strip() for t in entry.get("target", "").split(",")]
            if len(targets) == 2 and targets[0] == targets[1]:
                return True
    return False


# ======= assemble MCP 서버 함수 로컬 복제(지연 제거 목적, 로직 동일) =======
def _as_float(x: Any) -> float:
    """리스트/튜플의 단일 원소도 float로 안전 변환."""
    return float(x if not (isinstance(x, (list, tuple)) and len(x) == 1) else x[0])


def _xy_z_from_position(pos: Any) -> Tuple[float, float, float]:
    """position은 [x, y, z] 또는 {'x':..,'y':..,'z':..}만 허용."""
    if isinstance(pos, (list, tuple)):
        if len(pos) != 3:
            raise ValueError(f"position must be [x,y,z], got {pos!r}")
        return _as_float(pos[0]), _as_float(pos[1]), _as_float(pos[2])
    # dict도 허용
    x = _as_float(pos["x"])
    y = _as_float(pos["y"])
    z = _as_float(pos["z"])
    return x, y, z


def area_to_radius(area: float) -> float:
    """면적을 동일 면적 원의 반지름으로 변환."""
    a = _as_float(area)
    if a < 0:
        raise ValueError("area must be non-negative")
    return math.sqrt(a / math.pi)


def _segment_circle_intersect(p0: Tuple[float, float], p1: Tuple[float, float], c: Tuple[float, float], r: float, eps: float = 1e-9) -> bool:
    """선분 p0→p1과 원(c,r)의 교차 여부 2D 판정."""
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = c
    vx, vy = x1 - x0, y1 - y0
    wx, wy = cx - x0, cy - y0
    vv = vx * vx + vy * vy
    if vv <= eps:
        dx, dy = x0 - cx, y0 - cy
        return (dx * dx + dy * dy) <= (r * r + eps)
    t = (wx * vx + wy * vy) / vv
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    qx, qy = x0 + t * vx, y0 + t * vy
    dx, dy = qx - cx, qy - cy
    return (dx * dx + dy * dy) <= (r * r + eps)


def _extract_obstacles(obstacles: List[Dict[str, Any]]) -> List[List[float]]:
    """
    각 항목: {'position':[x,y,z], 'area': ...}  (다른 키 무시)
    반환: [[x,y,z,area], ...]  — 잘못된 항목은 스킵
    """
    out: List[List[float]] = []
    for obj in obstacles:
        try:
            x, y, z = _xy_z_from_position(obj["position"])
            area = _as_float(obj["area"])
            out.append([x, y, z, area])
        except Exception:
            continue
    return out


def check_collision(
    start: Union[List[float], Dict[str, Any]],
    end: Union[List[float], Dict[str, Any]],
    obstacles: List[Dict[str, Any]],
) -> List[Union[bool, float, str]]:
    """
    start.position(또는 start 벡터)에서 end(xy)로 이동 중 2D 충돌을 평가.
    충돌 시 z 여유(margin)를 계산해 반환.
    반환: [collided(bool), z_margin(float)]
    """
    z_tol: float = 0.04
    z_tol = _as_float(z_tol)

    EPS_POS2 = 0.01  # xy 거리 제곱 허용
    EPS_Z = 0.01
    EPS_AREA = 0.01  # 현재 로직에선 사용하지 않지만 유지

    try:
        sx, sy, sz = _xy_z_from_position(start)
        # start_area = float(start["area"])
        start_area = 0.01
        r_start = area_to_radius(start_area)
        p0 = (sx, sy)
        p1 = (float(end[0]), float(end[1]))

        collided_any = False
        max_margin = 0.0
        if obstacles == []:
            return [False, 0.0]

        for i, (ox, oy, oz, area) in enumerate(_extract_obstacles(obstacles)):
            # 0) self-filter
            if ((ox - sx) ** 2 + (oy - sy) ** 2) <= EPS_POS2:
                continue

            # 1) z-필터
            if oz < (sz - z_tol):
                continue

            # 2) 2D 충돌 판정
            r_eff = area_to_radius(area) + r_start
            hit = _segment_circle_intersect(p0, p1, (ox, oy), r_eff)
            if hit:
                collided_any = True
                margin = max(0.0, (oz + z_tol) - sz + 0.05)
                if margin > max_margin:
                    max_margin = margin

        return [True, max_margin] if collided_any else [False, 0.0]
    except Exception as e:
        print("[collision detect non mcp ] error occured :", e)
        return [False, f"error occured{e}"]


# ============================ 중복 물체 제거 관련 ============================
def is_close_position(p1: List[float], p2: List[float], threshold: float = 0.01) -> bool:
    """두 위치 p1, p2가 좌표별 오차 `threshold` 이내인지 여부."""
    return all(abs(a - b) <= threshold for a, b in zip(p1, p2))


def deduplicate_by_position(objects: List[Dict[str, Any]], threshold: float = 0.006) -> List[Dict[str, Any]]:
    """같은 위치의 물체는 score가 높은 항목만 남긴다."""
    filtered: List[Dict[str, Any]] = []
    for obj in objects:
        matched = False
        for i, existing in enumerate(filtered):
            if is_close_position(obj["position"], existing["position"], threshold):
                if obj["score"] > existing["score"]:
                    filtered[i] = obj
                matched = True
                break
        if not matched:
            filtered.append(obj)
    return filtered


# ================================ assembly agent ================================
def add_shape_from_id(objects: List[Dict[str, Any]], name_shape_map: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    객체 리스트에 'shape' 항목 추가. id 내 키워드로 도형 추론.
    - 기본 매핑: box/can/slab/cube/cylinder/brick/plate/ball/block
    """
    if name_shape_map is None:
        name_shape_map = {
            "box": "rectangular",
            "can": "cylinder",
            "slab": "flat",
            "cube": "rectangular",
            "cylinder": "cylinder",
            "brick": "rectangular",
            "plate": "flat",
            "ball": "sphere",
            "block": "rectangular",
        }

    def extract_shape_flexible(object_id: str) -> str:
        object_id_lower = object_id.lower()
        for keyword, shape in name_shape_map.items():
            if keyword in object_id_lower:
                return shape
        return "unknown"

    for obj in objects:
        obj["shape"] = extract_shape_flexible(obj["id"])
    return objects


def filter_objs_dimensions_height_name_area(objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """다양한 키 변종을 허용하며 id/dimensions/area만 추려 정규화."""

    g = lambda d, *ks: next((d[k] for k in ks if k in d), None)
    pos = lambda p: [float(p[0]), float(p[1]), float(p[2])] if isinstance(p, (list, tuple)) and len(p) >= 3 else None
    nums = lambda v: [float(x) for x in v] if isinstance(v, (list, tuple)) else None

    out: List[Dict[str, Any]] = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        p = pos(g(o, "position", "pos", "center"))
        d = nums(g(o, "dimensions", "dimension", "size"))
        a = g(o, "area", "surface", "area_m2")
        out.append(
            {
                "id": g(o, "id", "object_id", "uuid"),
                # 'height': p[2],
                "dimensions": [min(d[0], d[1], p[2]), max(d[0], d[1], p[2])],
                "area": float(a),
            }
        )
    return out


def get_area_sorted_objects(normalized_objs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """넓이 순 정렬 후 `id`, `dimensions`, `shape`만 추출."""
    sorted_objs = sorted(normalized_objs, key=lambda x: x["area"], reverse=True)
    area_sorted: List[Dict[str, Any]] = []
    for obj in sorted_objs:
        area_sorted.append(
            {
                "id": obj["id"],
                "dimensions": obj["dimensions"],  # [shortest, longest]
                "shape": obj["shape"],
            }
        )
    return area_sorted


def get_similar_height_groups(normalized_objs: List[Dict[str, Any]], threshold: float = 0.15) -> List[Dict[str, Any]]:
    """장축(높이)이 비슷한 물체들을 그룹으로 묶어 반환."""
    objects_with_height: List[Dict[str, Any]] = []
    for obj in normalized_objs:
        longest_axis = obj["dimensions"][1]
        objects_with_height.append({"id": obj["id"], "longest_axis": longest_axis, "shape": obj["shape"]})

    objects_with_height.sort(key=lambda x: x["longest_axis"])

    groups: List[Dict[str, Any]] = []
    used_objects: set[str] = set()

    for i, obj1 in enumerate(objects_with_height):
        if obj1["id"] in used_objects:
            continue

        group = [obj1]
        used_objects.add(obj1["id"])

        for j, obj2 in enumerate(objects_with_height[i + 1 :], i + 1):
            if obj2["id"] in used_objects:
                continue
            longest_diff = abs(obj1["longest_axis"] - obj2["longest_axis"]) / max(obj1["longest_axis"], obj2["longest_axis"])
            if longest_diff <= threshold:
                group.append(obj2)
                used_objects.add(obj2["id"])

        if len(group) >= 1:
            group_objects = [{"id": o["id"], "shape": o["shape"]} for o in group]
            groups.append({"group_id": len(groups) + 1, "objects": group_objects})

    return groups


# ================================ 방향 맵핑 그룹 ================================
direction_map_groups: Dict[str, List[str]] = {
    "above": ["above", "up"],
    "below": ["below", "down"],
    "left": ["left"],
    "right": ["right"],
    "front": ["front", "forward", "straight"],
    "behind": ["back", "behind"],
    "center": ["center", "middle"],
}


def map_direction(input_word: str) -> str:
    """여러 동의어를 표준 방향 토큰으로 매핑."""
    input_word = input_word.lower()
    for standard, synonyms in direction_map_groups.items():
        if input_word in synonyms:
            return standard
    return "center"


def apply_offset_and_reencode(
    parsed_objects: List[Dict[str, Any]],
    target_object_id: str,
    offset: Tuple[float, float, float],
    skip_if_none: bool = True,
) -> List[Dict[str, Any]]:
    """
    특정 id에 대해 포인트클라우드(base64) 디코드 → 오프셋 적용 → 재인코딩 후 교체.
    - 실패 항목은 건너뛰되, `skip_if_none=False`이면 None 값에 대해 예외
    """
    off = np.array(offset, dtype=np.float32)

    for obj_dict in parsed_objects:
        for obj_id, encoded_data in list(obj_dict.items()):
            if obj_id != target_object_id:
                continue

            if encoded_data is None:
                if skip_if_none:
                    print(f"[INFO] {obj_id} has None, skipped")
                    continue
                raise ValueError(f"{obj_id} value is None")

            try:
                b = base64.b64decode(encoded_data)
                pts = np.frombuffer(b, dtype=np.float32).reshape(-1, 3)
            except Exception as e:
                print(f"[WARNING] decode failed for {obj_id}: {e}")
                continue

            # 오프셋 적용
            pts = pts + off

            # 재인코딩
            new_bytes = pts.astype(np.float32).tobytes()
            new_b64 = base64.b64encode(new_bytes).decode("ascii")

            # 값 교체
            obj_dict[obj_id] = new_b64
            print(f"[INFO] {obj_id}: offset applied ({len(pts)} pts)")

    return parsed_objects
