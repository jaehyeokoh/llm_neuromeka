import math
from typing import List, Tuple, Optional, Dict, Any, Union
from mcp.server.fastmcp import FastMCP
import os
import logging
# --- logger setup (모듈 로드 시 1회) ---
import sys

# _LOG = logging.getLogger("assemble_tool_mcp_server.check_collision")
# if not _LOG.handlers:
#     h = logging.StreamHandler(stream=sys.stderr)
#     h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
#     _LOG.addHandler(h)
#     _LOG.propagate = False     # 루트로 전파 금지 (중복 출력 방지)
# # 개발 중엔 고정 DEBUG, 배포시 INFO로 내리면 됨
# _LOG.setLevel(logging.DEBUG)



# 서버에 고유한 이름 부여
mcp = FastMCP("assemble_tool_mcp_server")


# 전역변수 
obstacles_save = None


# --- utils ---
def _as_float(x: Any) -> float:
    return float(x if not (isinstance(x, (list, tuple)) and len(x) == 1) else x[0])

def _xy_z_from_position(pos: Any) -> Tuple[float, float, float]:
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

@mcp.tool()
def save_obstacles(obstacles: List[Dict[str,Any]]) ->str:
    """
    obstacles 정보를 저장함
    """
    global obstacles_save
    obstacles_save = obstacles
    return "[save_obstacles] completed_obstacles_saved_(assemble_server)"


@mcp.tool()
def get_obstacles() -> List[Dict[str,Any]]:
    """저장된 obstacles 정보를 가져옴"""
    global obstacles_save
    return obstacles_save


# ---- 최종: start는 start_obj.position ----
@mcp.tool()
def check_collision(
    start_obj: Dict[str, Any],
    end: Union[List[float], Dict[str, Any]],  # 두 타입 모두 허용
) -> Dict[str, Union[str, float]]:
    """
    Check collision along movement path and calculate safe lift height.

    Checks if moving an object from start_obj.position to end position 
    would collide with any obstacles in the saved obstacles data.

    Args:
        start_obj: Object to move
            - position: {x: float, y: float, z: float} - current position  
            - area: float - object area
        end: if target is location = [x,y], if target is object = position: {x: float, y: float, z: float}

    Returns:
        {
            "result": "collided"/"not collided"/ "error as e"
            "z_margin": float   # Additional height needed to avoid collision (meters)
        }

    Example:
    - example ( if B is object )
    Your plan: move A to B
    Objects info you received: [
        {"name": "A", "position": {"x": 1.0, "y": 2.0, "z": 0.5}, "area": 0.1},
        {"name": "B", "position": {"x": 3.0, "y": 4.0, "z": 0.3}, "area": 0.05}
    ]
    
    Function call:
    start_obj = {"position": {"x": 1.0, "y": 2.0, "z": 0.5}, "area": 0.1}  # A's info
    end =   {"position": {"x": 3.0, "y": 4.0, "z": 0.3}, "area": 0.05} # B's info

    - example (if B is location coordinate)
    Your plan: move A to [0.2,0.3,0.2]
    Objects info you received: [
        {"name": "A", "position": {"x": 1.0, "y": 2.0, "z": 0.5}, "area": 0.1},
    ]
    Function call:
    start_obj = {"position": {"x": 1.0, "y": 2.0, "z": 0.5}, "area": 0.1}  # A's info
    end =   [0.2,0.3,0.2]
    """

    #       - obstacles: [{ 'object_name': str , 'position': {x, y, z},'area':float},{} ...] -> the **every** objects

    global obstacles_save
    obstacles = obstacles_save

    z_tol: float = 0.0
    clearance: float = 0.07   # z 여유 -> 물건 잡고 미리 7cm 올린다고 가정 후 이걸 반영한거임
    z_tol = _as_float(z_tol)
    clearance = _as_float(clearance)
    
    EPS_POS2 = 0.01   # xy 거리 제곱 허용
    try:
        sx, sy, sz = _xy_z_from_position(start_obj["position"])
        start_area = float(start_obj["area"])

        try:
            endx, endy, endz = _xy_z_from_position(end["position"])
        except:
            endx = end[0]
            endy = end[1]
            

        r_start = area_to_radius(start_area)
        p0 = (sx, sy); p1 = (float(endx), float(endy))

        collided_any = False
        max_margin = 0.0
        max_idx = -1

        for i, (ox, oy, oz, area) in enumerate(_extract_obstacles(obstacles)):
            # filtering start obj from obstacles
            if ((ox - sx)**2 + (oy - sy)**2 <= EPS_POS2):
                # _LOG.debug("[obs %d] skipped: same as start object", i)
                continue

            # filtering end from obstacles
            if (abs(ox - endx) <= EPS_POS2 and abs(oy - endy) <= EPS_POS2):
                continue

            # 1) z-필터 (장애물이 시작 지점보다 낮으면 그냥 통과)
            if oz < (sz - z_tol + clearance): # 시작지점의 물건 z +그리퍼가 위로 기본적으로 clearance만큼 든다는 가정
                # _LOG.debug("[obs %d] ignored by z-filter (oz=%.6f < %.6f)", i, oz, sz - z_tol - clearance)
                continue

            # 2) 2D 충돌 판정
            r_eff = area_to_radius(area) + r_start
            hit = _segment_circle_intersect(p0, p1, (ox, oy), r_eff)
            # _LOG.debug("[obs %d] pos=(%.6f,%.6f,%.6f) area=%.6f r_eff=%.6f hit=%s",
                    # i, ox, oy, oz, area, r_eff, hit)
            if hit:
                collided_any = True
                margin = max(0.0, (oz + z_tol - clearance) - sz)
                if margin > max_margin:
                    max_margin = margin
                    # max_idx = i
                # _LOG.debug("[obs %d] COLLISION -> margin=%.6f (current max=%.6f idx=%d)",
                #         i, margin, max_margin, max_idx)

        # 루프 종료 후 최종 반환
        if collided_any:
            # _LOG.debug("[check_collision] FINAL: collided=True, z_margin=%.6f (from obs %d)", max_margin, max_idx)
            return {"result": "collided", "z_margin": float(max_margin)}
        else:
            # _LOG.debug("[check_collision] FINAL: collided=False, z_margin=0.0")
            return {"result": "not collided", "z_margin": float(0.0)}
    except Exception as e:
        # _LOG.debug(f"[check_collision] Error Occured: collided=False, z_margin=0.0, error is : {e}")
        return {"result": f"error as {e}", "z_margin": float(0.4)}

# --- MCP 서버 시작 ---
if __name__ == "__main__":
    try:
        mcp.run()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass


