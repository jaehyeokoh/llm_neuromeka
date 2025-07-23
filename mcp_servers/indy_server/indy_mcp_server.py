from neuromeka import IndyDCP3,TaskBaseType, JointBaseType,BlendingType,Limits,PostCondition
from mcp.server.fastmcp import FastMCP
import time 
import logging
import numpy as np


# 프로그램 시작 시 한 번만 설정
logging.basicConfig(
    level=logging.INFO,                # 로그 레벨 설정
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# 서버에 고유한 이름 부여
mcp = FastMCP("indy7_mcp_server")

# indy = IndyDCP3(robot_ip='192.168.0.91', index=0)
indy = IndyDCP3(robot_ip='192.168.0.144', index=0)
data = indy.get_robot_data()



def tool_x_tilt():
    """
    툴의 Z축을 기준으로 툴의 X축이 아래 방향으로부터 얼마나 회전(편향)했는지
    부호 있는 각도로 계산하여 반환합니다.
    """
    # 1. RPY 각도 가져오기
    #    indy.get_robot_data()['p']는 [x, y, z, rx, ry, rz] 형태로 반환
    _, _, _, rx_deg, ry_deg, rz_deg = indy.get_robot_data()['p']
    
    # 2. 각도를 라디안으로 변환 (numpy 삼각 함수 사용을 위해)
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    
    # 3. Roll, Pitch, Yaw의 사인/코사인 미리 계산
    cx, sx = np.cos(rx), np.sin(rx)  # Roll
    cy, sy = np.cos(ry), np.sin(ry)  # Pitch
    cz, sz = np.cos(rz), np.sin(rz)  # Yaw
    
    # 4. 회전 행렬 R = Rz * Ry * Rx 구성
    #    이는 툴 프레임에서 베이스 프레임으로 벡터를 변환
    R = np.array([
        [ cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx ],
        [ sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx ],
        [   -sy,           cy*sx,           cy*cx   ]
    ])
    
    # 5. 베이스 프레임에서의 툴 축 벡터 추출
    x_tool = R[:, 0]  # 툴 X축
    z_tool = R[:, 2]  # 툴 Z축
    
    # 6. 아래 방향 기준 벡터 정의 (베이스 Z축 아래쪽)
    v_down = np.array([0, 0, -1])
    
    # 7. 툴 Z축 평면(툴 XY평면)에 투영하여 컴포넌트 구하기
    def project_onto_plane(vec, normal):
        return vec - normal * np.dot(vec, normal)
    
    proj_x = project_onto_plane(x_tool, z_tool)
    proj_down = project_onto_plane(v_down, z_tool)
    
    # 8. 투영된 벡터 간 부호 있는 각도 계산
    #    arctan2( z_tool · (proj_down × proj_x), proj_down · proj_x )
    angle_rad = np.arctan2(
        np.dot(z_tool, np.cross(proj_down, proj_x)),
        np.dot(proj_down, proj_x)
    )
    
    # 9. 라디안을 도(degree)로 변환하여 반환
    return np.degrees(angle_rad)


def wait_idle():
    """
    이거 넣으면 이전 동작 완료 전까지 다음 동작 안함
    """
    a = 0
    while indy.get_control_data()['op_state'] !=5:
        a = a+1
        time.sleep(0.1)
    return

@mcp.tool()
def check_IK(target_pose, meter = True):
    """
    주어진 좌표를 갈 수 있는지 확인
    갈 수 있음: True
    갈 수 없음: False
    meter = 입력이 m단위인지
    """

    # numpy 타입을 일반 float/int로 변환
    if isinstance(target_pose, (list, tuple, np.ndarray)):
        target_pose = [float(v) for v in target_pose]

    if meter:
        target_pose[:3] = [v * 1000 for v in target_pose[:3]]

    init_jpos = indy.get_control_data()['q']
    success = indy.inverse_kin(target_pose, init_jpos)
    print(success)
    b = success["response"]["code"]
    if b == "0":
        return True
    # return False
    return {"msg":success, "pos":target_pose}

@mcp.tool()
def robot_get_tcp_pose() -> list:
    wait_idle()
    state = indy.get_control_data()['p']

    state = [round(0.001*p, 4) for p in state[:3]] +  [round(p, 3) for p in state[3:]]
    return state  # [x, y, z, r,p,w] in m
# 엔드 포인트를 taskmove로 이동, 단위는 mm, -> x,y,z,r,p,y(deg)

@mcp.tool()
def move_home():
    home_pos = indy.get_home_pos()['jpos']
    indy.movej(home_pos,
                blending_type=BlendingType.NONE,
                base_type=JointBaseType.ABSOLUTE,
                blending_radius=0.0,
                vel_ratio=20,
                acc_ratio=20,
                post_condition=PostCondition(),
                teaching_mode=False)

    # indy.move_home()
    return

@mcp.tool()
def robot_move(x: float, y: float, z: float, w: float=0, r: float = 0, p: float = 180,  bypass_singular = False) -> str:
    pos = [x*1000, y*1000, z*1000, r, p, w]
    wait_idle()

    if not check_IK(pos, meter = False): # 역기구학 계산해서 출력 true, false 내보내는 코드
        indy.movel(pos, vel_ratio=20, acc_ratio=15, bypass_singular =  bypass_singular)
        logging.log(logging.ERROR, "역기구학 계산 실패")
        return False
    else:
        indy.movel(pos, vel_ratio=20, acc_ratio=15, bypass_singular =  bypass_singular)
        # 로봇 작동중이면 while에 갖히게 하는 코드

        return f"Moved to position: {pos} (mm) in task space."

@mcp.tool()
def robot_grab():
    wait_idle()
    a = indy.execute_tool('grip_close')
    return a

@mcp.tool()
def robot_release():
    wait_idle()
    a = indy.execute_tool('grip_open')
    return a


@mcp.tool()
def robot_grip_rotate():
    """
    그리퍼를 수평으로 맞춘다.
    """
    wait_idle()
    angle = tool_x_tilt()
    jpos = indy.get_control_data()['q']
    # jpos[5] -= angle
    # indy.movej(jtarget=jpos, vel_ratio=80, acc_ratio=150)
    # wait_idle()
    new_angle = jpos[5] - angle
    
    # 관절 제한 체크 및 보정 (인디는 -215에서 215)
    if new_angle > 190:
        new_angle -= 360  # 반대 방향으로
    elif new_angle < -190:
        new_angle += 360  # 반대 방향으로
    
    jpos[5] = new_angle
    indy.movej(jtarget=jpos, vel_ratio=80, acc_ratio=150)
    wait_idle()
    _, _, _, r, p, w = indy.get_robot_data()['p']
    return [r,p,w]



##############################################################################


# @mcp.tool()
# def robot_check_grab() -> str:
#     if indy.get_gripper_state() == "holding":
#         return "Grab success"
#     return "Grab failed"

# --- MCP 서버 시작 ---
if __name__ == "__main__":
    print("MCP 서버 루프 시작...")
    try:
        # move_home() # 먼저 홈 위치로
        # indy.movel([450,150,450,0.1,179.9,0.1])
        # indy.execute_tool('grip_open') # 그리퍼 오픈하고
        # wait_idle()
        mcp.run()
    except KeyboardInterrupt:
        print("\n사용자에 의해 서버 중지됨 (Ctrl+C).")
    except Exception as e:
        print(f"\nMCP 서버 오류 발생: {e}")



# # --- 디버그용---
# if __name__ == "__main__":
#     # indy.movel([450,150,450,0.1,179.9,0.1])
#     # indy.movel([450,150,450,10,150,0])
#     robot_move_with_grip_align(0.52,0.19,0.25,0,-32,52)




















