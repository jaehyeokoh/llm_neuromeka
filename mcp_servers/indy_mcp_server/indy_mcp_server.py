from neuromeka import IndyDCP3,TaskBaseType, JointBaseType,BlendingType,Limits,PostCondition,StopCategory
from mcp.server.fastmcp import FastMCP
import time 
import logging
import numpy as np
import threading
import asyncio


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

@mcp.tool()
def wait_idle():
    """
    이거 넣으면 이전 동작 완료 전까지 다음 동작 안함
    """
    # a = 0
    while indy.get_control_data()['op_state'] !=5:
        # a = a+1
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
    # print(success)
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
    wait_idle() # 이거 꼭 넣기!!! 안넣으면 로봇 이상한 코브라 자세 함
    home_pos = indy.get_home_pos()['jpos']
    indy.movej(home_pos,
                blending_type=BlendingType.NONE,
                base_type=JointBaseType.ABSOLUTE,
                blending_radius=0.0,
                vel_ratio=25,
                acc_ratio=20,
                post_condition=PostCondition(),
                teaching_mode=False)

    # indy.move_home()
    wait_idle()
    indy.movel([450,150,500,0.001,179.99,0.001])
    wait_idle()
    return

@mcp.tool()
def move_default_pos():
    """
    기본 자세로 돌아옴
    """
    wait_idle()
    indy.movel([450,140,500,0.001,179.99,0.001],vel_ratio=27, acc_ratio=23, bypass_singular =  True)
    wait_idle()
    return

@mcp.tool()
def robot_move(x: float, y: float, z: float, w: float=0, r: float = 0, p: float = 180,  bypass_singular = False) -> str:
    pos = [x*1000, y*1000, z*1000, r, p, w]
    wait_idle()
    if not check_IK(pos, meter = False): # 역기구학 계산해서 출력 true, false 내보내는 코드
        indy.movel(pos, vel_ratio=23, acc_ratio=20, bypass_singular =  bypass_singular)
        logging.log(logging.ERROR, "역기구학 계산 실패")
        return False
    else:
        indy.movel(pos, vel_ratio=27, acc_ratio=23, bypass_singular =  bypass_singular)
        return f"Moved to position: {pos} (mm) in task space."

@mcp.tool()
def robot_custom_move(x=None, y=None, z=None, r=None, p=None, w=None, bypass_singular=True) -> str:
    """
    내가 원하는 것만 바꾸고 나머지는 현재 자세 유지하는 코드
    """
    wait_idle()
    pos = indy.get_control_data()['p']
    inputs = [x, y, z, r, p, w]
    
    # mm 변환이 필요한 인덱스 (0,1,2)
    for i, val in enumerate(inputs):
        if val is not None:
            pos[i] = val * 1000 if i < 3 else val
    indy.movel(pos, vel_ratio=25, acc_ratio=20, bypass_singular =  bypass_singular)
    return f"Moved to position: {pos} (mm) in task space."

async def wait_idle_async():
    """이전 동작 완료 전까지 다음 동작 안함 (비동기 버전)"""
    while indy.get_control_data()['op_state'] != 5:
        await asyncio.sleep(0.1)
    return

# @mcp.tool()
# async def robot_move(x: float, y: float, z: float, w: float = 0, r: float = 0, p: float = 180, bypass_singular: bool = False) -> str:
#     """로봇을 지정된 위치로 이동"""
#     pos = [x*1000, y*1000, z*1000, r, p, w]
#     indy.recover()
#     await wait_idle_async()
    
#     # 역기구학 체크 (나중에 필요시 주석 해제)
#     # if not check_IK(pos, meter=False):
#     #     logging.log(logging.ERROR, "역기구학 계산 실패")
#     #     return f"Moved to position: {pos} (mm) in task space without IK."
    
#     indy.movel(pos, vel_ratio=25, acc_ratio=20, bypass_singular=bypass_singular)
#     await asyncio.sleep(0.7)
    
#     success_count = 0
#     while True:
#         violation_code = indy.get_violation_data()["violation_str"]
        
#         if violation_code != '':  # 충돌 감지
#             logging.log(logging.ERROR, "충돌됨!!!!!!!!!!!!!!!!!")
#             indy.stop_motion(StopCategory.CAT2)
#             indy.recover()
#             logging.log(logging.ERROR, "충돌 후 회복 완료")
#             move_home
#             indy.movel([450, 150, 460, 0.1, 179.9, 0.1])
#             logging.log(logging.ERROR, "충돌 후 이동 완료")
#             await wait_idle()
#             indy.stop_motion(StopCategory.CAT2)
#             return 'collision'
#         else:
#             motion_status = indy.get_motion_data()["is_target_reached"]
#             if motion_status:  # 이동 완료
#                 success_count += 1
#                 if success_count >= 3:
#                     logging.log(logging.ERROR, "충돌없이 이동 완료")
#                     return f"Moved to position: {pos} (mm) in task space."
#             else:
#                 success_count = 0
            
#             await asyncio.sleep(0.1)

    

@mcp.tool()
def robot_prepare_assemble():
    """
    assemble 하기 전 눕는 자세 취하는 코드
    """
    wait_idle() # 이거 꼭 넣기!!! 안넣으면 로봇 이상한 코브라 자세 함
    home_pos = indy.get_home_pos()['jpos']
    indy.movej(home_pos,
                blending_type=BlendingType.NONE,
                base_type=JointBaseType.ABSOLUTE,
                blending_radius=0.0,
                vel_ratio=25,
                acc_ratio=20,
                post_condition=PostCondition(),
                teaching_mode=False)
    # wait_idle()
    robot_move(0.50,0.15,0.35,180,105,0)
    robot_grip_rotate()
    return

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

@mcp.tool()
def monitoring_and_recover():
    """
    충돌 감지 시 복구하고 True 반환
    """
    while True:
        violation_code = indy.get_violation_data()["violation_str"]
        if violation_code != '':  # 충돌 감지
            indy.stop_motion(StopCategory.CAT2) # 모션 멈추는 코드
            indy.recover() # 회복하는 코드
            move_home # 회복 후 호밍
            indy.movel([450,150,460,0.1,179.9,0.1]) # 그후 지정 위치
            wait_idle()
            indy.stop_motion(StopCategory.CAT2)
            return True  # 충돌이 발생했음을 알림
        
        motion_status = indy.get_motion_data()["is_target_reached"]
        if motion_status:  # 이동 완료
            return False  # 정상 완료
        # time.sleep(0.1)
##############################################################################




# --- MCP 서버 시작 ---
if __name__ == "__main__":
    # print("MCP 서버 루프 시작...")
    indy.execute_tool('grip_open') # 그리퍼 오픈하고
    try:
        # current_pose = robot_get_tcp_pose()
        # if abs(current_pose[0] - 0.45) < 0.1 and abs(current_pose[1] - 0.15) < 0.1 and abs(current_pose[2] - 0.46) < 0.1:
        #     indy.movel([450,150,500,0.1,179.9,0.1])
        #     # print("현재 위치가 툴의 초기 위치와 일치합니다.")
        #     pass
        # else:
        #     move_home() # 먼저 홈 위치로
        #     indy.movel([450,150,500,0.1,179.9,0.1])
        
        # wait_idle()
        mcp.run()
    except KeyboardInterrupt:
        # print("\n사용자에 의해 서버 중지됨 (Ctrl+C).")
        pass
    except Exception as e:
        # print(f"\nMCP 서버 오류 발생: {e}")
        pass



# --- 디버그용---
# if __name__ == "__main__":
#     # for i in range (6):
#         robot_move(x = 0.45,y = 0.15,z = 0.45,w =0)
#         # indy.movel([450,150,450,0.1,179.9,0.1])
#         # indy.stop_motion(StopCategory.CAT2) 
#         # indy.movel([450,50,450,0.1,179.9,0.1])
#         robot_move(x = 0.35,y = 0.25,z = 0.45,w =0)
#         robot_move(x = 0.45,y = 0.15,z = 0.45,w =0)
#         robot_move(x = 0.35,y = 0.15,z = 0.45,w =0)
#         robot_move(x = 0.45,y = 0.15,z = 0.45,w =0)
#         robot_move(x = 0.35,y = 0.15,z = 0.45,w =0)
#         robot_move(x = 0.45,y = 0.15,z = 0.45,w =0)
#         robot_move(x = 0.35,y = 0.15,z = 0.45,w =0)

        # a = monitoring_and_recover()
        # print(a)
        # if a:
        #     indy.move_home()
        #     print("got home")
        # indy.movel([450,150,450,0.1,179.9,0.1])
        # indy.movel([450,50,450,0.1,179.9,0.1])
        # indy.movel([450,150,450,10,150,0])
        # print(indy.get_violation_data())
        # indy.move_home()


# if __name__ == "__main__":
#     robot_prepare_assemble()

















