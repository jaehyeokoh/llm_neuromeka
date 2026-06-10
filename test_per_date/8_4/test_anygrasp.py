from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
import json
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
import open3d as o3d
import time

# import logging

# # ✅ 루트 로거에 달려 있는 기존 핸들러 모두 제거
# logging.getLogger().handlers.clear()

# # ✅ 전파 방지 (중복 로깅 방지)
# logging.getLogger().propagate = False

# # ✅ 기본 로그 설정 복구 (필요에 따라 파일 또는 콘솔 설정)
# logging.basicConfig(level=logging.WARNING)  # WARNING 이상 로그만 출력


############################## 상태 정의 (state_schema) ##########################
# 상태 정의 (state_schema)
class AgentState(TypedDict):
    user_input: str # 사용자 입력을 저장하는 필드
    divided_user_input:  Optional[List[str]] = [] # 나눠진 사용자 입력 저장 필드
    all_objects_name: List[str] = []  # 기본값을 빈 리스트로 설정

initial_state = {
}
# object_names = [ "white box", "black box", "can" ] # 마지막에 잡을 대상 넣으면 됨.
object_names = ["mouse"]
target_name = "mouse"  # 네가 잡으려는 물체

if target_name in object_names:
    object_names.remove(target_name)
    object_names.append(target_name)

print("타겟 물건 이름",target_name)

debug = True


async def grasp_agent_without_llm(data,detect_res,target_boundary_info, tool_node, debug = debug):
    """입력
    data = point cloud
    detect_res = GroundedSAM2 실행 결과
    target_boundary_info = 타겟의 target_boundary_info(윤곽선, 최대 최소 높이 -> anygrasp에서 사용하려고 만듬)
    debug = 시각화 디버그
    """

    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 3
    attempt = 0
    try:
        print("target side pts cloud is ", target_boundary_info[:200])
        print(f"[anygrasp Agent] Tool call attempt {attempt + 1}/{max_retries}")
        # mask_b64 = detect_res["mask_base64"]
        # mask_shape = tuple(detect_res["mask_shape"])
        grasp_msg = anygrasp_message(data,target_boundary_info, debug)
        # print("message grasp_msg is ", grasp_msg)
        grasp_res = await tool_node.ainvoke([grasp_msg])
        print("grasp res is : ",grasp_res[0].content[:2000])

        # List[ToolMessage] 반환 가정
        first_msg = grasp_res[0]
        raw = first_msg.content or ""
        raw = json.loads(raw)
        
################################################ 시각적 디버그용 코드#####################
        if debug:
            def deserialize_mesh(data):
                """
                그리퍼의 파지점 형태를 랜더링 하는 코드
                """

                vertices_array = np.array(data["vertices"], dtype=np.float64)
                vertices_array[:, 0] *= -1  # X축 반전으로 좌우 대칭 해결

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices_array)
                mesh.triangles = o3d.utility.Vector3iVector(np.array(data["triangles"], dtype=np.int32))
                mesh.compute_vertex_normals()
                return mesh

            def deserialize_pointcloud(data):
                """
                포인트 클라우드 데이터를 받아 np로 바꿔 인식하게 만들어 시각화 하게 하는 코드
                """
                points_array = np.array(data["points"], dtype=np.float64)
                points_array[:, 0] *= -1  # X축 반전으로 좌우 대칭 해결

                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(points_array)

                colors_array = np.array(data["colors"], dtype=np.float64)
                colors_array = colors_array[:, [2, 1, 0]]  # BGR → RGB

                pc.colors = o3d.utility.Vector3dVector(colors_array)
                return pc

            data = raw["result_data"]
            data = json.loads(data)
            # 예시
            grippers_data = [json.loads(g) if isinstance(g, str) else g for g in data["grippers"]]
            grippers_meshes = [deserialize_mesh(g) for g in grippers_data]
            

            cloud_pc = deserialize_pointcloud(data["cloud"])

            o3d.visualization.draw_geometries([*grippers_meshes, cloud_pc])
            grippers_meshes2 = [deserialize_mesh(grippers_data[0])]
            o3d.visualization.draw_geometries([*grippers_meshes2, cloud_pc])
####################################################################################################


        gg = raw["gg"]
        print("gg is ", gg[0:5])
        target_x = detect_res["position"][0] # 타겟 중심 위치
        target_y = detect_res["position"][1]

        # anygrasp의 출력을 로봇 좌표계로 바꾸는거
        for g in gg:
            x, y, z = g["translation"]
            g["translation"] = [-y, -x, z]


        from scipy.spatial.transform import Rotation as R
           
           
        try:
            # # 로봇팔 엔드툴 포즈
            tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
            tool_pos = tool_pos[0].content
            tool_pos = list(map(float, eval(tool_pos)))
            print("robot_arm_attached",tool_pos)

            x,y,w = tool_to_world(tool_pos,target_x,target_y, 0) # 월드 좌표계로 전환 , x,y 위치는 GroundedSAM2로 구했음 (캔같은 원통형은 이거로 하고 박스는 위에 주석으로 할것)
            print("x,y in sam", x,y)

            if detect_res["position"][2] >= 0.12:
                grasp_height = detect_res["position"][2] - 0.03
            else:
                grasp_height = detect_res["position"][2]

            g = None
            pos1 = None  
            approach_pos = None  

            for i in gg:
                rpy_deg = np.array(i["rpy"])
                rpy_rad = np.radians(rpy_deg)
                rot_matrix = R.from_euler('xyz', rpy_rad).as_matrix()
                tool_z = rot_matrix[:, 2]
                # xyz = i["translation"] # 이건 graspnet으로 나온 거임
                # x, y, w = tool_to_world(tool_pos, xyz[0], xyz[1], 0)
                pos1 = [x, y, grasp_height, *rpy_deg]
                offset = -0.08 * tool_z
                approach_pos = [pos1[0] + offset[0], pos1[1] + offset[1], pos1[2] + offset[2], *rpy_deg]
                
                check_msg = check_IK_message(pos1)
                ik_res = await tool_node.ainvoke([check_msg])
                check_msg2 = check_IK_message(approach_pos)
                ik_res2 = await tool_node.ainvoke([check_msg2])
                
                if ik_res[0].content == "true" and ik_res2[0].content == "true":
                    g = i
                    print("ik check success!!", ik_res)
                    break
                else:
                    print("ik failed: pos1 is: ", pos1, "result:", ik_res)
            print("Target pos1 is", pos1)



            # 만약 접근 위치의 z가 10 이하면 +10cm 해줌
            if pos1[2] + offset[2] <= 0.1:
                height_offset = 0.1
            else:
                height_offset = 0

            # 뒤로 0.1m (즉, -Z 방향으로 0.1m)
            offset = -0.1 * tool_z

            # 접근 위치 1: 위로 20cm 상승한 뒤 offset 적용
            approach_pos1 = [
                pos1[0] + offset[0],
                pos1[1] + offset[1],
                pos1[2] + offset[2] + height_offset,
                pos1[3], pos1[4], pos1[5]
            ]
            resmove1 = await tool_node.ainvoke([moveL_rpy_message(approach_pos1)])
            print("resmove1 is", resmove1)

            tilt_from_xy = 90 - abs(np.degrees(np.arccos(tool_z[2])))  # 수평면으로부터의 기울기 구하는 코드, z값 클수록 수직
            print(f"수평면 기준 기울기: {tilt_from_xy:.2f}°")

            if abs(tilt_from_xy)<=58: # 기울기가 수폄면으로부터 x 도 이하라면 그리퍼 재정렬

                # 그리퍼 정렬 (새 RPY)
                rpy_raw = await tool_node.ainvoke([gripper_rotate_message()])
                new_rpy = [float(v) for v in json.loads(rpy_raw[0].content)]

            else:
                new_rpy = [pos1[3], pos1[4], pos1[5]]

            # 접근 위치 2: 정렬된 RPY로 offset 적용
            approach_pos2 = [
                pos1[0] + offset[0],
                pos1[1] + offset[1],
                pos1[2] + offset[2],
                *new_rpy
            ]
            resmove2 = await tool_node.ainvoke([moveL_rpy_message(approach_pos2)])
            print("resmove2 is", resmove2)

            # 최종 목표 위치 이동 (정렬된 RPY 적용)
            target_pos = [pos1[0], pos1[1], pos1[2], *new_rpy]
            await tool_node.ainvoke([moveL_rpy_message(target_pos)])
            print("movel pos1:", target_pos)


            grip_res = await tool_node.ainvoke([gripper_message("Grab")])
            time.sleep(0.5)
            print("그립 완료:", grip_res)
################################################ 테스트용임 지우삼 ######################################################
            resmove2 = await tool_node.ainvoke([moveL_rpy_message(approach_pos2)])
            print("resmove2 is", resmove2)

            await tool_node.ainvoke([moveL_rpy_message(target_pos)])
            print("movel pos1:", target_pos)

            _ = await tool_node.ainvoke({"messages": [gripper_message("Release")]})
###################################################################################################################
        except Exception as e:
            print("robot error is", e)

    except Exception as e:
        print("error",e)
    
    return 


async def detector_agent_without_llm(state: AgentState, tool_node) -> Dict[str, Any]:
    """ all_object_name 또는 detect_failed_objects를 기반으로 위치 + 크기 + 면적 + 각도를 탐지
        같은 이름의 객체가 여러개 감지되는 경우를 대비해 객체별로 id를 부여함
         ex) white box가 2개면 white_box_1, white_box_2 이런식으로 """

    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 3
    attempt = 0
    raw = ""
    try:
        print(f"[Detector Agent] Tool call attempt {attempt + 1}/{max_retries}")


        cleaned_names = [n.strip().strip('"').strip("'") for n in object_names if n.strip()]

        # 결과 리스트 초기화
        mask_base64_list = []
        mask_shape_list = []
        obj_height_list = []


        # 객체별 마스킹 및 높이 추출 루프
        for name in cleaned_names:
            for attempt in range(3):
                print(f"[Detector Agent] Detecting '{name}' (attempt {attempt + 1}/3)")
                detect_msg = find_object_3d_properties_message([name])
                detect_res = await tool_node.ainvoke([detect_msg])
                detect_res = detect_res[0].content or ""

                try:
                    detect_data = json.loads(detect_res)
                except Exception as e:
                    print(f"[ERROR] Failed to parse detect result for '{name}':", e)
                    continue

                # 여러 객체 감지된 경우 첫 번째만 사용
                if isinstance(detect_data, list):
                    detect_data = detect_data[0]

                # 다시 JSON 디코딩 시도 (이중 JSON일 경우)
                if isinstance(detect_data, str):
                    try:
                        detect_data = json.loads(detect_data)
                    except Exception as e:
                        print(f"[ERROR] Second decode failed for '{name}':", e)
                        continue

                # 마스크, 높이, 모양 추출
                try:
                    mask_b64 = detect_data["mask_base64"]
                    mask_shape = tuple(detect_data["mask_shape"])
                    obj_height = detect_data["position"][2]
                except KeyError as e:
                    print(f"[ERROR] Missing key in result for '{name}':", e)
                    continue

                # 성공한 경우에만 append
                mask_base64_list.append(mask_b64)
                mask_shape_list.append(list(mask_shape))  # 리스트로 변환
                obj_height_list.append(obj_height)
                break  # 성공했으면 retry 루프 탈출

        # 메시지 생성
        cloud_msg = generate_point_cloud_message(
            mask_base64_list,
            mask_shape_list,
            obj_height_list
        )
        cloud_res = await tool_node.ainvoke([cloud_msg])
        # List[ToolMessage] 반환 가정
        cloud_res = cloud_res[0]
        cloud_res = cloud_res.content or ""
        print("cloud res is ",cloud_res[:100])
        try:
            cloud_res = json.loads(cloud_res)  # 리스트 안의 문자열들
            target_boundary_info = cloud_res["target_boundary_info"]

        except Exception as e:
            print("파싱중 오류남",e)


    except Exception as e:
        print("error during detect",e)
    
    return await grasp_agent_without_llm(cloud_res,detect_data,target_boundary_info,tool_node)

# Graph 생성
async def create_graph( tools):
    graph = StateGraph(state_schema=AgentState)  # 상태 스키마 전달
    tool_node = ToolNode(tools)  # 도구 노드 생성
    
    # 비동기 함수로 추가
    graph.add_node("detectorAgent", wrap_async(detector_agent_without_llm, tool_node=tool_node))
    graph.add_node("anygraspAgent", wrap_async(grasp_agent_without_llm, tool_node=tool_node))
    graph.set_entry_point("detectorAgent")  # 시작 지점을 설정
    
    # 비동기적으로 그래프를 컴파일할 수 있도록 확인
    compiled_graph = graph.compile()  # 비동기적으로 컴파일
    return compiled_graph


async def main():
    mcp_server_configs = create_server_config()  # MCP 서버 설정 불러오기

    print("--- MCP Client Initializing ---")
    try:
        async with MultiServerMCPClient(mcp_server_configs) as client: # mcp 서버에 연결 및 실행상태 유지
            print("--- MCP Client Initialized ---")
            tools = client.get_tools() # MCP 서버에서 도구 가져오기
            compiled_graph = await create_graph(tools)  # 그래프 비동기적으로 생성

            # 그래프 처리
            async for event in compiled_graph.astream(initial_state):
                for node, output in event.items():
                    print(f"[{node}] output: {output}")
    except Exception as e:
        print(f"Error during MCP Client initialization: {e}")

 
if __name__ == "__main__":
    try:
        asyncio.run(main())  # 비동기 메인 함수 실행
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nCritical error: {e}")
        import traceback
        traceback.print_exc()

