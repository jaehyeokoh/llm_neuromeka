from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
import json
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
import base64
import open3d as o3d



############################## 상태 정의 (state_schema) ##########################
# 상태 정의 (state_schema)
class AgentState(TypedDict):
    user_input: str # 사용자 입력을 저장하는 필드
    divided_user_input:  Optional[List[str]] = [] # 나눠진 사용자 입력 저장 필드
    all_objects_name: List[str] = []  # 기본값을 빈 리스트로 설정

initial_state = {
}



async def grasp_agent_without_llm(data,detect_res, tool_node):
    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 3
    attempt = 0
    try:
        print(f"[anygrasp Agent] Tool call attempt {attempt + 1}/{max_retries}")

        grasp_msg = anygrasp_message(data)
        # print("message is ", detect_msg)
        grasp_res = await tool_node.ainvoke([grasp_msg])
        # print("grasp res is : ",grasp_res)

        # List[ToolMessage] 반환 가정
        first_msg = grasp_res[0]
        raw = first_msg.content or ""
        raw = json.loads(raw)
        

        def deserialize_mesh(data):
            """
            그리퍼의 파지점 형태를 랜더링 하는 코드
            """

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(np.array(data["vertices"]))
            mesh.triangles = o3d.utility.Vector3iVector(np.array(data["triangles"]))
            mesh.compute_vertex_normals()
            return mesh

        def deserialize_pointcloud(data):
            """
            포인트 클라우드 데이터를 받아 np로 바꿔 인식하게 만들어 시각화 하게 하는 코드
            """
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(np.array(data["points"], dtype=np.float64))
            colors_array = np.array(data["colors"], dtype=np.float64)
            colors_array = colors_array[:, [2, 1, 0]]  # BGR → RGB

            pc.colors = o3d.utility.Vector3dVector(colors_array)
            return pc
        

        data = raw["result_data"]
        data = json.loads(data)

        gg = raw["gg"]
        print("gg is ", gg[0:5])
        target_x = detect_res["position"][0] # 타겟 중심 위치
        target_y = detect_res["position"][1]

        radius = detect_res["dimensions"][0]/2 # 타겟의 장축

        # anygrasp의 출력을 로봇 좌표계로 바꾸는거
        for g in gg:
            x, y, z = g["translation"]
            g["translation"] = [-y, -x, z]

        # 타겟 물체 중심값
        x0, y0 = target_x, target_y
        radius +=0.01  # 타겟 장축/2 + 1cm = 필터링 영역

        # 중심값으로 부터 장축 이내의 것들만 필터링
        filtered = [
            g for g in gg
            if ((g["translation"][0] - x0) ** 2 + (g["translation"][1] - y0) ** 2) ** 0.5 <= radius
        ]

        print("filtered is ", filtered[:3])
        # 거리 기준 정렬 (타겟으로부터 가까운 것부터)
        filtered_sorted = sorted(
            filtered,
            key=lambda g: ((g["translation"][0] - x0) ** 2 + (g["translation"][1] - y0) ** 2) ** 0.5
        )

        print("filtered sort is ", filtered_sorted[:3])

        from scipy.spatial.transform import Rotation as R

        # 회전 행렬 정의
        rot_mat = np.array(filtered_sorted[0]["rotation"])

        # 회전 객체 생성 및 RPY 추출 (ZYX 순서: yaw, pitch, roll)
        r = R.from_matrix(rot_mat)
        # rpy_rad = r.as_euler('zyx', degrees=False)  # radians
        rpy_deg = r.as_euler('zyx', degrees=True)   # degrees

        # print("RPY (rad):", rpy_rad)
        print("RPY (deg):", rpy_deg)

        # print(data["cloud"],"cloud")
        # print(type(data["cloud"]))
        # cloud = data["cloud"]
        # if isinstance(cloud, dict):
        #     print("vertices" in cloud)
        #     print("triangles" in cloud)
        #     # print(type(cloud["vertices"]))
        # else:
        #     print("cloud is not a dict. It's a", type(cloud))
        # print("CLOUD TYPE:", type(data["cloud"]))
        # print("CLOUD KEYS:", data["cloud"].keys())
        for k, v in data["cloud"].items():
            print(f"Key: {k} | Type: {type(v)} | Sample: {str(v)[:100]}")


        # 예시
        grippers_data = [json.loads(g) if isinstance(g, str) else g for g in data["grippers"]]
        grippers_meshes = [deserialize_mesh(g) for g in grippers_data]
        cloud_pc = deserialize_pointcloud(data["cloud"])

        o3d.visualization.draw_geometries([*grippers_meshes, cloud_pc])


    except Exception as e:
        print("error",e)
    
    return 


def decode_point_cloud(data: dict):
    points = np.frombuffer(base64.b64decode(data["points"]), dtype=np.float32).reshape(-1, 3)
    colors = np.frombuffer(base64.b64decode(data["colors"]), dtype=np.float32).reshape(-1, 3)
    return points, colors

async def detector_agent_without_llm(state: AgentState, tool_node) -> Dict[str, Any]:
    """ all_object_name 또는 detect_failed_objects를 기반으로 위치 + 크기 + 면적 + 각도를 탐지
        같은 이름의 객체가 여러개 감지되는 경우를 대비해 객체별로 id를 부여함
         ex) white box가 2개면 white_box_1, white_box_2 이런식으로 """
    cleaned_names = input("input:")

    # # 로봇팔 엔드툴 포즈
    # tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    # tool_pos = tool_pos[0].content
    # tool_pos = list(map(float, eval(tool_pos)))
    # print("robot_arm_attached",tool_pos)


    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 3
    attempt = 0
    raw = ""
    try:
        print(f"[Detector Agent] Tool call attempt {attempt + 1}/{max_retries}")
        detect_msg = find_object_3d_properties_message(cleaned_names)

        detect_res = await tool_node.ainvoke([detect_msg])
        # print("detect res is : ",detect_res)

        detect_res = detect_res[0]
        detect_res = detect_res.content or ""
        try:
            detect_res = json.loads(detect_res)
        except Exception as e:
            print(e)
        print("detect_res is :", detect_res)

        # 만약 객체 여러개 감지되면 일단 첫번째만 받음
        detect_res = detect_res[0]
        
        try:
            detect_res = json.loads(detect_res)
        except Exception as e:
            print(e)

        print("first detect_res is ", detect_res)
        obj_height = detect_res["position"][2]
        print("height is ",obj_height)
        mask_b64 = detect_res["mask_base64"]
        mask_shape = tuple(detect_res["mask_shape"])

        cloud_msg = generate_point_cloud_message(mask_b64, mask_shape, obj_height)
        # print("message is ", cloud_msg)
        cloud_res = await tool_node.ainvoke([cloud_msg])
        # List[ToolMessage] 반환 가정
        cloud_res = cloud_res[0]
        cloud_res = cloud_res.content or ""
        cloud_res = json.loads(cloud_res)  # 리스트 안의 문자열들
        points, colors = decode_point_cloud(cloud_res)
        print("p,c",points[:2],colors[:2])


    except Exception as e:
        print("error",e)
    return await grasp_agent_without_llm(cloud_res,detect_res,tool_node)

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

