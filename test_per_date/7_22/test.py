from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
import json
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
import base64
import open3d as o3d

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
object_names = [ "white box", "black box", "can" ] # 마지막에 잡을 대상 넣으면 됨.
# object_names = ["can"]

# decode_mask 함수 (기존 서버 코드와 동일하게)
def decode_mask(mask_base64: str, mask_shape: tuple) -> np.ndarray:
    """
    Decode base64-encoded and zlib-compressed binary mask back to numpy array.
    Parameters:
    mask_base64 (str): The base64-encoded mask string.
    mask_shape (tuple): The original (H, W) shape of the mask.
    Returns:
    np.ndarray: Decoded binary mask as a boolean numpy array.
    """
    try:
        import zlib
        # 1. base64 디코딩
        compressed = base64.b64decode(mask_base64)
        # 2. zlib 압축 해제
        decompressed = zlib.decompress(compressed)
        # 3. numpy 배열로 변환 및 bool mask로 변경
        mask_flat = np.frombuffer(decompressed, dtype=np.uint8)
        mask = mask_flat.reshape(mask_shape).astype(bool)
        print(f"✅ Mask decoded successfully: {mask.shape}, True pixels: {np.sum(mask)}")
        return mask
    except Exception as e:
        print(f"❌ Mask decoding failed: {e}")
        return None

def filter_grasps_with_mask_pixels(grasps, detect_res, cloud_data):
    """
    마스크 픽셀 좌표를 직접 사용하여 포인트 클라우드에서 해당 영역만 추출 후 필터링
    """
    if 'mask_base64' not in detect_res or 'mask_shape' not in detect_res:
        return grasps
    
    mask = decode_mask(detect_res['mask_base64'], tuple(detect_res['mask_shape']))
    if mask is None:
        return grasps
    
    # 포인트 클라우드 데이터 (리스트 → numpy 배열)
    points = np.array(cloud_data['points'], dtype=np.float32)
    print(f"📊 Points shape: {points.shape}")
    
    # 마스크가 True인 픽셀 좌표들 찾기
    mask_y, mask_x = np.where(mask)
    print(f"📊 Mask pixels: {len(mask_y)}")
    
    # 마스크 영역에 해당하는 포인트들만 추출
    h, w = mask.shape
    mask_points = []
    
    for i in range(len(mask_y)):
        y, x = mask_y[i], mask_x[i]
        point_idx = y * w + x
        if point_idx < len(points):
            mask_points.append(points[point_idx])
    
    if len(mask_points) == 0:
        print("⚠️ No points found in mask region")
        return grasps
    
    mask_points = np.array(mask_points)
    print(f"📊 Mask points: {len(mask_points)}")
    
    # 마스크 영역의 3D 바운딩 박스 계산
    min_bounds = np.min(mask_points, axis=0)
    max_bounds = np.max(mask_points, axis=0)
    
    print(f"🎯 Mask bounds: X[{min_bounds[0]:.3f}~{max_bounds[0]:.3f}] Y[{min_bounds[1]:.3f}~{max_bounds[1]:.3f}] Z[{min_bounds[2]:.3f}~{max_bounds[2]:.3f}]")
    
    # grasp 필터링
    filtered = []
    for g in grasps:
        pos = g["translation"]
        if (min_bounds[0] <= pos[0] <= max_bounds[0] and
            min_bounds[1] <= pos[1] <= max_bounds[1] and
            min_bounds[2] <= pos[2] <= max_bounds[2]):
            filtered.append(g)
    
    print(f"✅ Mask filtering: {len(grasps)} -> {len(filtered)} grasps")
    return filtered


async def grasp_agent_without_llm(data,detect_res,target_side_pts, tool_node):
    """입력
    data = point cloud
    detect_res = GroundedSAM2 실행 결과
    target_side_pts = 타겟의 side의 포인트 클라우드
    """

    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 3
    attempt = 0
    try:
        print("target side pts cloud is ", target_side_pts[:200])
        print(f"[anygrasp Agent] Tool call attempt {attempt + 1}/{max_retries}")
        mask_b64 = detect_res["mask_base64"]
        mask_shape = tuple(detect_res["mask_shape"])
        grasp_msg = anygrasp_message(data, mask_b64, mask_shape, target_side_pts)
        # print("message grasp_msg is ", grasp_msg)
        grasp_res = await tool_node.ainvoke([grasp_msg])
        print("grasp res is : ",grasp_res[0].content[:2000])

        # List[ToolMessage] 반환 가정
        first_msg = grasp_res[0]
        raw = first_msg.content or ""
        raw = json.loads(raw)
        

################################################ 시각적 디버그용 코드#####################
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
        # for i, g in enumerate(grippers_data):
        #     vertices = np.array(g["vertices"], dtype=np.float64)
        #     # X축 반전과 일치시키려면 -x
        #     x_coords = -vertices[:, 0]
        #     y_coords = vertices[:, 1]
            
        #     print(f"Gripper {i} - X range: {x_coords.min()} ~ {x_coords.max()}, Y range: {y_coords.min()} ~ {y_coords.max()}")


        cloud_pc = deserialize_pointcloud(data["cloud"])

        o3d.visualization.draw_geometries([*grippers_meshes, cloud_pc])

        gg = raw["gg"]
        print("gg is ", gg[0:5])
        target_x =  - detect_res["position"][1] # 타겟 중심 위치
        target_y =  - detect_res["position"][0]

        radius = detect_res["dimensions"][0]/2 # 타겟의 장축

        target_name = detect_res["object_name"]
        # (1) 중심값 계산
        grippers_centroids = []
        for g in grippers_data:
            vertices = np.array(g["vertices"], dtype=np.float64)
            centroid = vertices.mean(axis=0)
            grippers_centroids.append(centroid)

        # (2) 중심 위치로 필터링
        # 중심 위치 설정
        x0, y0 = target_x, target_y
        if radius >= 0.2:
            radius -= 0.01
        if radius <= 0.04:
            radius += 0.02
        print(f"🎯 Target name = {target_name}, Position: x0 = {x0:.4f}, y0 = {y0:.4f}, radius = {radius:.4f}")

        # 필터링 + 위치 출력
        filtered_indices = []
        for i, c in enumerate(grippers_centroids):
            dist = ((c[0] - x0) ** 2 + (c[1] - y0) ** 2) ** 0.5
            if dist <= radius:
                print(f"✅ Gripper {i}: centroid = ({c[0]:.4f}, {c[1]:.4f}), distance = {dist:.4f}")
                filtered_indices.append(i)
            else:
                print(f"❌ Gripper {i}: centroid = ({c[0]:.4f}, {c[1]:.4f}), distance = {dist:.4f} (outside radius)")
                continue



        # (3) 시각화
        filtered_meshes = [grippers_meshes[i] for i in filtered_indices]
        o3d.visualization.draw_geometries([*filtered_meshes, cloud_pc])
####################################################################################################


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

        # 기존 필터링 코드를 이것으로 교체:
        if 'mask_base64' in detect_res and 'mask_shape' in detect_res:
            print("🎯 Using pixel-based mask filtering")
            print(f"📊 Mask info: shape={detect_res['mask_shape']}")
            
            # 마스크로 필터링 시도
            filtered = filter_grasps_with_mask_pixels(gg, detect_res, data["cloud"])
            
            if len(filtered) == 0:
                print("⚠️ No grasps in mask area, using circular filtering")
                filtered = [
                    g for g in gg
                    if ((g["translation"][0] - x0) ** 2 + (g["translation"][1] - y0) ** 2) ** 0.5 <= radius
                ]
                print(f"✅ Circular fallback: {len(gg)} -> {len(filtered)} grasps")
            
            # 마스크 필터링이 너무 적게 나왔다면 원형 필터링과 섞기
            if len(filtered) > 0 and len(filtered) < 5:
                print("⚠️ Too few mask results, expanding with circular filter")
                circular_filtered = [
                    g for g in gg
                    if ((g["translation"][0] - x0) ** 2 + (g["translation"][1] - y0) ** 2) ** 0.5 <= radius * 1.2
                ]
                # 마스크 결과와 원형 결과 합치기 (중복 제거)
                mask_positions = {tuple(g["translation"]) for g in filtered}
                for g in circular_filtered:
                    if tuple(g["translation"]) not in mask_positions:
                        filtered.append(g)
                print(f"✅ Combined filtering: {len(filtered)} grasps")
        else:
            print("🎯 Using circular filtering")
            filtered = [
                g for g in gg
                if ((g["translation"][0] - x0) ** 2 + (g["translation"][1] - y0) ** 2) ** 0.5 <= radius
            ]

        print(f"✅ Final filtering result: {len(filtered)} grasps")

        # 거리 기준 정렬 (타겟으로부터 가까운 것부터)
        filtered_sorted = sorted(
            filtered,
            key=lambda g: ((g["translation"][0] - x0) ** 2 + (g["translation"][1] - y0) ** 2) ** 0.5
        )

        print("filtered sort is ", filtered_sorted[:3])

        from scipy.spatial.transform import Rotation as R

        # 회전 행렬 정의
        # rpy = np.array(filtered_sorted[0]["rpy"])
        rpy = np.array(filtered[0]["rpy"])

        print("Converted to robot:", rpy)
           
           
        try:
            # # 로봇팔 엔드툴 포즈
            tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
            tool_pos = tool_pos[0].content
            tool_pos = list(map(float, eval(tool_pos)))
            print("robot_arm_attached",tool_pos)
            xyz = filtered_sorted[0]["translation"]
            print("xyz", xyz)
            x,y,w = tool_to_world(tool_pos, xyz[0],xyz[1], 0) # 월드 좌표계로 전환
            print("x,y, rpy", x,y, rpy)
            print("rpy is ", rpy[0], rpy[1], rpy[2])


            pos1 = [x,y,detect_res["position"][2],rpy[0],rpy[1],rpy[2]]
            print("pos1 is ", pos1)
            msg1 = moveL_rpy_message(pos1)
            movel_res1 = await tool_node.ainvoke([msg1])
            print("movel pos1:", movel_res1)
        except Exception as e:
            print("robot error is ", e)




        for k, v in data["cloud"].items():
            print(f"Key: {k} | Type: {type(v)} | Sample: {str(v)[:100]}")


##################################################################필터링

        def is_pitch_roll_within_limit(R_matrix, limit_deg=85):
            """ 절대좌표계 기준 pitch/roll 제한 (ZYX) """
            r = R.from_matrix(R_matrix)
            yaw, pitch, roll = r.as_euler('zyx', degrees=True)
            return abs(pitch) <= limit_deg and abs(roll) <= limit_deg

        def convert_to_tool_rpy(R_matrix):
            """ 툴 좌표계 기준 RPY (XYZ) """
            r = R.from_matrix(R_matrix)
            roll, pitch, yaw = r.as_euler('xyz', degrees=True)
            return roll, pitch, yaw

        def convert_to_global_rpy(R_matrix):
            """ 절대 좌표계 기준 RPY (ZYX) """
            r = R.from_matrix(R_matrix)
            yaw, pitch, roll = r.as_euler('zyx', degrees=True)
            return -roll, pitch, yaw  # 순서를 roll-pitch-yaw로 맞춰서 반환

        def compute_distance(pos):
            return np.linalg.norm(pos)

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
    # cleaned_names = input("input:")




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

        # 최종 출력 확인
        print("[INFO] mask_base64_list:", len(mask_base64_list))
        print("[INFO] mask_shape_list:", mask_shape_list)
        print("[INFO] obj_height_list:", obj_height_list)

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
        cloud_res = json.loads(cloud_res)  # 리스트 안의 문자열들
        points, colors = decode_point_cloud(cloud_res)
        target_side_pts = cloud_res["target_side_pts"]
        print("p,c",points[:2],colors[:2])



    except Exception as e:
        print("error during detect",e)
    
    return await grasp_agent_without_llm(cloud_res,detect_data,target_side_pts,tool_node)

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

