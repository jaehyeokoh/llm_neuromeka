from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
import json
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
import open3d as o3d
import time
from AI_messages import *
from parsing_util import *
# import logging

# # ✅ 루트 로거에 달려 있는 기존 핸들러 모두 제거
# logging.getLogger().handlers.clear()

# # ✅ 전파 방지 (중복 로깅 방지)
# logging.getLogger().propagate = False

# # ✅ 기본 로그 설정 복구 (필요에 따라 파일 또는 콘솔 설정)
# logging.basicConfig(level=logging.WARNING)  # WARNING 이상 로그만 출력

import json
import pickle
from datetime import datetime


from ast import literal_eval


async def _get_tcp_pose(tool_node) -> list[float]:
    msg = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    return list(map(float, literal_eval(msg[0].content)))



def debug_save_to_file(result_list, filename=None):
    """출력을 파일로 저장해서 확인"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_output_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== 함수 1 출력 디버그 ===\n")
        f.write(f"전체 타입: {type(result_list)}\n")
        f.write(f"전체 길이: {len(result_list) if hasattr(result_list, '__len__') else 'N/A'}\n\n")
        
        if isinstance(result_list, list):
            for i, item in enumerate(result_list):
                f.write(f"--- item[{i}] ---\n")
                f.write(f"타입: {type(item)}\n")
                if isinstance(item, str):
                    f.write(f"길이: {len(item)}\n")
                    f.write(f"처음 100자: {item[:100]}...\n")
                    f.write(f"마지막 100자: ...{item[-100:]}\n")
                else:
                    f.write(f"내용: {item}\n")
                f.write("\n")
        else:
            f.write(f"전체 내용: {str(result_list)[:1000]}...\n")
    
    print(f"디버그 정보가 {filename}에 저장되었습니다.")
    return filename

def debug_structure_only(result_list):
    """구조만 간단히 출력"""
    print("=== 구조 요약 ===")
    print(f"전체 타입: {type(result_list)}")
    
    if isinstance(result_list, list):
        print(f"리스트 길이: {len(result_list)}")
        for i, item in enumerate(result_list[:3]):  # 처음 3개만
            print(f"item[{i}] 타입: {type(item)}")
            if isinstance(item, dict):
                print(f"  키들: {list(item.keys())}")
            elif isinstance(item, str):
                print(f"  문자열 길이: {len(item)}")
                print(f"  시작: {item[:50]}...")
        
        if len(result_list) > 3:
            print(f"... (총 {len(result_list)}개 항목)")
    else:
        print(f"타입: {type(result_list)}")

def debug_json_safe(result_list, filename=None):
    """JSON으로 저장 (base64 데이터는 길이만 표시)"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_structure_{timestamp}.json"
    
    def make_json_safe(obj):
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if isinstance(v, str) and len(v) > 100:
                    result[k] = f"<문자열 길이: {len(v)}>"
                else:
                    result[k] = make_json_safe(v)
            return result
        elif isinstance(obj, list):
            return [make_json_safe(item) for item in obj]
        else:
            return obj
    
    safe_data = make_json_safe(result_list)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(safe_data, f, indent=2, ensure_ascii=False)
    
    print(f"JSON 구조가 {filename}에 저장되었습니다.")
    return filename
############################## 상태 정의 (state_schema) ##########################
# 상태 정의 (state_schema)
class AgentState(TypedDict):
    user_input: str # 사용자 입력을 저장하는 필드
    divided_user_input:  Optional[List[str]] = [] # 나눠진 사용자 입력 저장 필드
    all_objects_name: List[str] = []  # 기본값을 빈 리스트로 설정

initial_state = {
}
object_names = [ "white box", "black box", "can" ] # 마지막에 잡을 대상 넣으면 됨.
# object_names = ["mouse"]
target_name = "mouse"  # 네가 잡으려는 물체

if target_name in object_names:
    object_names.remove(target_name)
    object_names.append(target_name)

print("타겟 물건 이름",target_name)

debug = True
import numpy as np
import open3d as o3d
import base64


def debug_visualize_function2_output(result_dict):
    """
    함수 2의 출력을 시각화
    
    Args:
        result_dict: 함수 2의 출력 딕셔너리
    """
    
    print("=" * 50)
    print("함수 2 결과 디버그")
    print("=" * 50)
    result_dict = json.loads(result_dict)

    
    # 포인트 클라우드 디코딩 및 시각화
    try:
        # base64 디코딩
        points_bytes = base64.b64decode(result_dict["points"])
        colors_bytes = base64.b64decode(result_dict["colors"])
        
        # numpy 배열로 변환
        points_flat = np.frombuffer(points_bytes, dtype=np.float32)
        colors_flat = np.frombuffer(colors_bytes, dtype=np.float32)
        
        points_array = points_flat.reshape(-1, 3).astype(np.float64)
        colors_array = colors_flat.reshape(-1, 3).astype(np.float64)
        
        # X축 반전 (참조 코드와 동일)
        points_array[:, 0] *= -1
        
        # BGR → RGB 변환 (참조 코드와 동일)
        colors_array = colors_array[:, [2, 1, 0]]
        
        # Open3D 포인트 클라우드 생성
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_array)
        pc.colors = o3d.utility.Vector3dVector(colors_array)
        
        # 포인트 범위 정보
        print(f"\n📍 포인트 클라우드 범위:")
        x_range = f"X: {points_array[:, 0].min():.3f} ~ {points_array[:, 0].max():.3f}"
        y_range = f"Y: {points_array[:, 1].min():.3f} ~ {points_array[:, 1].max():.3f}"
        z_range = f"Z: {points_array[:, 2].min():.3f} ~ {points_array[:, 2].max():.3f}"
        print(f"  {x_range}")
        print(f"  {y_range}")
        print(f"  {z_range}")
        
        # 색상 분포 분석
        print(f"\n🎨 색상 분포 분석:")
        
        # 빨간색 계열 (저장된 사이드 포인트들) - R > 0.6, G < 0.4, B < 0.4
        red_mask = (colors_array[:, 0] > 0.6) & (colors_array[:, 1] < 0.4) & (colors_array[:, 2] < 0.4)
        red_count = np.sum(red_mask)
        
        # 초록색 계열 (새 타겟 사이드 포인트들) - G > 0.6, R < 0.4, B < 0.4  
        green_mask = (colors_array[:, 1] > 0.6) & (colors_array[:, 0] < 0.4) & (colors_array[:, 2] < 0.4)
        green_count = np.sum(green_mask)
        
        # 나머지 (환경 포인트들)
        other_count = len(colors_array) - red_count - green_count
        
        print(f"  🔴 빨간색 (저장된 사이드): {red_count:,}개 ({red_count/len(colors_array)*100:.1f}%)")
        print(f"  🟢 초록색 (새 타겟 사이드): {green_count:,}개 ({green_count/len(colors_array)*100:.1f}%)")
        print(f"  🌫️ 기타 (환경): {other_count:,}개 ({other_count/len(colors_array)*100:.1f}%)")
        
        # 시각화
        print(f"\n🔍 시각화 시작...")
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pc, coord_frame], 
                                        window_name="Integrated Point Cloud (Function 2)")
        
        print("✅ 시각화 완료!")
        
    except Exception as e:
        print(f"❌ 디코딩/시각화 실패: {str(e)}")

def debug_compare_before_after_downsampling(result_dict, original_points_before_ds=None):
    """
    다운샘플링 전후 비교 (선택적)
    
    Args:
        result_dict: 함수 2 출력
        original_points_before_ds: 다운샘플링 전 포인트들 (있다면)
    """
    
    if original_points_before_ds is None:
        print("다운샘플링 전 데이터가 없어서 비교를 건너뜁니다.")
        return
    
    print("\n" + "=" * 30)
    print("다운샘플링 전후 비교")
    print("=" * 30)
    
    try:
        # 다운샘플링 후 포인트
        points_bytes = base64.b64decode(result_dict["points"])
        points_after = np.frombuffer(points_bytes, dtype=np.float32).reshape(-1, 3)
        
        print(f"다운샘플링 전: {len(original_points_before_ds):,}개")
        print(f"다운샘플링 후: {len(points_after):,}개") 
        reduction = (1 - len(points_after) / len(original_points_before_ds)) * 100
        print(f"감소율: {reduction:.1f}%")
        
        # 밀도 비교
        before_vol = np.prod(np.ptp(original_points_before_ds, axis=0))
        after_vol = np.prod(np.ptp(points_after, axis=0))
        
        if before_vol > 0 and after_vol > 0:
            density_before = len(original_points_before_ds) / before_vol
            density_after = len(points_after) / after_vol
            print(f"밀도 변화: {density_before:.0f} → {density_after:.0f} points/m³")
        
    except Exception as e:
        print(f"비교 분석 실패: {e}")


def debug_visualize_function1_output(result_str):
    """
    함수 1의 출력을 받아서 디코딩 후 시각화
    
    Args:
        result_str: 함수 1의 출력 (JSON 문자열)
    """
    
    try:
        # 1차 JSON 파싱 (외부 리스트)
        result_list = json.loads(result_str)
        print(f"1차 JSON 파싱 성공: {len(result_list)}개 항목")
        
        # 2차 JSON 파싱 (내부 딕셔너리들)
        parsed_objects = []
        for item in result_list:
            if isinstance(item, str):
                # 문자열인 경우 다시 JSON 파싱
                parsed_item = json.loads(item)
                parsed_objects.append(parsed_item)
                # print(f"2차 파싱: {parsed_item}")
            else:
                # 이미 딕셔너리인 경우 그대로 사용
                parsed_objects.append(item)
        
        result_list = parsed_objects
        print(f"최종 파싱 완료: {len(result_list)}개 객체")
        
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 실패: {e}")
        return
    
    point_clouds = []
    
    # 각 물체별로 다른 색상
    colors = [
        [1.0, 0.0, 0.0],  # 빨강
        [0.0, 1.0, 0.0],  # 초록  
        [0.0, 0.0, 1.0],  # 파랑
        [1.0, 1.0, 0.0],  # 노랑
        [1.0, 0.0, 1.0],  # 자홍
        [0.0, 1.0, 1.0],  # 청록
        [0.5, 0.5, 0.5],  # 회색
    ]
    
    color_idx = 0
    
    for obj_dict in result_list:
        for obj_id, encoded_data in obj_dict.items():
            if encoded_data is not None:
                try:
                    # base64 디코딩
                    points_bytes = base64.b64decode(encoded_data)
                    
                    # numpy 배열로 변환
                    points_flat = np.frombuffer(points_bytes, dtype=np.float32)
                    points_array = points_flat.reshape(-1, 3).astype(np.float64)
                    
                    # X축 반전 (참조 코드와 동일)
                    points_array[:, 0] *= -1
                    
                    # Open3D 포인트 클라우드 생성
                    pc = o3d.geometry.PointCloud()
                    pc.points = o3d.utility.Vector3dVector(points_array)
                    
                    # 색상 적용
                    color = colors[color_idx % len(colors)]
                    pc.paint_uniform_color(color)
                    
                    point_clouds.append(pc)
                    color_idx += 1
                    
                    print(f"✅ {obj_id}: {len(points_array)}개 포인트")
                    
                except Exception as e:
                    print(f"❌ {obj_id}: 디코딩 실패 - {str(e)}")
            else:
                print(f"❌ {obj_id}: 처리 실패 (None)")
    
    # 시각화
    if point_clouds:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([*point_clouds, coord_frame])
        print(f"총 {len(point_clouds)}개 물체의 사이드 포인트 클라우드 시각화 완료")
    else:
        print("시각화할 포인트 클라우드가 없습니다.")

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



        except Exception as e:
            print("robot error is", e)

    except Exception as e:
        print("error",e)
    
    return 


async def detector_agent_without_llm(state: AgentState, tool_node, all_object_detect = False) -> Dict[str, Any]:
    """ all_object_name 또는 detect_failed_objects를 기반으로 위치 + 크기 + 면적 + 각도를 탐지
        같은 이름의 객체가 여러개 감지되는 경우를 대비해 객체별로 id를 부여함
         ex) white box가 2개면 white_box_1, white_box_2 이런식으로 """

    ########################## 피드백 발동 시 사용하는 것들 ###################################
    locations = state.get("information_of_object", []) or [] # 이전에 탐지 한 객체 정보들
    locations = remove_duplicates_by_id(locations)
    for_side_cloudes = []

    ################################################################################################

    # # Determine targets
    # if all_object_detect: # 만약 모든 물체 탐지하고 싶다면
    #     object_names = state.get("all_object_name", [])
    #     object_names = [name.strip() for name in object_names.split(",") if name.strip()]
    # else:
    #     object_names = state.get("object_user_want_name", [])
    cleaned_names = [n.strip().strip('"').strip("'") for n in object_names if n.strip()]

    # 로봇팔 엔드툴 포즈
    tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    tool_pos = tool_pos[0].content
    tool_pos = list(map(float, eval(tool_pos)))
    print("robot_arm_attached",tool_pos)


    # 툴 호출 + 재시도 로직 (각 객체별로 개별 처리)
    max_retries = 3
    all_results = []  # 모든 객체의 결과를 합칠 리스트
    failed_objects = []  # 감지에 실패한 객체들

    for name in cleaned_names:
        print(f"[Detector Agent] Processing object: '{name}'")
        
        attempt = 0
        raw = ""
        object_success = False  # 현재 객체의 성공 여부
        
        while attempt < max_retries:
            print(f"[Detector Agent] '{name}' attempt {attempt + 1}/{max_retries}")
            
            # 단일 객체로 호출 (원래 리스트로 넣어도 되게 하려고 했는데 텐서가 남아있어서 이상해짐)
            detect_msg = find_object_3d_properties_message([name])
            detect_res = await tool_node.ainvoke([detect_msg])
            
            first_msg = detect_res[0]
            raw = first_msg.content or ""
            
            if raw:
                print(f"[Detector Agent] Received non-empty response for '{name}'")
                # JSON 파싱 및 필터링
                try:
                    parsed = json.loads(raw)
                    # Case 1: 단일 객체
                    if isinstance(parsed, dict):
                        raw_list_strs = [json.dumps(parsed)]
                    # Case 2: 리스트 (문자열 or dict)
                    elif isinstance(parsed, list):
                        if all(isinstance(elem, dict) for elem in parsed):
                            raw_list_strs = [json.dumps(obj) for obj in parsed]
                        else:
                            raw_list_strs = parsed # 문자열 리스트라고 가정
                    else:
                        print(f"[ERROR] 예상치 못한 JSON 구조 for '{name}'")
                        raw_list_strs = []
                    # 각 항목 필터링
                    cleaned_response = []
                    for idx, item_str in enumerate(raw_list_strs):
                        try:
                            item = json.loads(item_str)
                            if "error" in item:
                                print(f"[{name}][{idx}] 필터됨 (error): {item['error']}")
                                continue
                            if item.get("area", 0) <= 0.001:
                                print(f"[{name}][{idx}] 필터됨 (area <= 0.001): {item['area']}")
                                continue
                            # 통과된 항목은 원래 포맷으로 다시 저장
                            cleaned_response.append(json.dumps(item, ensure_ascii=False, indent=2))
                        except Exception as e:
                            print(f"[{name}][{idx}] JSON 항목 처리 실패:", e)
                            print(f"[RAW] {repr(item_str)}")
                    
                    # 유효한 결과가 하나라도 있으면 성공
                    if cleaned_response:
                        all_results.extend(cleaned_response)
                        object_success = True
                        break  # 성공했으면 재시도 루프 탈출
                    else:
                        print(f"[WARNING] '{name}' - No valid items after filtering")
                        
                except Exception as e:
                    print(f"Top-level JSON parse failed for '{name}':", e)
            
            # 빈 응답이거나 파싱 실패시 재시도
            attempt += 1
            if attempt < max_retries:
                print(f"[Detector Agent] Retrying '{name}' in 0.1 seconds...")
                await asyncio.sleep(0.1)
        
        # 재시도 후에도 실패했으면 실패 목록에 추가
        if not object_success:
            failed_objects.append(name)
            print(f"[FAILED] '{name}' could not be detected after {max_retries} attempts")

    # 최종 결과
    raw = all_results  # 모든 객체의 결과가 합쳐진 리스트
    print(f"[Detector Agent] Total processed items: {len(raw)}")
    print(f"[Detector Agent] Successfully detected: {len(cleaned_names) - len(failed_objects)}/{len(cleaned_names)} objects")
    if failed_objects:
        print(f"[Detector Agent] Failed objects: {failed_objects}")

    try:
        # 이후 기존 로직대로 raw → parse_to_obj → det_list 처리
        # 만약 빈 값이 들어오면 except로 빠짐
        parsed_raw = parse_to_obj(raw) # 이후 파싱
        det_list = ensure_list_of_dicts(parsed_raw)
        # det_list 안에 아직 문자열(JSON) 요소가 남아 있을 경우, dict로 파싱
        det_list = [json.loads(item) if isinstance(item, str) else item for item in det_list]
        det_list = deduplicate_by_position(det_list)

    except Exception as e:
        print(f"[Detector Agent] Error parsing raw data: {e}")

    # 1) 루프 시작 전에 counts dict 초기화
    counts: Dict[str, int] = {}

    for idx, det in enumerate(det_list):
        try:
            # 키 검증…
            name = det["object_name"]
            key  = name.replace(" ", "_")
            count = counts.get(key, 0)
            obj_id = f"{key}_{count}"
            counts[key] = count + 1
            pos  = det["position"]
            x, y, z = [round(c, 3) for c in pos]
            x,y,_ = tool_to_world(tool_pos, x,y,0) # 월드 좌표계로 전환

            obj = {
                "id": obj_id,
                "object_name": name,
                "position": [x, y, z],
                "area": det["area"],
                "dimensions": det["dimensions"],
                "rotation_degree": det["rotation_degree"],
                "mask_base64" : det["mask_base64"],
                "score" : det["score"]
            }
            
            side_pc = {
                "id": obj_id,
                "height": z,
                "mask_base64" : det["mask_base64"],
            }

            print("obj is",obj)

            locations.append(obj)
            for_side_cloudes.append(side_pc)

        except Exception as e:
            print(f"[Detector Agent] Failed parsing for {cleaned_names[idx]}: {e}")
    locations = remove_duplicates_by_id(locations)
    # print(for_side_cloudes,"side cloude")
    side_pc_msg = generate_side_point_clouds_for_storage_message(for_side_cloudes, tool_pos[:3])
    side_pc_res = await tool_node.ainvoke([side_pc_msg])
    # print(side_pc_res, "[side pc res]")
    side = side_pc_res[0].content

    debug_visualize_function1_output(side)
    time.sleep(3)
    # 로봇팔 엔드툴 포즈
    tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    tool_pos = tool_pos[0].content
    tool_pos = list(map(float, eval(tool_pos)))
    integrated_msg = generate_integrated_point_cloud_message(side,locations[0]["id"],locations[0]["mask_base64"],locations[0]["position"][2],tool_pos[:3],[])
    integrated_res = await tool_node.ainvoke([integrated_msg])
    integrated_res = integrated_res[0].content
    # debug_structure_only(integrated_res)
    # debug_save_to_file(integrated_res)
    # debug_json_safe(integrated_res)
    # 바로 디버그
    debug_visualize_function2_output(integrated_res)
    return {
        "information_of_object": locations,
    }

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

