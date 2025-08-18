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
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from datetime import datetime
from langchain_openai import ChatOpenAI
import cv2
from ast import literal_eval
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_NEUROMEKA_API") 

async def _get_tcp_pose(tool_node) -> list[float]:
    """ output : tcp's  [x,y,z, r, p w ]"""
    msg = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    return list(map(float, literal_eval(msg[0].content)))


def _text_or_raw(content: Any) -> Any:
    """Responses 블록에서 text만 모아 합치되, 없으면 원본 content 그대로 반환."""
    # 리스트: {'type':'text','text':...} 블록 모으기
    if isinstance(content, list):
        texts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
        return "\n".join(texts).strip() if texts else content

    # 딕셔너리 한 개: text 키가 있으면 그걸, 없으면 원본
    if isinstance(content, dict):
        if content.get("type") == "text" and isinstance(content.get("text"), str):
            return content["text"].strip()
        if isinstance(content.get("text"), str):
            return content["text"].strip()
        if isinstance(content.get("content"), str):
            return content["content"].strip()
        return content

    # 문자열/기타 타입
    return content if content is not None else ""

class TextLLM:
    """LLM 래퍼: 텍스트 블록을 합쳐 문자열 반환, 없으면 원본 content 반환."""
    def __init__(self, base_llm):
        self.base = base_llm

    def invoke(self, messages, config: RunnableConfig | None = None) -> Any:
        res = self.base.invoke(messages, config=config)
        return _text_or_raw(getattr(res, "content", res))

    async def ainvoke(self, messages, config: RunnableConfig | None = None) -> Any:
        res = await self.base.ainvoke(messages, config=config)
        return _text_or_raw(getattr(res, "content", res))
    


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
            tool_pos = _get_tcp_pose(tool_node)
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

def extract_answer_content(full_text: str):
    """
    전체 텍스트에서 'Answer:' 키워드를 찾아 그 이후의 내용을 추출합니다.

    Args:
        full_text (str): 원본 전체 문자열.

    Returns:
        str or None: 'Answer:' 이후의 내용이 있으면 해당 문자열을, 없으면 None을 반환합니다.
    """
    try:
        # 1. 'Answer:'를 기준으로 문자열을 두 부분으로 나눕니다.
        #    결과는 ['"Answer:" 앞부분', '"Answer:" 뒷부분'] 형태의 리스트가 됩니다.
        parts = full_text.split("Answer:", 1) # 1은 한 번만 나누도록 하는 옵션
        
        # 2. 나뉜 리스트의 길이가 2이면 'Answer:'가 존재한다는 의미입니다.
        if len(parts) == 2:
            # 3. 두 번째 부분(parts[1])이 우리가 원하는 내용입니다.
            #    strip()으로 앞뒤 공백이나 줄바꿈을 제거하여 깔끔하게 만듭니다.
            answer_content = parts[1].strip()
            return answer_content
        else:
            # 'Answer:' 키워드가 없는 경우
            return None
            
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


def _parse_sam_content(s: str) -> list[dict]:
    """SAM ToolMessage.content -> list[dict]로 정규화."""
    try:
        outer = json.loads(s or "[]")
    except json.JSONDecodeError:
        outer = ast.literal_eval(s or "[]")
    if not isinstance(outer, list):
        outer = [outer]
    out = []
    for e in outer:
        if isinstance(e, str):
            try:
                e = json.loads(e)
            except Exception:
                continue
        if isinstance(e, dict):
            out.append(e)
    return out

async def _sam_to_dets(
    tool_node, boxes, idx_list,
    *, name_map: dict | None = None,
    purpose_map: dict | None = None,
    id_prefix: str | None = "unselected",
    default_purpose: str | None = None
) -> list[dict]:
    if not idx_list:
        return []

    bx = [boxes[i] for i in idx_list]
    resp = await tool_node.ainvoke({"messages": [get_3d_properties_from_boxes_message(bx)]})
    msgs = resp.get("messages", [])
    if not msgs:
        return []

    dets = _parse_sam_content(msgs[0].content)
    

    
    # SAM이 idx_list 순서로 반환한다고 가정
    for det, idx in zip(dets, idx_list):
        if name_map is not None:
            name = name_map.get(idx, f"obj_{idx}")
            det["id"] = f"{name}_{idx}"
            det["object_name"] = name
        else:
            det["id"] = f"{(id_prefix or 'unselected')}_{idx}"

        # purpose 채우기(있을 때만)
        if purpose_map is not None:
            det["purpose"] = purpose_map.get(idx, default_purpose)
        elif default_purpose is not None:
            det["purpose"] = default_purpose

    return dets


def save_base64_image(base64_string):
    """
    base64 이미지를 받아서 자동으로 저장
    
    Args:
        base64_string (str): "data:image/jpg;base64,..." 형태의 base64 문자열
    
    Returns:
        str: 저장된 파일 경로, 실패시 None
    """
    
    try:
        # base64 헤더 제거
        if base64_string.startswith("data:image/jpg;base64,"):
            base64_data = base64_string.replace("data:image/jpg;base64,", "")
        elif base64_string.startswith("data:image/jpeg;base64,"):
            base64_data = base64_string.replace("data:image/jpeg;base64,", "")
        else:
            return None
        
        # base64 디코딩 → numpy 배열 → OpenCV 이미지
        image_bytes = base64.b64decode(base64_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None
        
        # 자동 파일명 생성 및 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dino_labeled_{timestamp}.jpg"
        
        cv2.imwrite(filename, image)
        print(f"이미지 저장: {filename}")
        return filename
        
    except:
        return None
    


def parse_llm_output(text):
    """
    Parses LLM output with 'Reasoning' and 'Answer' sections into a structured dictionary.
    Handles slight variations in key names and formats.

    Args:
        text (str): The LLM output string.

    Returns:
        dict: A dictionary with 'reasoning' and 'answer' keys. The 'answer' value is
              a list of lists, where each inner list contains [number, object name, purpose].
    """
    text = text.lower()
    
    # Use regular expressions to find reasoning and answer sections robustly
    reasoning_match = re.search(r'reasoning:\s*(.*?)(?=\nanswer:|\Z)', text, re.DOTALL)
    answer_match = re.search(r'answer:\s*(.*)', text, re.DOTALL)

    reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ''
    
    parsed_answer = []
    if answer_match:
        answer_text = answer_match.group(1).strip()
        # Split the answer text into lines and parse each line
        for line in answer_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Use regex to split the line into number, object name, and purpose
            # This is more robust than a simple split(':')
            parts = re.split(r'\s*:\s*', line, 2)
            if len(parts) == 3:
                parsed_answer.append([parts[0].strip(), parts[1].strip(), parts[2].strip()])

    return {
        'reasoning': reasoning_text,
        'answer': parsed_answer
    }


def apply_world_transform(d, tool_pos, nd=3):
    """det list에서 position을 절대 좌표로 바꿈,d는 det list ([{id:,object name:,....},{},...]) nd는 소숫점 자리 수"""
    def one(t):
        t = t.copy()
        p = t.get("position")
        if not (isinstance(p, (list, tuple)) and len(p) >= 3): return t
        x, y, z = p[:3]
        try: ox, oy = tool_to_world(tool_pos, x, y, 0)[:2]
        except: ox, oy = x, y
        t["position"] = [round(ox, nd), round(oy, nd), round(z, nd)]
        return t
    if isinstance(d, dict): return one(d)
    if isinstance(d, (list, tuple)):
        if d and isinstance(d[0], (list, tuple)): return [apply_world_transform(i, tool_pos, nd) for i in d]
        return [one(i) for i in d]
    return d


async def vlm_dino_finding_agent(tool_node, user_input=None, config=None) -> Dict[str, Any]:
    """DINO로 라벨링된 이미지→VLM이 고른 번호→SAM 3D 추출.
    - 선택된 박스들: det['id'] = VLM이 준 물체 이름
    - 선택되지 않은 박스들: det['id'] = f'unselected_obj_{k}'
    반환: {'det_list': [...], 'selected_label2name': {idx: name, ...}}
    """
    llm = ChatOpenAI(
        model="gpt-5",
        use_responses_api=True,                 # ★ Responses API로 강제
        output_version="responses/v1",          # 블록형 출력 정규화
        extra_body={
            "text":{"verbosity": "low"},                 # low | medium | high
            "reasoning": {"effort": "minimal"},     # minimal | low | medium | high
        },
        api_key=api_key,
    )
    llm = TextLLM(llm) # 바로 text 출력 나오게 하는 래퍼

    # 1) DINO: 라벨 이미지 + 박스
    detect_res = await tool_node.ainvoke({"messages": [capture_dino_labeled_image_message()]})
    resp_msgs = detect_res.get("messages", [])
    props, boxes = None, None

    if resp_msgs:  # 감지된게 있다면
        content_raw = resp_msgs[0].content
        content_raw = parse_to_dict_for_dino(content_raw)   # parsing_util의 함수
        content = content_raw.get("image")
        boxes = content_raw.get("boxes", [])
        print("boxes are:", boxes)
        if isinstance(content, str) and content.startswith("data:image/jpg;base64,"):
            props = content
            save_base64_image(props)  # 디버깅용 저장

    if not props or not boxes:
        print("[VLM Agent] No valid image/boxes")
        return {"det_list": [], "selected_label2name": {}}

    # (테스트용) 사용자 입력이 없으면 임시로 고정 프롬프트 사용
    if not user_input:
        user_input = "곧 떨어질거 같은 거를 잡아"
        # user_input = "앞에 보이는거로 고인돌 모양 만들기 위해 적절한 것을 알려줘"
        # user_input = "작은 물건부터 순서대로 음료수 앞에 놔"

    # 2) VLM: 번호→이름 추출
    prompt = f"""
User_input : {user_input}

Goal: Describe the reasoning of which objects are necessary for user_input based on the overhead image. Identify the objects by their numbers. and answer all objects necessary for user_input.

Environment : The objects are on the wood_color table and photo is overhead image. and i made color black to outside of table (not applied to objects inside table)

Output format:
Reasoning: <your reasoning description>

Answer:
number: object name : purpose
number: object name : purpose

Area-Sort Exception:

If an instruction asks to sort ALL objects (or a multi-object set) by area/size (“smallest to largest”, “largest first”, “by area”): NEVER compute/guess area.
  → For each such object: "ID: name : i can't determine area".
If an object is only a spatial reference (e.g., “in front of the beverage can”):
  → "ID: name : reference point to place items <relation>".
Only exception: assembly tasks (finding/matching parts).
Output: one line per object, exactly "ID: name : purpose". Use label on image.
If size ordering is mentioned without naming specific target objects, treat it as above (do NOT compare sizes).

Example
Input: "Place items in front of the beverage can from smallest to largest."
Output:
0: beverage can : reference point to place items in front of
6: blue candy : i can't determine area
"""
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": props}},
    ])
    res = llm.invoke([msg], config=config) # 주의 ainvoke 쓰면 느려짐 ㅋ
    print(res)


    parsed = parse_llm_output(res)  # 네 파일의 함수: [number, object name, purpose]

    # 라벨 번호→이름 매핑 만들기 (인덱스 보정 포함)
    def _label_to_index(lbl: int, n: int) -> int | None:
        # 0-based면 그대로, 1-based면 -1 보정
        if 0 <= lbl < n:
            return lbl
        if 1 <= lbl <= n:
            return lbl - 1
        return None

    label2name: Dict[int, str] = {}
    label2purpose: Dict[int, str] = {}

    for triplet in parsed.get("answer", []):
        if len(triplet) < 2:
            continue
        try:
            raw_num = int(str(triplet[0]).strip())
        except ValueError:
            continue
        obj_name = str(triplet[1]).strip()
        purpose = str(triplet[2]).strip()
        idx = _label_to_index(raw_num, len(boxes))
        if idx is None:
            print(f"[VLM Agent] skip invalid label {raw_num}")
            continue
        label2name[idx] = obj_name
        label2purpose[idx] = purpose

    selected_idxs = sorted(label2name.keys())
    unselected_idxs = [i for i in range(len(boxes)) if i not in selected_idxs]

    det_list_selected = await _sam_to_dets(
        tool_node, boxes, selected_idxs,
        name_map=label2name,         # => id는 "{name}_{idx}", object_name=name
        purpose_map=label2purpose
    )

    det_list_unselected = await _sam_to_dets(
        tool_node, boxes, unselected_idxs,
        id_prefix="unselected",      # => id는 "unselected_{idx}"만, object_name 없음
        default_purpose=None
    )
    tool_pos = await _get_tcp_pose(tool_node)
    det_list_selected = apply_world_transform(det_list_selected, tool_pos) # 툴 포즈 받아서 절대 좌표로 바꾸는거
    det_list_unselected = apply_world_transform(det_list_unselected, tool_pos)
    det_list_all = det_list_selected + det_list_unselected # object informations 형식 : id, 
    # print(det_list_selected)

    # (옵션) 디버깅 출력
    print(f"[VLM Agent] selected_idxs={selected_idxs}, unselected_idxs={unselected_idxs}")
    if det_list_all:
        print("[VLM Agent] sample det 0:", {k: det_list_all[0].get(k) for k in ("id", "position", "dimensions", "area")})
        print("[VLM Agent] sample det 1:", {k: det_list_all[1].get(k) for k in ("id", "position", "dimensions", "area")})
    
    def extract_side_pc(dets):
        """ 출력에서 id와 height 그리고 mask_base64만 추출 -> 송 수신 데이터 절약 위해 만듬"""
        return [
            {"id": d["id"],
            "height": ((d.get("position") or []) + [None, None, None])[2],
            "mask_base64": d["mask_base64"]}
            for d in dets if d.get("id") and d.get("mask_base64")
        ]
    side_cloud_args = extract_side_pc(det_list_all)
    # print(side_cloud_args)
    side_pc_msg = generate_side_point_clouds_for_storage_message(side_cloud_args, tool_pos[:3])
    side_pc_res = await tool_node.ainvoke([side_pc_msg])
    side = side_pc_res[0].content

    target_id = det_list_selected[0]["id"]
    debug_visualize_function1_output(side)

    time.sleep(5)

    # 1차 JSON 파싱 (외부 리스트)
    result_list = json.loads(side)
    
    # 2차 JSON 파싱 (내부 딕셔너리들)
    parsed_objects = []
    for item in result_list:
        if isinstance(item, str):
            # 문자열인 경우 다시 JSON 파싱
            parsed_item = json.loads(item)
            parsed_objects.append(parsed_item)
        else:
            # 이미 딕셔너리인 경우 그대로 사용
            parsed_objects.append(item)
    
    side = parsed_objects
    
    # side_target = side[0][target_id] # 타겟의 사이드 클라우드
    # 로봇팔 엔드툴 포즈
    tool_pos = await _get_tcp_pose(tool_node)
    
    bbox = await tool_node.ainvoke([bbox_from_sidecloud_message(side, tool_pos)]) # 중요 !!! side point cloud로부터 bbox 뽑는 코드
    print(bbox[0].content)
    bbox = bbox[0].content
    # bbox 가 문자열이면 먼저 파싱
    if isinstance(bbox, str):
        bbox = json.loads(bbox)  # -> ["600","114","721","252"]

    bbox_debug = [bbox[target_id]]
    print(bbox_debug)

    resp_debug = await tool_node.ainvoke([match_input_bboxes_with_dino_message(bbox_debug)])
    if resp_debug:  # 감지된게 있다면
        content_raw = resp_debug[0].content
        content_raw = parse_to_dict_for_dino(content_raw)   # parsing_util의 함수
        content = content_raw.get("image")
        boxes_debug = content_raw.get("boxes", [])
        print("boxes debug are:", boxes_debug)
        if isinstance(content, str) and content.startswith("data:image/jpg;base64,"):
            props = content
            save_base64_image(props)  # 디버깅용 저장

    bbox = bbox[target_id]

    # 숫자로 캐스팅 (문자/실수 모두 대응)
    bbox = [[int(float(v)) for v in bbox]]

    print(bbox)
    resp = await tool_node.ainvoke({"messages": [get_3d_properties_from_boxes_message(bbox)]})
    msgs = resp.get("messages", [])
    dets = _parse_sam_content(msgs[0].content)
    print(dets)


    integrated_msg = generate_integrated_point_cloud_message(side,target_id,dets[0]["mask_base64"],dets[0]["position"][2],tool_pos[:3],[])
    integrated_res = await tool_node.ainvoke([integrated_msg])
    integrated_res = integrated_res[0].content

    debug_visualize_function2_output(integrated_res)

    return {
        "det_list": det_list_all,
        "selected_label2name": {int(i): label2name[i] for i in selected_idxs},
    }




# Graph 생성
async def create_graph(tools):
    graph = StateGraph(state_schema=AgentState)  # 상태 스키마 전달
    tool_node = ToolNode(tools)  # 도구 노드 생성
    
    # 비동기 함수로 추가
    graph.add_node("detectorAgent", wrap_async(vlm_dino_finding_agent, tool_node=tool_node))
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

