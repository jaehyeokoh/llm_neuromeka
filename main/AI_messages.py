from uuid import uuid4
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from typing import Any

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
    """
    LLM 래퍼: 텍스트 블록을 합쳐 문자열 반환, 없으면 원본 content 반환.
    gpt-5로 업데이트 후 result = (content{text: 실제 result})꼴로 나오면서 이걸 파싱하기 위해 만든 함수
    llm = ChatopentAI()로 나온 llm을 이 클래스로 감싸면 llm의 출력을 파싱할 필요 없이 바로 사용 가능
    사용법 : llm = ChatopenAI() -> llm = TextLLM(llm(앞서 만든거)) -> llm.invoke()
    """
    def __init__(self, base_llm):
        self.base = base_llm

    def invoke(self, messages, config: RunnableConfig | None = None) -> Any:
        res = self.base.invoke(messages, config=config)
        return _text_or_raw(getattr(res, "content", res))

    async def ainvoke(self, messages, config: RunnableConfig | None = None) -> Any:
        res = await self.base.ainvoke(messages, config=config)
        return _text_or_raw(getattr(res, "content", res))
    




def _tool_call_message(name: str, args: dict) -> AIMessage:
    """Create an AIMessage with a single tool_call entry."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": str(uuid4()), "type": "tool_call"}],
    )


# --------------------------- Tool message ---------------------------

def find_object_3d_properties_message(quoted_name):
    """
    Request 3D properties for quoted object names. (only used in test code)
    input: object to find
    """
    return _tool_call_message("find_object_3d_properties", {"object_names": quoted_name})


def generate_side_point_clouds_for_storage_message(objects_data, tool_pos):
    """
    Generate side point clouds for later storage using current tool position.
    input: object_data [{id:.., object name:..,},...], current tool_pos
    output: [{object_id1: base64 encoded side point cloud},{},...]
    """
    return _tool_call_message(
        "generate_side_point_clouds_for_storage",
        {"objects_data": objects_data, "tool_pos": tool_pos},
    )


def generate_integrated_point_cloud_message(
    stored_side_clouds_str, target_object_id, target_mask_base64, target_obj_height, current_tool_pos
):
    """
    Generate an integrated point cloud for a target object using stored side views and the current tool pose.
    inputs: stored_side_clouds_str = 미리 저장된 사이드 포인트 클라우드, target ~ 새로 찍어서 최신화된 타겟 정보(find3dproperty 함수로 구한거), current_tool_pos = 현재 툴 포즈
    outputs: 타겟을 중심으로 크롭된 보간된 전체 포인트 클라우드
    """
    return _tool_call_message(
        "generate_integrated_point_cloud",
        {
            "stored_side_clouds_str": stored_side_clouds_str,  # 함수 1 출력과 동일한 이중 JSON 구조
            "target_object_id": target_object_id,              # "target_cup_001"
            "target_mask_base64": target_mask_base64,          # 타겟 마스크
            "target_obj_height": target_obj_height,            # 타겟 높이
            "current_tool_pos": current_tool_pos,              # 현재 도구 위치 [x, y, z]
        },
    )


def bbox_from_sidecloud_message(side_cloud, current_tool_pos):
        """Generate an bbox from target side pc using the current tool pose.
            기존 타겟의 사이드 포인트 클라우드로 로봇이 움직인 후 새로운 bbox 출력

            input : stored target side pc, current tool pos
            output : [x1, y1, x2, y2] (bbox to input sam)
        """
        return _tool_call_message(
            "bboxes_from_sideclouds",
            {
                "side_cloud_items": side_cloud,  # 함수 1 출력과 동일한 이중 JSON 구조
                "current_tool_pos": current_tool_pos,              # "target_cup_001"

            }
        )

def capture_image_as_jpg_message(index):
    """
    Capture a JPG image from the specified camera index.
    input : camera index (default = 0)
    output : base64 encoded image
    """
    return _tool_call_message("capture_image_as_jpg", {"camera_index": index})


def moveL_message(pos):
    """Linear move with [x, y, z, w] inputs., roll and pitch will fixed over_look pos"""
    return _tool_call_message("robot_move", {"x": pos[0], "y": pos[1], "z": pos[2], "w": pos[3]})


def moveL_rpy_message(pos):
    """Linear move with [x, y, z, r, p, w] inputs; bypass singularities."""
    return _tool_call_message(
        "robot_move",
        {"x": pos[0], "y": pos[1], "z": pos[2], "r": pos[3], "p": pos[4], "w": pos[5], "bypass_singular": True},
    )


def gripper_message(action):
    """Operate gripper: action should be 'Grab' or 'Release''"""
    if action == "Grab":
        name = "robot_grab"
    elif action == "Release":
        name = "robot_release"
    return _tool_call_message(name, {})  # NOTE: invalid action will raise before this call.


def gripper_rotate_message():
    """Rotate gripper to upright. (x direction of tcp will vertically down that wrist camera will vertically upright)"""
    return _tool_call_message("robot_grip_rotate", {})


def check_IK_message(pos):
    """Check inverse kinematics for a target pose. output: true = passed ik, not true = didn't passed ik"""
    return _tool_call_message("check_IK", {"target_pose": pos})


def robot_get_tcp_pose_message():
    """Get robot TCP pose."""
    return _tool_call_message("robot_get_tcp_pose", {})


def robot_move_home_message():
    """Move robot to the homming position."""
    return _tool_call_message("move_home", {})

def move_default_pos_message():
    """Move robot to the predefined default position., default pos is definded in indy_mcp_server.py"""
    return _tool_call_message("move_default_pos", {})

def anygrasp_message(data, target_boundary_info, debug):
    """
    Run AnyGrasp analysis.
    data = generated integrated point cloud(base64 encoded)
    target_boundary_info = target boundary from interpolated side cloud and target height(see detail in detect_object_mcp_server.py -> def generate_side_point_cloud())
    debug = for visualiz
    """
    return _tool_call_message(
        "anygrasp_analyze",
        {"data": data, "target_boundary_info": target_boundary_info, "debug": debug},
    )

################################################# 이 코드는 아직 사용 안함 mcp로 tool로 만든 에이전트가 시작점과 도착점 사이에 장애물 있는지 확인하게 하는 코드임 ##########
def check_collision_message(start, end):
    """Check collision between start and end. 
    NOTE: 'obstacles' parameter is currently unused to preserve original behavior.
    """
    return _tool_call_message("check_collision", {"start_obj": start, "end": end})


def save_obstacles2assemble_server(obstacles):
    """Save obstacles to assemble server."""
    return _tool_call_message("save_obstacles", {"obstacles": obstacles})


def get_obstacles2assemble_server():
    """Retrieve obstacles from assemble server."""
    return _tool_call_message("get_obstacles", {})
########################################################################################################################################################################

def robot_prepare_assemble_message():
    """Prepare robot for assembly workflow.
    (미리 지정한 자세로 이동 : 옆면 잡는 자세)"""
    return _tool_call_message("robot_prepare_assemble", {})


def robot_custom_move_message(x=None, y=None, z=None, r=None, p=None, w=None):
    """Custom robot move with optional axes and angles.
    입력한 parameter만 현재 위치에서 갱신해서 바꿈
    """
    return _tool_call_message("robot_custom_move", {"x": x, "y": y, "z": z, "r": r, "p": p, "w": w})


def capture_dino_labeled_image_message():
    """Capture an image labeled by DINO model.
    박스와 라벨링이 입혀진 이미지를 base64 encoded 로 받음 -> vlm 입력용
    """
    return _tool_call_message("capture_dino_labeled_image", {})


def get_3d_properties_from_boxes_message(boxes_coords):
    """Get 3D properties inferred from 2D box coordinates. -> bbox_coords 내부에 있는 물체의 3d 정보를 반환"""
    return _tool_call_message("get_3d_properties_from_boxes", {"boxes_coords": boxes_coords})


def match_input_bboxes_with_dino_message(input_boxes):
    """입력된 bbox와 그 프레임에서 찍은 dino를 비교해 가장 가까운 dino의 박스를 현재 카메라 프레임에 그려서 JPG(base64)와 정규화된 박스 리스트를 반환. (디버그용)
    입력 : [[bbox][]..]
    출력 : {"boxes": final_boxes, "matches": match_info, "image": out_img_b64}
    """
    return _tool_call_message("match_input_bboxes_with_dino", {"input_boxes": input_boxes})

def render_dino_excluding_overlaps_message(input_boxes):
    """ 입력 bbox들과 IoU > min_iou 로 겹치는 DINO 라벨을 제외하고,
    나머지 라벨 박스만 '네가 준 함수와 동일한 스타일'로 그려서 이미지(data URL)만 반환.
    반환: {"image": "data:image/jpg;base64,..."}
    """
    return _tool_call_message("render_dino_excluding_overlaps", {"input_boxes": input_boxes})

