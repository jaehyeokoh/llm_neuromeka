"""Utility constructors for LangChain `AIMessage` tool calls.

This module provides small helper functions that return pre-shaped
`AIMessage` instances with `tool_calls` populated for robot control,
vision, and 3D geometry workflows.

Behavior preserved: only structure and comments were tidied for readability.
"""

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
    """LLM 래퍼: 텍스트 블록을 합쳐 문자열 반환, 없으면 원본 content 반환."""
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
    """Request 3D properties for quoted object names."""
    return _tool_call_message("find_object_3d_properties", {"object_names": quoted_name})


def generate_point_cloud_message(mask_base64, mask_shape, obj_height):
    """Generate completed point cloud from mask data and object heights."""
    return _tool_call_message(
        "generate_completed_point_cloud",
        {"mask_base64_list": mask_base64, "mask_shape_list": mask_shape, "obj_height_list": obj_height},
    )


def generate_side_point_clouds_for_storage_message(objects_data, tool_pos):
    """Generate side point clouds for later storage using current tool position."""
    return _tool_call_message(
        "generate_side_point_clouds_for_storage",
        {"objects_data": objects_data, "tool_pos": tool_pos},
    )


def generate_integrated_point_cloud_message(
    stored_side_clouds_str, target_object_id, target_mask_base64, target_obj_height, current_tool_pos, tool_pos_feedback
):
    """Generate an integrated point cloud for a target object using stored side views and the current tool pose."""
    return _tool_call_message(
        "generate_integrated_point_cloud",
        {
            "stored_side_clouds_str": stored_side_clouds_str,  # 함수 1 출력과 동일한 이중 JSON 구조
            "target_object_id": target_object_id,              # "target_cup_001"
            "target_mask_base64": target_mask_base64,          # 타겟 마스크
            "target_obj_height": target_obj_height,            # 타겟 높이
            "current_tool_pos": current_tool_pos,              # 현재 도구 위치 [x, y, z]
            "tool_pos_feedback": tool_pos_feedback,
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
    """Capture a JPG image from the specified camera index."""
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
    """Rotate gripper."""
    return _tool_call_message("robot_grip_rotate", {})


def check_IK_message(pos):
    """Check inverse kinematics for a target pose."""
    return _tool_call_message("check_IK", {"target_pose": pos})


def robot_get_tcp_pose_message():
    """Get robot TCP pose."""
    return _tool_call_message("robot_get_tcp_pose", {})


def robot_move_home_message():
    """Move robot to the predefined home position."""
    return _tool_call_message("move_home", {})


def anygrasp_message(data, target_boundary_info, debug):
    """Run AnyGrasp analysis."""
    return _tool_call_message(
        "anygrasp_analyze",
        {"data": data, "target_boundary_info": target_boundary_info, "debug": debug},
    )


def wait_idle_message():
    """Wait until the robot/system reports idle."""
    return _tool_call_message("wait_idle", {})


def monitoring_and_recover_message():
    """Perform monitoring and recovery routine."""
    return _tool_call_message("monitoring_and_recover", {})


def check_collision_message(start, end, obstacles):
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


def robot_prepare_assemble_message():
    """Prepare robot for assembly workflow."""
    return _tool_call_message("robot_prepare_assemble", {})


def robot_custom_move_message(x=None, y=None, z=None, r=None, p=None, w=None):
    """Custom robot move with optional axes and angles."""
    return _tool_call_message("robot_custom_move", {"x": x, "y": y, "z": z, "r": r, "p": p, "w": w})


def capture_dino_labeled_image_message():
    """Capture an image labeled by DINO model."""
    return _tool_call_message("capture_dino_labeled_image", {})


def get_3d_properties_from_boxes_message(boxes_coords):
    """Get 3D properties inferred from 2D box coordinates."""
    return _tool_call_message("get_3d_properties_from_boxes", {"boxes_coords": boxes_coords})

def render_image_with_boxes_debug_message(boxes_coords):
    """입력된 bbox들로 현재 카메라 프레임에 박스를 그려서 JPG(base64)와 정규화된 박스 리스트를 반환. (디버그용)"""
    return _tool_call_message("render_image_with_boxes_debug", {"boxes_coords": boxes_coords})

