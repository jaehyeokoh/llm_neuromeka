from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import HumanMessage
import json
from prompts import * # 프롬프트 불러오기
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import re
from AI_messages import *
from parsing_util import *

load_dotenv()
api_key = os.getenv("OPENAI_MINE_API") # 내 계정

import base64
import cv2
import numpy as np
import os
from datetime import datetime

# text_extractor.py
from langchain_core.runnables import RunnableConfig
from ast import literal_eval


async def _get_tcp_pose(tool_node) -> list[float]:
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

async def _sam_to_dets(tool_node, boxes, idx_list, *, id_map: dict | None = None, prefix: str | None = None) -> list[dict]:
    if not idx_list:
        return []
    bx = [boxes[i] for i in idx_list]
    resp = await tool_node.ainvoke({"messages": [get_3d_properties_from_boxes_message(bx)]})
    msgs = resp.get("messages", [])
    if not msgs:
        return []
    dets = _parse_sam_content(msgs[0].content)

    # ID 부여 (SAM이 입력 순서 유지한다고 가정)
    if id_map is not None:
        for det, idx in zip(dets, idx_list):
            det["id"] = id_map.get(idx, f"selected_obj_{idx}")
    else:
        for k, det in enumerate(dets):
            det["id"] = f"{prefix}{k}"
    return dets


# 상태 정의 (state_schema)
class AgentState(TypedDict):
    user_input : str # 실제로 받은 오리지널 사용자 입력
    one_of_divided_user_input: str # 사용자 입력을 저장하는 필드 (실제 사용자 입력은 아니고 user_input_divider_agent가 분배해준 입력을 의미)
    divided_user_input:  Optional[List[str]] = [] # 나눠진 모든 사용자 입력 시컨스 저장 필드
    all_object_name: List[str] = []  # 모든 물체의 이름만
    object_user_want_name: List[str] = []  # 물체의 정보들도
    messages: list # 메시지 기록을 저장하는 필드
    information_of_object: Optional[List[Dict[str, Any]]] # 초반 탐지된 이미지속 물체의 정보를 저장하는 필드
    user_input_analyzed_by_LLM: Optional[str] # 사용자 입력을 수행하기 위한 방법을 LLM이 생각해서 저장하는 필드
    done: Optional[bool] = False  # 작업 완료 여부를 저장하는 필드
    result_of_divide_seq : Optional[List[str]] = [] # route_divide 가 분배된 명령을 저장하는 필드
    history : Optional[List[str]] = [] # 이전까지 분배된 명령-plan 저장 필드

initial_state = {
    "user_input": "",
    "one_of_divided_user_input": "",
    "divided_user_input": [],
    "all_object_name": [],
    "object_user_want_name": [],
    "messages": [],
    "information_of_object": None,
    "user_input_analyzed_by_LLM": None,
    "done": False,
    "result_of_divide_seq" : [],
    "history" : []
}

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


async def vlm_dino_finding_agent(tool_node, user_input=None, config=None) -> Dict[str, Any]:
    """DINO로 라벨링된 이미지→VLM이 고른 번호→SAM 3D 추출.
    - 선택된 박스들: det['id'] = VLM이 준 물체 이름
    - 선택되지 않은 박스들: det['id'] = f'unselected_obj_{k}'
    반환: {'det_list': [...], 'selected_label2name': {idx: name, ...}}
    """
    llm = ChatOpenAI(
        model="gpt-5",
        use_responses_api=True,                 # Responses API로 강제
        output_version="responses/v1",          # 블록형 출력 정규화
        extra_body={
            "text":{"verbosity": "low"},                 # low | medium | high
            "reasoning": {"effort": "low"},     # minimal | low | medium | high
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
        # user_input = "후레쉬베리를 공구 상자와 음식 상자중 적절한 상자에 넣어"
        user_input = "앞에 보이는거로 고인돌 모양 만들기 위해 적절한 것을 알려줘"

    # 2) VLM: 번호→이름 추출
    prompt = f"""
User_input : {user_input}

Goal: Describe the reasoning of which objects are necessary for user_input based on the overhead image. Identify the objects by their numbers. and answer all objects necessary for user_input.

Environment : The objects are on the wood_color table and photo is overhead image.

Output format:
Reasoning: <your reasoning description>

Answer:
number: object name : purpose
number: object name : purpose
"""
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": props}},
    ])
    res = llm.invoke([msg], config=config) # 주의 ainvoke 쓰면 느려짐 ㅋ
    print(res)
    # text = "\n".join(b["text"] for b in res.content if b.get("type")=="text").strip()
    # print(text)



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
    for triplet in parsed.get("answer", []):
        if len(triplet) < 2:
            continue
        try:
            raw_num = int(str(triplet[0]).strip())
        except ValueError:
            continue
        obj_name = str(triplet[1]).strip()
        idx = _label_to_index(raw_num, len(boxes))
        if idx is None:
            print(f"[VLM Agent] skip invalid label {raw_num}")
            continue
        label2name[idx] = obj_name
    print(label2name)

    selected_idxs = sorted(label2name.keys())
    unselected_idxs = [i for i in range(len(boxes)) if i not in selected_idxs]

    det_list_selected   = await _sam_to_dets(tool_node, boxes, selected_idxs, id_map=label2name) # 선택된 박스
    det_list_unselected = await _sam_to_dets(tool_node, boxes, unselected_idxs, prefix="unselected_obj_") # 선택 안된 박스


    det_list_all = det_list_selected + det_list_unselected

    # (옵션) 디버깅 출력
    print(f"[VLM Agent] selected_idxs={selected_idxs}, unselected_idxs={unselected_idxs}")
    if det_list_all:
        print("[VLM Agent] sample det 0:", {k: det_list_all[0].get(k) for k in ("id", "position", "dimensions", "area")})
        print("[VLM Agent] sample det 1:", {k: det_list_all[1].get(k) for k in ("id", "position", "dimensions", "area")})

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
    config_name = "mcp_config_test"
    mcp_server_configs = create_server_config(config_name)  # MCP 서버 설정 불러오기

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


