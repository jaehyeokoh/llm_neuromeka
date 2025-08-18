from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from prompts import * # 프롬프트 불러오기
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
from itertools import zip_longest
import time
import open3d as o3d
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()
api_key = os.getenv("OPENAI_MINE_API") # 내 계정

import base64
import cv2
import numpy as np
import os
from datetime import datetime

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


async def vlm_dino_finding_agent(
    tool_node,user_input=None, config=None) -> Dict[str, Any]:
    """ user_input에 찾아달라는 물체 입력하면 그 물체 찾아줌

    원리:

    1. dino로 모든 객체에 박스 및 숫자 라벨링 한 이미지 추출
    2. 그 이미지와 user_input 입력해 VLM은 그에 맞는 객체가 있는 박스의 숫자 말함
    3. 그 숫자에 대한 박스를 sam으로 입력해 상세 정보 확인, id는 vlm이 추출한 id로 -> 근데 따로 저장해야 할듯 왜냐하면 dino에 입력한 id와 충돌하니까

    """
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key = api_key)

    # 1. Capture image and make labeled image with dino
    detect_res = await tool_node.ainvoke({"messages": [capture_dino_labeled_image_message()]})
    resp_msgs = detect_res.get("messages", [])
    props = None

    if resp_msgs: # 감지된게 있다면
        content_raw = resp_msgs[0].content
        content_raw = parse_to_dict_for_dino(content_raw)
        content = content_raw["image"]
        boxes = content_raw["boxes"]
        print("boxes are : ", boxes)
        if content.startswith("data:image/jpg;base64,"):
            props = content
            save_base64_image(props) # 이미지 디코딩 후 저장하는 코드 -> 디버깅용

    if not props: # 이미지가 들어오지 않을 때
        print("[VLM Agent] No valid image data, returning empty lists")
        return "No valid image data, returning empty lists"

    user_input = "후레쉬베리를 공구 상자와 음식 상자중 적절한 상자에 넣어"
    # 3. Invoke LLM
    # prompt = "이건 수직으로 아래로 찍은 사진이야 고인돌 모양을 만들기 위해 필요한 객체의 번호를 reasoning:, answer:형식으로 말해줘"
    prompt = f"""
User_input : {user_input}

Goal: Describe the reasoning of which objects are necessary for user_input based on the overhead image. Identify the objects by their numbers. and answer all objects necessary for user_input.

Environment : The objects are on the wood_color table and photo is overhead image.

Output format:
Reasoning: <your reasoning description>

Answer:
[number]: [object name] : [purpose]
[number]: [object name] : [purpose]

Output example :
Reasoning : <I can see the .....>

Answer:
1 : black rectangle : support of ~
3 : ... : ...
"""
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": props}}
    ])
    res = llm.invoke([msg], config=config)

    print(res.content)

    res_parsed = parse_llm_output(res.content)
    print("res_parsed is :", res_parsed)
    print(res_parsed["answer"][0][0])
    numbers_to_access = [int(item[0]) for item in res_parsed['answer']]

    # 3. 추출된 숫자 순서대로 boxes 리스트 출력
    print("Boxes to access:")
    boxes_selected = []
    for number in numbers_to_access:
        print(f"Index {number}: {boxes[number]}")
        boxes_selected.append(boxes[number])
    
    print("selected boxes :", boxes_selected)

    sam_res = await tool_node.ainvoke({"messages": [get_3d_properties_from_boxes_message(boxes_selected)]})
    sam_res = sam_res.get("messages", [])
    sam_res  = sam_res[0].content
    # print(sam_res)
    python_list = ast.literal_eval(sam_res)
    parsed_raw = parse_to_obj(python_list) # 이후 파싱
    det_list = ensure_list_of_dicts(parsed_raw)
    # det_list 안에 아직 문자열(JSON) 요소가 남아 있을 경우, dict로 파싱
    det_list = [json.loads(item) if isinstance(item, str) else item for item in det_list]

    print(det_list[0]["position"])
    return 


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


