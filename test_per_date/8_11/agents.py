from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, SystemMessage
import json
from prompts import * # 프롬프트 불러오기
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
from AI_messages import *
from parsing_util import *
from itertools import zip_longest
import time
import open3d as o3d
from ast import literal_eval
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_MINE_API") # 내 계정

async def _get_tcp_pose(tool_node) -> list[float]:
    """ output : tcp's  [x,y,z, r, p w ]
        warining ! always use with 'await'
    """
    msg = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    return list(map(float, literal_eval(msg[0].content)))

"""
주의사항
1. state[""] = X 로는 State 업데이트 못함 무조건 return으로 해야함
2. await는 그 함수가 끝날때까지 기다리겠다는거임 return으로 다음 에이전트 부를 때는 await넣어야 함
3. 만약 return으로 업데이트 불가능한 변수들은 그냥 전역 변수로 만들것

"""
# from kokoro_tts import KokoroEnglishTTS # tts 엔진

# tts = KokoroEnglishTTS()
# tts.start_initialization()

############################## 상태 정의 (state_schema) ##########################
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
    selected_objects_per_input : Optional[Dict[str, List[Dict[str, Any]]]] # [{"받은 명령" : [{id : , object name : , ....}]}, {}, ...]

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
    "history" : [],
    "selected_objects_per_input" : {}
}


##################################### 전역 변수 ##################################
grasp_z = 0
action_num = 0

debug = True

detect_feedback_trial = 0 # detect_failed_feedback_agent의 시도 횟수

all_objects_information = [] # 수정해나갈 물체 정보들

last_gripped = None
last_gripped_angle = None

first_grip = True

side_point_clouds = None
last_tool_pos = None # 마지막으로 잡았을 때의 툴 포즈

single_feedback = False
############################### LLM 사용하는 에이전트 함수 정의 ##########################
async def user_input_divider_agent(llm, state: AgentState, tool_node, config=None, index=0) -> dict:
    """
    사용자 입력을 받아 
    (~를 ~에 둬라, ~를 가져와라, ~를 따라라, ~를 옮겨라)등의 단순한 동작은 simple_user_input_analyze_agent에게,
    (~모양을 만들어라, ~를 닦아라, ~를 조립해라)등의 복잡한 동작은 hard_user_input_analyze_agent에게 분배하는 에이전트

    (7_29일 업데이트 내역): 
    1. vlm도 여기에 넣어서 미리 물체 이름 알게 만듬
    2. simple agent에게 할당된 Divided된 명령과 관련된 물체들을 object_user_want_name에 저장 및 vlm 에이전트 패스, 물체 특성 관련된 명령이면 아직 GroundedSAM 이전이니 object_user_want_name = []로 하고 이게 빈 리스트일때만 VLM에이전트 작동
    3. USER_INPUT_DIVIDER_PROMPT는 명령 분배, DIVIDED_INPUT_EDITOR_PROMPT는 simple agent에게 준 명령을 구체화.
    """
    user_input = input("Enter user input: ") # 유저의 입력이 여기에서 들어감

###############################################VLM_AGENT_PROMPT#########################################3
    # 1. Capture image
    print(f"[user_input_divider_agent] Capturing image with index: {index}")
    detect_msg = capture_image_as_jpg_message(index)
    detect_res = await tool_node.ainvoke({"messages": [detect_msg]})

    resp_msgs = detect_res.get("messages", [])
    props = None
    if resp_msgs: # 감지된게 있다면
        content = resp_msgs[0].content
        if content.startswith("data:image/jpg;base64,"):
            props = content

    if not props: # 이미지가 들어오지 않을 때
        print("[user_input_divider_agent] No valid image data, returning empty lists")
        return {
            "object_user_want_name": [],
            "messages": [
                {"role": "assistant", "content": "No objects detected or invalid response."}
            ]
        }

    # 3. Invoke LLM
    prompt = VLM_AGENT_PROMPT
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": props}}
    ])
    res = llm.invoke([msg], config=config)
    # text = res.content.strip()
    text = res

#################################################USER_INPUT_DIVIDER_PROMPT###############################

    system_prompt = USER_INPUT_DIVIDER_PROMPT
    divided_plan = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ], config=config)
    # print("[user_input_divider_agent] raw output: ", divided_plan)
    # divided_plan = divided_plan.content
    divided_plan = clean_keep_outer_braces(divided_plan) # 중괄호 {}만 남기는 코드
    divided_plan = json.loads(divided_plan)
    print("[user_input_divider_agent] output: ", divided_plan)
    divided_plan = divided_plan["answer"] # 주의!! reasoning삭제하는거임 만약 이거 디버그 하려면 이 코드 주석
    print("[user_input_divider_agent] divided_plan is : ", divided_plan)

    # 1. simple_agent 인덱스와 문장 추출
    simple_indices = []
    simple_actions = []
    try:
        for i, item in enumerate(divided_plan):
            if item['agent'] == 'simple_agent':
                simple_indices.append(i)
                simple_actions.append(item['action'])
    except Exception as e:
        print("error while indexing:", e)
    actions = format_actions_for_llm(simple_actions)
    print("[user_input_divider_agent] extracted actions are: ",actions)
    
###############################################DIVIDED_INPUT_EDITOR_PROMPT#############################################
    if actions != []:
        system_prompt = (
        DIVIDED_INPUT_EDITOR_PROMPT
        .replace("{objects}", text)
        .replace("{action_list}", str(actions))
        )
        editor_output = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ], config=config)

        editor_output = parse_plan_editor_output(editor_output)
        print("[user_input_divider_agent] output: ", editor_output)

        # 3. 다시 원래 위치에 덮어쓰기
        for idx, step_data in zip(simple_indices, editor_output["steps"]):
            divided_plan[idx] = {
                'agent': 'simple_agent', 
                'action': step_data["action"], 
                'objects': step_data["objects"]  # 이미 리스트 형태
            }

    final = divided_plan
    print("[user_input_divider_agent] final output is :", final)


    return{
        "divided_user_input": final,
    }



async def _sam_to_dets(
    tool_node, boxes, idx_list,
    *, name_map: dict | None = None,
    purpose_map: dict | None = None,
    id_prefix: str | None = "unselected",
    default_purpose: str | None = None
) -> list[dict]:
    """
    bbox를 넣어 나온 sam에 id, purpose, mask key를 매칭시키는거
    
    """

    if not idx_list:
        return []

    bx = [boxes[i] for i in idx_list]
    resp = await tool_node.ainvoke({"messages": [get_3d_properties_from_boxes_message(bx)]})
    msgs = resp.get("messages", [])
    if not msgs:
        return []

    dets = parse_sam_content(msgs[0].content)
    
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


async def vlm_dino_finding_agent(state: AgentState, tool_node, user_input=None, config=None) -> Dict[str, Any]:
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
            "reasoning": {"effort": "low"},     # minimal | low | medium | high
        },
        api_key=api_key,
    )


    llm = TextLLM(llm) # 바로 text 출력 나오게 하는 래퍼

    user_input = state.get("one_of_divided_user_input", "")
    selected_objects_per_input = state.get("selected_objects_per_input")

    # bytes면 문자열로
    if isinstance(selected_objects_per_input, (bytes, bytearray)):
        selected_objects_per_input = selected_objects_per_input.decode('utf-8', errors='ignore')

    # BOM 제거
    selected_objects_per_input = (selected_objects_per_input or "").lstrip('\ufeff')

    # 비었으면 기본값 사용
    if not selected_objects_per_input.strip():
        selected_objects_per_input = {}   # 혹은 []
    else:
        selected_objects_per_input = json.loads(selected_objects_per_input)

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

Volume-Sort Exception:

If an instruction asks to sort ALL objects (or a multi-object set) by area/size/volume (“smallest to largest”, “largest first”, “by area”): NEVER compute/guess volume.
  → For each such object: "ID: name : vlm can't determine volume".
If an object is only a spatial reference (e.g., “in front of the beverage can”):
  → "ID: name : reference point to place items <relation>".
Only exception: assembly tasks (finding/matching parts).
Output: one line per object, exactly "ID: name : purpose". Use label on image.
If size ordering is mentioned without naming specific target objects, treat it as above (do NOT compare sizes).

Example
Input: "Place items in front of the beverage can from smallest to largest."
Output:
0: beverage can : reference point to place items in front of
6: blue candy : vlm can't determine volume
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
    det_list_selected = apply_world_transform(det_list_selected, tool_pos) # 절대 좌표로 바꿈
    det_list_unselected = apply_world_transform(det_list_unselected, tool_pos)

    det_list_all = det_list_selected + det_list_unselected # object informations 형식 : [{id :, object_name : , ....}{}...] 

    global all_objects_information

    all_objects_information = det_list_all 
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
    side_pc_msg = generate_side_point_clouds_for_storage_message(side_cloud_args, tool_pos[:3]) # 사이드 포인트 클라우드 제작 함수 호출 (절대 좌표)
    side_pc_res = await tool_node.ainvoke([side_pc_msg])
    global side_point_clouds
    side_point_clouds = side_pc_res[0].content # 전역변수에 저장

    def collect_ids(items):
        """
        리스트 내부 id만 따로 뽑는 용도
        입력 : det_list_selected ([{id : , object_name : , ....}, {} , ...])
        출력 : [id1, id2, ...]

        나중에 상위 llm에게 전달할 때는 이거로 id와 purpose 뽑아서 줄것 근데 purpose는 아직 안만듬
        """
        ids = []
        for x in items:
            if isinstance(x, str):
                try:
                    x = json.loads(x)
                except json.JSONDecodeError:
                    continue
            if isinstance(x, dict) and "id" in x:
                ids.append(x["id"])
        return ids
    

    selected_objects_per_input[user_input] = det_list_selected
    # print("detlist", det_list_selected)
    # print("selected",selected_objects_per_input)

    return {
        "information_of_object" : det_list_all,
        "selected_objects_per_input" : selected_objects_per_input
    }


async def simple_user_input_analyze_agent(llm, state: AgentState, config=None) -> dict:
    """
    사용자 입력을 받아 어떻게 수행할지를 LLM이 판단해서 순서를 정하는 에이전트
    단순 pick n place 뿐 아니라 더 다양한 동작에 대해서도 액션을 구체적으로 나눔
    사용자 입력인 user_input만 받음 -> 물체의 이름만 받아 토큰 단순화
    출력은 1. action 1, 2. action2, 3.~ 이런식으로 할 예정
    """
    user_input = state.get("one_of_divided_user_input")
    objects_user_input = state.get("selected_objects_per_input")
    print("objects user input is :", objects_user_input)
    objects_user_input = objects_user_input[user_input] # 해당 입력을 key로 사용해 value 구함


    try:
        sorted_objs = sorted(objects_user_input, key=lambda o: o['area'])

        objects_user_input = [
            f"id : {o['id']}, purpose : {(o.get('purpose') or '').strip() or 'None'}" # 형태는 [ id : , purpose : , id : ,... 형태]
            for o in sorted_objs
        ]
        objects_user_input = str(objects_user_input)
        print("[simple_user_input_analyze_agent] objects_user_input : ", objects_user_input)
        system_prompt = SIMPLE_USER_INPUT_ANALYZE_PROMPT.replace("{objects}",objects_user_input)
        # print("system_prompt of simple_user_input_analyze_agent:", system_prompt)
    except Exception as e:
        print("Error in simple_user_input_analyze_agent:", e)
        return

    res = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ], config=config)
    """1. Move gripper to position of 'blue_and_silver_can_0'.
        2. Adjust gripper orientation to grasp 'blue_and_silver_can_0'.
        3. Close gripper to pick up 'blue_and_silver_can_0'.
        4. Move gripper to position above 'white_cup_0'.
        이런 꼴로 출력된다는 가정 하에서 밑에 코드 넣은거임 -> 줄바꿈 기준으로 액션 시컨스 나누고 앞에 번호 날림.
        따로 프롬포트에는 넣지 않았는데 만약 오류나면 출력 보고 이거 수정하거나 프롬포트 고칠것
        """
    # print("user_input_analyze_agent:",res.content)

    blocks = [
        block.strip().splitlines()
        for block in res.strip().split('\n\n')
        if block.strip()
    ]

    # 2. 출력은 [[액션1],[액션2],[]...] 이런 형식
    cleaned_blocks = [
        [re.sub(r'^\d+\.\s*', '', line) for line in block]
        for block in blocks
    ]
    # print("user_input_analyze_agent:",res.content)
    print("[user_input_analyze_agent] blocks:", cleaned_blocks)
    return{
        "user_input_analyzed_by_LLM": cleaned_blocks
    }

async def simple_plan_editor_agent(llm, state: AgentState, tool_node, config=None) -> dict:
    """
    simple_user_input_analyze_agent가 만들어준 plan을 받아 내부에 부족한 물체가 있는지 체크 및 user의 입력이 ~옆에, ~위에 같이 위치 지정이 있다면 그것도 추가하는 코드
    출력은 simple_user_input_analyze_agent의 출력 형식과 동일하게 [[액션1],[액션2],[]...] 이런 형식으로  나옴
    """
    user_input_analyzed_by_LLM = state.get("user_input_analyzed_by_LLM", [])
    user_input = state.get("one_of_divided_user_input")
    print("[simple_plan_editor] assigned one of divided_user_input : ", user_input)
    system_prompt = SIMPLE_PLAN_EDITOR_PROMPT.replace("{plan}", str(user_input_analyzed_by_LLM))
    system_prompt = system_prompt.replace("{user_input}", str(user_input))
    # print("system_prompt of simple_plan_editor_agent:", system_prompt)
    res = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage("follow system prompt")
            
        ], config=config)
    print("simple_plan_editor_agent:", res)
    parsed = parse_simple_plan_editor_output(res)
    print("parsed is :", parsed)


    # 액션 시퀀스가 정상적으로 파싱되었을 때
    print("simple_plan_editor_agent output:", parsed["blocks"])
    # splitted = split_plan_by_grip_release(parsed["answer"])
    blocks = parsed["blocks"]


    # print("[simple_plan_editor_agent] history updated :" ,state["history"])
    return {
        "user_input_analyzed_by_LLM": blocks,
    }

############################################################################################################################################################################ action supervisor
async def analyzed_user_input_regenerate_supervisor(llm, state: AgentState, tool_node, config=None) -> dict:
    """
    user_input_analyzed_by_LLM 받아 액션 시컨스 별로 나누고 정규화 시키는 에이전트
    user_input_analyzed_by_LLM리스트를 받음 -> 리스트를 action_num번째를 뽑아서 실행
    """

    user_input_analyzed_by_LLM = state.get("user_input_analyzed_by_LLM")
    print("user input analyzed by llm:",user_input_analyzed_by_LLM,"len",len(user_input_analyzed_by_LLM) )

    global action_num
    bias = 0
    # hard agent의 출력인 경우
    if isinstance(user_input_analyzed_by_LLM, list) and user_input_analyzed_by_LLM:
        if "hard_agent" in user_input_analyzed_by_LLM[0]:
            print(f"hard_agent 키 발견, {user_input_analyzed_by_LLM[0]}코드 실행")
            bias = 1
    
    # 액션 다 끝나면 종료 및 action num 초기화
    if action_num + bias >= len(user_input_analyzed_by_LLM):
        print("done")
        action_num = 0
        return{
            "done" : True
        }
    
    print("supervisor start with action num:", action_num+bias)
    user_input = user_input_analyzed_by_LLM[action_num+bias]
    if bias == 0:
        system_prompt = ANALYZED_USER_INPUT_REGENERATE_SIMPLE_PROMPT
        user_input = str(user_input)
        print("current action to execute:", user_input)
        res = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ], config=config)
    
        output = str(res)
        output = output.strip()  # 맨 앞뒤 공백·개행 제거
        if output.startswith("```") or output.endswith("```"):
            output = output[3:-3].strip()  # 맨 앞뒤 ```만 잘라내고 다시 trim

        keys = ("agent", "action", "target") # 각각 사용할 agnet, 수행할 action, 타겟 물체 or 장소
        parsed = [
            dict(zip(keys, (part.strip() for part in line.split(":"))))
            for line in output.strip().splitlines()
        ]
        print("[analyzed_user_input_regenerate_supervisor] parsed is:", parsed)
        if detect_duplicate_targets_on_release(parsed): # 만약 잡는 대상과 목표 위치의 대상이 같다면 액션 패스
            action_num += 1
            print("[analyzed_user_input_regenerate_supervisor] action passed because target is same to location")
            return await analyzed_user_input_regenerate_supervisor(llm,state,tool_node)
    elif bias == 1:

        if isinstance(user_input_analyzed_by_LLM, dict):
            parsed = list(user_input_analyzed_by_LLM.values())
        elif isinstance(user_input_analyzed_by_LLM, list):
            parsed = user_input_analyzed_by_LLM
        else:
            raise TypeError("user_input_analyzed_by_LLM must be a list or dict.")

    print("[analyzed_user_input_regenerate_supervisor] user input analyzed by llm : ",  parsed)
    
    for curr, nxt in zip_longest(parsed[bias:], parsed[bias+1:]):
        # print("In loop type:", type(parsed))
        if curr["agent"] == "move":
            if nxt is not None and nxt["agent"] == "grip": # move다음에 grip이면 이거로 해서 grip하기 전에 그 물체로 바로 가지 않게
                print("[analyzed_user_input_regenerate_supervisor] move start")
                result = await move_agent( state, tool_node, curr, grip = True)
            elif nxt is not None and nxt["agent"] == "release":
                print("[analyzed_user_input_regenerate_supervisor] move start2")
                print("move before release")
                result = await move_agent(state, tool_node, curr, release = True)
            else:
                print("[analyzed_user_input_regenerate_supervisor] move start3")
                result = await move_agent( state, tool_node, curr)

        elif curr["agent"] == "grip":
            print("[analyzed_user_input_regenerate_supervisor] grip start")
            result = await grip_agent(llm, state, tool_node,curr)
            if result == "failed": # 역기구학 실패한 경우
                return  {
                "done" : True
                }

            print("grasp_z is ", grasp_z)

        elif curr["agent"] == "release":
            print("[analyzed_user_input_regenerate_supervisor] release start")
            await release_agent(tool_node, curr)

        elif curr["agent"] == "tilt":
            print("[analyzed_user_input_regenerate_supervisor] tilt start")
            print("tilt activate")

        elif curr["agent"] == "done":
            print("hard agent plan is done")
            action_num = 0
            return{
                "done" : True
            }

    action_num +=1
    return await analyzed_user_input_regenerate_supervisor(llm,state,tool_node)


############################################################################################################################################################################ move agent
async def move_agent(state:AgentState , tool_node, move_plan, config=None, grip = False, release = False) -> dict:
    """
    analyzed_user_input_regenerate_supervisor가 호출 시 기본적인 move와 관련된 명령(move_plan)을 수행
    목표는 정확한 좌표로 이동한 후 이동완료 신호를 보내는것
    이동할때는 다른 물건과 충돌이 없도록 일단은 수직, 수평이동으로 함
    앞서 받은 액션이 단순한 move같은 거면 llm없이 실행, 복잡한 액션이면 llm 실행
    """
    offset_distance = 0.15
    print("tool_node in move_agent:", type(tool_node), getattr(tool_node, "keys", lambda: [])())

    # 로봇팔 엔드툴 포즈
    tool_pos = await _get_tcp_pose(tool_node)
    print("robot_arm_attached",tool_pos)

    plan = move_plan["action"]
    target = move_plan["target"]
    global last_gripped
    global action_num
    global all_objects_information
    global grasp_z
    collision_margin = 0

    if grip or plan == "above" and not release: # 잡을 때나 놓을 때는 오프셋 적용
        height_offset = 0.33 # 물체 위 30cm에서 바라봄
        x_offset = -0.1 # 카메라 오프셋 적용 (yaw가 180이라는 가정 즉, 정방향)
    elif release:
        print("[move_agent] moving before releasing start")
        grasp_z # 마지막으로 파지한 위치
        height_offset = grasp_z + 0.0 # 마지막으로 파지한 위치에서 xcm위에서 놓음
        x_offset = 0
    else: # 그냥 이동일 때는 해당 위치로 이동
        x_offset = 0
        height_offset = 0

    print("move plan:", plan)
    print("target:", target)
    print("action_num", action_num)
    objects_original = state.get("information_of_object", [])
    if all_objects_information == []:
        all_objects_information = objects_original

    objects = all_objects_information
    print("objects is ", objects)
    # 정상적인 action이 들어왔을 때 = llm 사용 안함
    if "home" not in str(target).strip() and "home" not in str(plan):
        print("moving before grip or release")
        obj = next((item for item in objects if item['id'] == str(target)), None) # 타겟이 저장된 리스트에 있는지 확인하는 코드

        if obj: # 타겟 물체가 저장되어 있다면
            print("[move_agent] 타겟 물체가 state에 저장되어 있음:", obj)
            x = obj["position"][0]
            y = obj["position"][1]
            z = obj["position"][2]

        else: # 타겟이 저장 되어있지 않으면 
            if plan == "center_of_2_obj": # 만약 두 타겟의 중심으로 가야 한다면
                print(f"[move agent] go to center of objects: {target[0]}, {target[1]}")
                obj1 = next((item for item in objects if item['id'] == str(target[0])), None)
                obj2 = next((item for item in objects if item['id'] == str(target[1])), None)
                x = (obj1["position"][0]+ obj2["position"][0])/2
                y = (obj1["position"][1]+ obj2["position"][1])/2
                z = (obj1["position"][2]+ obj2["position"][2])/2
            else:
                x = 0.5
                y = 0.1
                z = 0.02

        if action_num == 0: # 첫번째 액션이라면
            pos = [x+x_offset,y,z+height_offset,tool_pos[3],tool_pos[4],tool_pos[5]]
            edited_pos = apply_plan_offset(plan, pos,offset_distance, grasp_z)
            print("edited_pos is : ", edited_pos)
            # 충돌 감지
            collision_res = check_collision(tool_pos[:3], edited_pos[:3], all_objects_information) # return[0] = True(collision), False(not collision)/ return[1] = z_margin / error
            print("collision res is ",collision_res)
            if collision_res[0]:
                print("collision occured , margin is ", collision_res[1])
                offset_current_pos = [tool_pos[0],tool_pos[1],tool_pos[2]+collision_res[1],tool_pos[3],tool_pos[4],tool_pos[5]]
                msg = moveL_rpy_message(offset_current_pos)
                result_msg = await tool_node.ainvoke({"messages": [msg]})
                edited_pos[2] = edited_pos[2] + collision_res[1]
                collision_margin = collision_res[1]

            if release:
                print("applying release offset")
                before_release_pos = edited_pos
                before_release_pos[2] = tool_pos[2] + collision_margin
                msg = moveL_rpy_message(before_release_pos)
                result_msg = await tool_node.ainvoke({"messages": [msg]})

            # msg = moveL_rpy_message(edited_pos)
            # result_msg = await tool_node.ainvoke({"messages": [msg]})
            
            edited_pos[2] -= collision_margin
            msg = moveL_rpy_message(edited_pos)
            result_msg = await tool_node.ainvoke({"messages": [msg]})
            print("[move_agent] result move", result_msg)
            if not result_msg:
                print("[move_agent] 경로로 못감")

        else: # 첫번째 액션이 아니라면
            # if obj["id"] in last_gripped: # 이전에 잡아서 옮겨서 release까지 끝낸 객체를 다시 잡는다면
            #     print(f"{obj['id']}를 이전에 이미 집은 적이 있습니다. 옮겼던 위치로 이동합니다.")

            #     """이동 후 탐지 -> find_closest_detection로 그 물체 확인, 만약 발견 하면 그 위치로 이동, 발견 못하면 피드백 루프"""
            #     # msg = find_object_3d_properties_message(obj["name"])
            #     # result_msg = await tool_node.ainvoke({"messages": [msg]})
            #     # 여기에 탐지 실패 코드 넣을것
            # else:
            pos = [x+x_offset,y,z+height_offset,tool_pos[3],tool_pos[4],tool_pos[5]]
            edited_pos = apply_plan_offset(plan, pos,offset_distance, grasp_z)
            collision_res = check_collision(tool_pos[:3], edited_pos[:3], all_objects_information) # return[0] = True(collision), False(not collision)/ return[1] = z_margin / error
            print("collision res is ",collision_res)
            if collision_res[0]:
                print("collision occured , margin is ", collision_res[1])
                offset_current_pos = [tool_pos[0],tool_pos[1],tool_pos[2]+collision_res[1],tool_pos[3],tool_pos[4],tool_pos[5]]
                msg = moveL_rpy_message(offset_current_pos)
                offset_msg = await tool_node.ainvoke({"messages": [msg]})
                print("offset movement : ",offset_msg)
                edited_pos[2] = edited_pos[2] + collision_res[1]
                collision_margin = collision_res[1]

            msg = moveL_rpy_message(edited_pos)
            result_msg = await tool_node.ainvoke({"messages": [msg]})
            edited_pos[2] -= collision_margin
            msg = moveL_rpy_message(edited_pos)
            result_msg = await tool_node.ainvoke({"messages": [msg]})
            print("[move_agent] result move2", result_msg)
            if not result_msg:
                print("경로로 못감")


    else: # 이상한 action이 들어왔을 때 -> llm 사용
        # llm 사용 코드 넣을 것
        print("[move_agent] go to home pose")
        # tool_pos[0],tool_pos[1],tool_pos[2] = 0.4,0.15,0.3
        # msg = moveL_rpy_message(tool_pos)

        result_msg = await tool_node.ainvoke({"messages": [robot_move_home_message()]})
        # msg = moveL_message([0.4,0.15,0.45,0])
        # print("msg is:", msg)
        # result_msg = await tool_node.ainvoke({"messages": [msg]})
        return
    
    return





############################################################################################################################################################################ grip agent
async def grip_agent(llm, state:AgentState , tool_node, move_plan, config=None) -> dict:
    """
    물체 잡으러 가는 자세, 잡는 파지점, 잡아서 올리는 거 까지 책임지는 에이전트
    입력 : 잡을 물체 이름
    내부 추론 : 물체 이름 GroundedSAM2에 넣어 정보 얻음-> 피드백 구조 만들 것!,
    물체 정보 LLM에 프롬포트와 함께 전달
    
    실행 단계:
    0. 타겟 상단까지 move했다는 가정 하에서 타겟의 상단에서 카메라로 사진 찍어 물체 이름으로 물체 위치 확인 및 같은 물체가 여러개 있다면 이전에 내가 목표로 한 타겟과 가장 가까운 물체를 타겟으로 함
    -> 리턴 : 타겟 정보 ( pos, mask)

    1. 타겟 포함 주변 물체 vlm으로 감지해서 옆면이 보간된 point cloud를 생성 및 anygrasp 서버 실행해 파지점 추출
    2. 파지점의 rpy를 분석 후 수직 그랩이 많으면 수직으로, 수평 그랩이 많으면 수평으로 취급 -> 이건 테스트 해봐야 할듯 아니면 윗면 넓이와 높이 보고 추측하던가
    3. 수평이면 각도 제한을 시도(먼저 수평면으로부터 10~20도 기울어진거-> 없으면 20~30-> 없으면 30~40 이런순서로.. 절대 -각도가 안나오게)-> 만약 정 없으면 그나마 되는것중 점수 높은거로
    성공 시 -> 안전 높이까지 올린 후 return
    실패 시 -> 일단 실패 감지 로직을 만들어야 함
    """

    ##################### State로부터의 값  #################
    objects = state.get("information_of_object") # 물체 정보들이 저장된 리스트

    target_object = move_plan["target"] # supervisor로부터 전달받은 타겟 물체 id

    # all_object_names = state.get("all_object_name")
    # print("all object gripper go :",all_object_names)
    ########################################################

    ################# 로봇팔 엔드툴 포즈 추출 ################
    tool_pos = await _get_tcp_pose(tool_node)
    print("엔드툴 pos : ",tool_pos)
    ########################################################

    ################ 전역 변수 #############################
    global grasp_z # 마지막으로 파지한 위치
    global first_grip # 처음 잡냐 아니냐 플래그
    global all_objects_information # 여기에 모든 물체 정보 저장 후 나중에 충돌 감지에서 사용
    global side_point_clouds # 저장된 모든 물건의 보간된 사이드 포인트 클라우드
    global last_gripped # 마지막으로 잡은 물체
    global last_gripped_angle # 마지막으로 물체 잡을 때 각도
    global last_tool_pos
    #######################################################
    last_gripped = target_object

    # 입력으로 전달받은 타겟 id를 추출함
    target_name = re.sub(r'_[0-9]+$', '', target_object) # 타겟에서 id뒤에있는 _숫자 이거 없애는 코드 (GroundedSAM2에는 id를 포함해서 넣으면 못찾으니까 이름을 넣음)
    print("[grip_agent] 타겟 이름은:",target_name)



######################################################################################################3
    max_retries = 2
    closest_obj = []
    # for name in target_name:
    name = target_name
    print(f"[그리퍼 Agent] Processing object: '{name}'")
    attempt = 0
    raw = ""
    
    # 1차 JSON 파싱 (외부 리스트)
    result_list = json.loads(side_point_clouds)
    
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
    bbox = bbox[target_object]
    # 숫자로 캐스팅 (문자/실수 모두 대응)
    bbox = [[int(float(v)) for v in bbox]]
    resp = await tool_node.ainvoke({"messages": [get_3d_properties_from_boxes_message(bbox)]})
    msgs = resp.get("messages", [])

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


    dets = _parse_sam_content(msgs[0].content)
    # dets = apply_world_transform(dets, tool_pos)
    dets = update_world_coordinates(tool_pos, dets)
    print(dets)

    integrated_msg = generate_integrated_point_cloud_message(side, target_object, dets[0]["mask_base64"], dets[0]["position"][2], tool_pos[:3],[]) # 전체 포인트 클라우드
    integrated_res = await tool_node.ainvoke([integrated_msg])
    cloud_res = integrated_res[0].content

    print("cloud res is ",cloud_res[:80])
    try:
        cloud_res = json.loads(cloud_res)  # 리스트 안의 문자열들
        target_boundary_info = cloud_res["target_boundary_info"]
        print("target boundary info:", target_boundary_info)

    except Exception as e:
        print("파싱중 오류남",e, "리턴된 클라우드: ", cloud_res)

############################################################ anygrasp 영역 ##################################################

    """입력
    cloud_res = point cloud
    detect_res = GroundedSAM2 실행 결과
    target_boundary_info = 타겟의 target_boundary_info(윤곽선, 최대 최소 높이 -> anygrasp에서 사용하려고 만듬)
    debug = 시각화 디버그
    """
    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 2
    attempt = 0
    try:
        # print("target side pts cloud is ", target_boundary_info[:200])
        print(f"[anygrasp Agent] Tool call attempt {attempt + 1}/{max_retries}")

        grasp_msg = anygrasp_message(cloud_res,target_boundary_info, debug)

        grasp_res = await tool_node.ainvoke([grasp_msg])
        print("grasp res is : ",grasp_res[0].content[:200])

        # List[ToolMessage] 반환 가정
        first_msg = grasp_res[0]
        raw = first_msg.content or ""
        raw = json.loads(raw)
        # print("[anygrasp Agent] raw answer is ",raw)
        
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
        # print("gg is ", gg[0:5])
        closest_obj = dets[0]
        target_x = closest_obj["position"][0] # 타겟 중심 위치
        target_y = closest_obj["position"][1]

        # anygrasp의 출력을 로봇 좌표계로 바꾸는거
        for g in gg:
            x, y, z = g["translation"]
            g["translation"] = [-y, -x, z]

        from scipy.spatial.transform import Rotation as R
           
        try:
            # # 로봇팔 엔드툴 포즈
            tool_pos = await _get_tcp_pose(tool_node)

            x = target_x
            y = target_y
            # if closest_obj["position"][2] >= 0.12:
            #     grasp_height = closest_obj["position"][2] - 0.04
            # else:
            grasp_height = closest_obj["position"][2]

            g = None
            pos1 = None  

            # 1단계: 그래스프 포인트들을 기울기에 따라 분류
            low_tilt_grasps = []  # x도 미만 (수평에 가까운)
            high_tilt_grasps = [] # x도 이상 (수직에 가까운)
            all_grasp = []
            tilt_deg_threshold = 50 # 수평면으로부터의 x도 threshold  

            for i in gg:
                rpy_deg = np.array(i["rpy"])
                rpy_rad = np.radians(rpy_deg)
                rot_matrix = R.from_euler('xyz', rpy_rad).as_matrix()
                tool_z = rot_matrix[:, 2]
                
                # 수평면으로부터의 기울기 계산
                tilt_from_xy = 90 - abs(np.degrees(np.arccos(tool_z[2])))
                
                # 기울기에 따라 분류
                grasp_data = {
                    'grasp': i,
                    'rpy_deg': rpy_deg,
                    'rot_matrix': rot_matrix,
                    'tool_z': tool_z,
                    'tilt': tilt_from_xy
                }
                all_grasp.append(grasp_data)
                if abs(tilt_from_xy) <= tilt_deg_threshold:
                    low_tilt_grasps.append(grasp_data)
                else:
                    high_tilt_grasps.append(grasp_data)
                

            print(f"수평에 가까운 그래스프 (≤{tilt_deg_threshold}°): {len(low_tilt_grasps)}개")
            print(f"수직에 가까운 그래스프 (>{tilt_deg_threshold}°): {len(high_tilt_grasps)}개")

            primary_grasp_height = 0
            secondary_grasp_height = 0
            # 2단계: 우선순위 결정 및 높이 오프셋 설정
            # 수평이 더 많을 때
            if len(low_tilt_grasps) >= len(high_tilt_grasps):
                primary_list = low_tilt_grasps
                secondary_list = high_tilt_grasps
                # 수평(기울기 작음): 4cm 아래 (단, 최소 높이 보장)
                if grasp_height >= 0.11:
                    primary_grasp_height = grasp_height - 0.045
                elif grasp_height<= 0.03:
                    primary_grasp_height = 0.03 + abs(grasp_height - 0.03) 
                else:
                    primary_grasp_height = grasp_height - 0.02

                # 수직(기울기 큼): 3cm 아래 (단, 최소 높이 보장) (나무 판이 3cm라서 3보다 작으면 무조건 3 이상으로 맞추기.)
                if grasp_height >= 0.09:
                    secondary_grasp_height = grasp_height - 0.01 
                elif grasp_height <= 0.03:
                    secondary_grasp_height = 0.03 + abs(grasp_height - 0.03) 

            else:
                primary_list = high_tilt_grasps
                secondary_list = low_tilt_grasps
                # 수직(기울기 큼): 3cm 아래 (단, 최소 높이 보장) (나무 판이 3cm라서 3보다 작으면 무조건 3 이상으로 맞추기.)
                if grasp_height >= 0.09:
                    primary_grasp_height = grasp_height - 0.01 
                elif grasp_height <= 0.03:
                    primary_grasp_height = 0.03 + abs(grasp_height - 0.03) 

                # 수평(기울기 작음): 4cm 아래 (단, 최소 높이 보장)
                                # 수평(기울기 작음): 4cm 아래 (단, 최소 높이 보장)
                if grasp_height >= 0.11:
                    secondary_grasp_height = grasp_height - 0.045
                elif grasp_height<= 0.03:
                    secondary_grasp_height = 0.03 + abs(grasp_height - 0.03) 
                else:
                    secondary_grasp_height = grasp_height - 0.02

            async def try_grasp_list(grasp_list, list_name, adjusted_height):
                """그래스프 리스트를 시도하고 성공한 그래스프 데이터를 반환"""
                print(f"\n=== {list_name} 시도 중 ===")
                
                for grasp_data in grasp_list:
                    i = grasp_data['grasp']
                    rpy_deg = grasp_data['rpy_deg']
                    tool_z = grasp_data['tool_z']
                    tilt_from_xy = grasp_data['tilt']
                    score = i["score"]
                    print("grasp score is ", score)
                    if score <= 0.06: # 점수 기준으로 필터링
                        continue
                    
                    print(f"기울기 {tilt_from_xy:.2f}° 그래스프 시도 중...")
                    
                    if tilt_from_xy > -3: # 기울기가 역방향이면 건너뛰기
                        continue
                    
                    # 기존 IK 체크 로직
                    pos1 = [x, y, adjusted_height, *rpy_deg]
                    offset = -0.08 * tool_z
                    approach_pos = [pos1[0] + offset[0], pos1[1] + offset[1], pos1[2] + offset[2], *rpy_deg]
                    
                    check_msg = check_IK_message(pos1)
                    ik_res = await tool_node.ainvoke([check_msg])
                    check_msg2 = check_IK_message(approach_pos)
                    ik_res2 = await tool_node.ainvoke([check_msg2])
                    
                    if ik_res[0].content == "true" and ik_res2[0].content == "true":
                        print(f"IK 체크 성공!! 기울기: {tilt_from_xy:.2f}°")
                        return grasp_data, i, adjusted_height
                    else:
                        print(f"IK 실패 - 기울기: {tilt_from_xy:.2f}°")
                
                print(f"{list_name} 모든 그래스프 실패")
                return None, None, adjusted_height

            # 3단계: 우선순위에 따라 IK 체크 (명확한 fallback)
            successful_grasp = None
            g = None
            
            # 최대 3개까지 순서대로 기울기 확인
            for best_grasp in all_grasp[:5]:
                try:
                    if abs(best_grasp['tilt']) >= 76:
                        successful_grasp = best_grasp
                        g = best_grasp['grasp']
                        if grasp_height >= 0.09:
                            adjusted_height = grasp_height - 0.01 
                        elif grasp_height <= 0.03:
                            adjusted_height = 0.03 + abs(grasp_height - 0.03)
                        else:
                            adjusted_height = grasp_height
                        print(f"\nIK 없이 선택된 그래스프 (수직 우선), 기울기: {best_grasp['tilt']:.2f}°")
                        break  # 성공했으면 루프 종료
                except Exception as e:
                    print("grasp 검사 중 에러:", e)

            # 위에서 실패할 시
            if successful_grasp is None and len(primary_list) > 0:
                # 먼저 primary 리스트 시도
                successful_grasp, g, adjusted_height  = await try_grasp_list(primary_list, "우선순위 그래스프 리스트",primary_grasp_height )

                # primary 리스트가 모두 실패하면 secondary 리스트 시도
                if successful_grasp is None and len(secondary_list) > 0:
                    print(f"\n우선순위 리스트 실패, 백업 리스트로 전환...")
                    successful_grasp, g, adjusted_height = await try_grasp_list(secondary_list, "백업 그래스프 리스트", secondary_grasp_height)

                if successful_grasp is None:
                    print("\n모든 그래스프 포인트에서 IK 체크 실패")
                    # 여기서 에러 처리 또는 다른 대안 로직
                    best_grasp = primary_list[0]
                    successful_grasp = best_grasp
                    g = best_grasp['grasp']
                    adjusted_height = primary_grasp_height
                    print(f"\n 가장 점수 높은것 사용, 기울기: {best_grasp['tilt']:.2f}°")
                    # raise Exception("사용 가능한 그래스프 포인트가 없습니다.")
                else:
                    print(f"\n최종 선택된 그래스프 기울기: {successful_grasp['tilt']:.2f}°")
                    
            try:
                # 이후 실제 동작 실행
                tilt_from_xy = successful_grasp['tilt']
                tool_z = successful_grasp['tool_z']
                pos1 = [x, y, adjusted_height, *successful_grasp['rpy_deg']]
                
                # 만약 접근 위치의 z가 x 이하면 +5cm 해줌
                offset = -0.08 * tool_z
                if pos1[2] + offset[2] <= 0.1 and pos1[0] < 0.65 and pos1[1] < 0.5:
                    height_offset = 0.08
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
                

                # resmove1 = await tool_node.ainvoke([moveL_rpy_message(approach_pos1)]) # ({"messages": [msg]})
                resmove1 = await tool_node.ainvoke({"messages":[moveL_rpy_message(approach_pos1)]})
                print("resmove1 is", resmove1)
                

                # 전역변수 last_gripped_angle 에 잡았을 때의 각도 저장
                last_gripped_angle= np.degrees(np.arccos(tool_z[2]))
                print("[Gripper agent] last_gripped_angle is saved ", last_gripped_angle)

                if abs(tilt_from_xy) <= tilt_deg_threshold:  # 기울기가 수평면으로부터 58도 이하라면 그리퍼 재정렬
                    # 그리퍼 정렬 (새 RPY)
                    rpy_raw = await tool_node.ainvoke([gripper_rotate_message()])
                    new_rpy = [float(v) for v in json.loads(rpy_raw[0].content)]
                    print(f"그리퍼 재정렬됨: {new_rpy}")
                else:
                    new_rpy = [pos1[3], pos1[4], pos1[5]]
                    print(f"재졍렬된 그리퍼 아닌 오리지널 값 사용: {new_rpy}")

                # 접근 위치 2: 정렬된 RPY로 offset 적용
                approach_pos2 = [
                    pos1[0] + offset[0],
                    pos1[1] + offset[1],
                    pos1[2] + offset[2],
                    *new_rpy
                ]
                # resmove2 = await tool_node.ainvoke([moveL_rpy_message(approach_pos2)])
                resmove2 = await tool_node.ainvoke({"messages":[moveL_rpy_message(approach_pos2)]})
                print("resmove2 is", resmove2)

                # 최종 목표 위치 이동 (정렬된 RPY 적용)
                if pos1[2] <= 0.03:
                    pos1[2] = 0.04 
                target_pos = [pos1[0], pos1[1], pos1[2]+0.007, *new_rpy]
                await tool_node.ainvoke({"messages":[moveL_rpy_message(target_pos)]})

                print("movel pos1:", target_pos)

                grip_res = await tool_node.ainvoke([gripper_message("Grab")])
                time.sleep(0.5)
                print("그립 완료:", grip_res)
                last_tool_pos = target_pos
                
                # 마지막으로 잡은 높이 저장
                grasp_z = pos1[2]
                print("[gripper agent] grasp z is updated", grasp_z)

                target_pos2 = [pos1[0], pos1[1], pos1[2] +0.1, *new_rpy] # 잡은 후 10cm 위로 이동
                await tool_node.ainvoke({"messages":[moveL_rpy_message(target_pos2)]})
                
                
                first_grip = False

                print("movel pos2:", target_pos2)
            except Exception as e:
                print("[grip_agent] error 3", e)
        except Exception as e:
            print("[grip_agent] error 2", e)
    except Exception as e:
        print("[grip_agent] error 1", e)


async def release_agent(tool_node, move_plan) -> dict:
    """
    그리퍼 릴리즈 하는 에이전트
    보통은 llm 없이 그냥 지정 위치가 맞는지 확인 후 놓는거로
    에러상황 발생시 llm호출하는 거로
    """
    global side_point_clouds
    global all_objects_information
    global last_tool_pos
    print("[release agent]: last tool pos",last_tool_pos)
    await tool_node.ainvoke({"messages": [gripper_message("Release")]})
    time.sleep(0.2) # release는 바로 안되니까 0.x초 정도 기다림
    tool_pos = await _get_tcp_pose(tool_node)

    releasing_object = move_plan["target"]
    targets = [t.strip() for t in releasing_object.split(',')]
    if len(targets) == 2:
        releasing_object = targets[0]

    obj = next((item for item in all_objects_information if item['id'] == str(releasing_object)), None)
    if obj != None:
        for obj in all_objects_information:
            if obj.get("id") == str(releasing_object):
                print("[Release agent ]물체 놔둔 후 위치 업데이트 :  놔둔 물체 :", releasing_object, "기존 : ",obj["position"][:2], "새로운 위치",  tool_pos[:2])
                obj["position"][:2] = tool_pos[:2]  # 새 위치 정보로 업데이트
                break  # 해당 객체만 수정하고 종료
    
    # 로봇 좌표계 → 카메라 좌표계 변환 및 이전과의 오프셋 차이 적용 및 위치 이동 적용
    tool_pos_side_pc = [
        -(tool_pos[1] -last_tool_pos[1]),   # -Y
        -(tool_pos[0] -0.07-last_tool_pos[0]),  # -X
        # -(tool_pos[2] - 0.15 -last_tool_pos[2])     # -Z
        0 # z놔줄때와 잡을때와 변화 거의 없다는 가정 하에서 한거 나중에 수정할것
    ]

    print(f"[release agent] applyed offset to {releasing_object} with offset {tool_pos_side_pc}")



    print("type(side_point_clouds)=", type(side_point_clouds))
    # print("side cloud :", side_point_clouds[:70])
    if isinstance(side_point_clouds, dict):
        result_list = [side_point_clouds]
    else:
        result_list = json.loads(side_point_clouds)

    # 2) 내부 요소 파싱 및 검증
    parsed_objects = []
    for i, item in enumerate(result_list):
        if isinstance(item, str):
            item = json.loads(item)  # 내부가 JSON 문자열
        if not isinstance(item, dict):
            raise TypeError(f"outer[{i}] must be dict, got {type(item)}")
        parsed_objects.append(item)

    side = parsed_objects
    # print("side cloud :", side[:70])
    print("type(parsed)=", type(side),
        "elem_type=", type(side[0]) if isinstance(side, list) and side else None)


    updated = apply_offset_and_reencode(side, releasing_object, tool_pos_side_pc)
    side_point_clouds = updated



    z = tool_pos[2] + 0.1
    await tool_node.ainvoke({"messages": [robot_custom_move_message(z=z)]}) # 놓은 후 위로 10cm 이동
    await tool_node.ainvoke({"messages": [robot_move_home_message()]}) # 홈 위치로 이동

    return

async def tilt_agent(tool_node, move_plan) -> dict:
    """
    그리퍼 릴리즈 하는 에이전트
    보통은 llm 없이 그냥 지정 위치가 맞는지 확인 후 놓는거로
    에러상황 발생시 llm호출하는 거로
    """
    global all_objects_information
    global last_gripped
    global last_gripped_angle

    obj_current = None # 잡았을 때 물체의 자세 (물체의 높이와 장축을 비교해서 확인함 -> 높이가 가장 크면 서있는거)
    rollNpitch = None # 물체의 롤, 피치 조정할 때 (누운걸 세우거나 서있는걸 눞일때)
    yaw = False # 물체의 yaw 조절할때 -> parallel에서 사용
    r_offset = 0

    tool_pos = await _get_tcp_pose(tool_node)

    tilting_object = last_gripped

    obj = next((item for item in all_objects_information if item['id'] == str(tilting_object)), None)
    if obj != None:
        if obj["position"][2] >= obj["dimention"][1]:
            obj_current = "standing"
        else:
            obj_current = "lying"

    if move_plan["action"] == "vertical":
        if obj_current == "standing":
            rollNpitch = False
        else:
            rollNpitch = True
    
    elif move_plan["action"] == "parallel":
        if obj_current == "standing":
            rollNpitch = True
            yaw = True
        else:
            rollNpitch = False
            yaw = True


    if move_plan["action"] == "vertical":# 먼저 vertical에서 누워있는걸 세우는거 위주로
        print("make object to vertical")
        await tool_node.ainvoke({"messages": [robot_prepare_assemble_message()]}) # 기본 조립 자세로 이동
        if rollNpitch:
            r_offset = last_gripped_angle
        print("[tilt agent] r_offset is :", r_offset, "and T pos is ", tool_pos)
        r_original = tool_pos[3]
        
        # await tool_node.ainvoke({"messages": [robot_custom_move_message(r = r_offset)]}) # 잡을 때 오프셋만큼 보정

        # 0.45,0.15,0.39,105,0,180
        # 0.45,0.15,0.39,30,90,90
    elif move_plan["action"] == "parallel":
        print("make object to parallel")
        await tool_node.ainvoke({"messages": [robot_move_home_message()]}) # 수평 자세로 호밍

    return




async def hard_agent_supervisor(llm, state: AgentState, tool_node, config=None, index: int = 0) -> Dict[str, Any]:
    """
    어려운 명령 받아서 분배 및 관리하는 에이전트 

    입력 : 어려운 명령
    출력 : 액션 시컨스

    에이전트 역할 : 
    1. 먼저 명령의 종류 구분(조립, 모호한 입력, 복잡한 경로 생성, 기타)
    2. 종류별로 그에 맞는 에이전트에게 배치
    3. 에이전트로부터 답변 받으면 분배한 명령과 답변받은 명령의 유사성 파악
    4. 목적에 맞으면 이걸 원래 액션 위치에 들어가게 divider 프롬포트 및 코드 통해 분리시킴 -> 다시 쉬운것만 실행 후 어려운걸 걸러서 주면 그걸 다시 분배
    
    AGENTS:
    - Assembly_agent: Handles requests to build or assemble structures or object shapes.
    - Analysis_agent: Handles vague, unclear, or ambiguous user instructions.
    - Hard_Trajectory_agent: Handles tasks involving complex or irregular movement paths.
    - Reasoning_agent: General-purpose reasoning agent used when no other category applies.

    예상되는 문제: 
    1. 쉬운걸 실행 후 어려운걸 받았을 때 이건 어떻게 배분? 원래의 목적이 남아 있을까?
    2. 어떻게 어려운 task를 해석한걸 평가할까 이상한 출력을 보정할 방법은?
    3. divider에이전트를 통해 다시 하는것보다 그냥 이 코드에 넣는게 더 좋을듯, 전에 피드백 구조를 detect agent에 넣다가 나중에 분리한거 생각하면 이게 맞음

    """
    ############################## 초기 입력 값 ##########################
    one_of_divided_user_input = state.get("one_of_divided_user_input")

    #####################################################################
    assign_prompt = HARD_AGENT_SUPERVISOR_ASSIGN_PROMPT.replace("{plan_input}", str(one_of_divided_user_input))
    assign_output = (llm.invoke([SystemMessage(content=assign_prompt)], config=config)).content
    selected_agent = parse_agent_assignment_output(assign_output)
    selected_agent = str(selected_agent["answer"])
    print("[HARD_AGENT_SUPERVISOR_ASSIGN] agent is :", selected_agent)
    if selected_agent == "Assembly_agent":
        # return await Assembly_agent(llm,state,tool_node,config)
    # elif selected_agent == "Analysis_agent":
    #     agent_output = await
    # elif selected_agent == "Hard_Trajectory_agent":
    #     agent_output = await
    # elif selected_agent == "Reasoning_agent":
    #     agent_output = await
        print("good")


async def Assembly_agent(llm, state: AgentState, tool_node, config=None, index: int = 0) -> Dict[str, Any]:
    """
    assembly agent
    """
    # global all_objects_information
    user_input = state.get("one_of_divided_user_input")
    # detect_result = await detector_agent_without_llm(state, tool_node, all_object_detect = True)
    detect_result = state.get("information_of_object")
    print("[assembly agent] detect raw output",detect_result)
    print("[assembly agent] detect information of object output",detect_result["information_of_object"])
    locations = detect_result["information_of_object"]
    all_objects_information = locations
    filtered_obj = filter_objs_dimensions_height_name_area(locations)
    print("object that llm got :", filtered_obj)
    shape_added_object_list = add_shape_from_id(filtered_obj)
    print("shape added is :", shape_added_object_list)

    # 첫 assemble 계획 세우는 에이전트 프롬포트
    system_prompt = """
WHO YOU ARE:
You are a planning agent who creates a general assembly plan for a requested structure.

GOAL:
Your goal is to generate an **abstract assembly plan** for the structure the user wants to build,
**without being given specific object information**.

INPUT:
- user_input: a description of the structure or plan to assemble

RULES:
- Do NOT use any specific object names unless the user explicitly requests them in quotes.
- If the user did not provide any specific object, then describe what kind of objects are needed in terms of shape, size, flatness, or matching height.
- Focus on clear reasoning in natural language.
- Your output should be easy for another LLM to understand and use.

HOW TO REASONING:
1. Explain what the requested structure or command to assemble is and its basic concept.
2. Identify what components are needed and their characteristics.
3. Describe the construction plan step by step in natural language.
4. For each step, explain why that step is needed and what kind of object would be suitable.

OUTPUT FORMAT (natural language):
reasoning:
Provide your reasoning in 1-4 complete sentences.

answer:
- Step 1: Describe the first construction step and the type of objects needed.
- Step 2: Describe the next step and the type of objects needed.
- Continue until the abstract plan is complete.
"""
    res = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ], config=None)
    res = res.content
    print("res is :", res)
    print("len of object is ", len(shape_added_object_list))

    # 첫 계획 받아 수정 및 검토하는 에이전트
    system_prompt2 = f"""
WHO YOU ARE:
You are an evaluation and refinement agent for plans executed by a robot arm that can only perform pick-and-place.

GOAL:
Evaluate whether the given plan is logically sound and compatible with pick-and-place-only execution, then refine it if needed. Keep everything in natural language.

INPUT FORMAT (description, not data):
    user_input: natural language description of the target structure.
    plan_input: the planner’s natural language reasoning and step-by-step plan.

CONSTRAINTS:
- Design structures achievable with {len(shape_added_object_list)} objects maximum
- Each object can only be used once
- If original plan requires more objects, simplify the structure
- Include width, size, or spacing descriptions if they affect stability or placement

ASSUMPTIONS:
- No human involvement (no gloves, no help).
- No physical calculations (torque, friction).
- Only basic geometric actions are allowed.
- Don't need minor alignment.

WHAT TO CHECK:
- Does the plan match the user's structure description?
- Are the steps clear, logically ordered, and PnP-compatible?
- Are required object features described? (e.g., similar height, flat surface)
- Is orientation (upright, horizontal) explicitly stated?
- Are there redundant or unclear steps?
- Does the plan stay object-agnostic?
- Does each step handle exactly one object? (Robot arm can only pick one object at a time)
- Are size, width, or spacing details provided where they are essential?
- Is object-to-object placement direction (left, right, forward, backward) or alignment (center) clearly specified when necessary?
- Reject vague directions like “from above” or “from surface” or "on surface"

REFINEMENT RULES:
- If a step mentions placing multiple objects simultaneously, split into separate steps
- Each step should describe placement of exactly one object
- Maintain logical order (dependencies between steps)
- Add width or spacing descriptions only if needed for correct placement or stability

REASONING SEQUENCE (Step-by-step):
1. Count total number of objects used in plan steps. Does it exceed {len(shape_added_object_list)}? If yes, simplify.
2. Check if each step describes exactly one pick-and-place action. If not, split or correct.
3. Verify logical order: Does each step build upon the previous one in a stable and feasible way?
4. Identify if object features (height, flatness) are described when needed for stability.
5. Evaluate if orientation (horizontal, upright, etc.) is specified where necessary.
6. Check for essential width, spacing, or size information — only where needed for correct placement.
7. Confirm plan is object-agnostic — avoid naming shapes unless required.
8. Identify redundant or ambiguous steps; remove or clarify as needed.

OUTPUT (natural language):
reasoning: Summarize the plan’s strengths and any critical issues found in the steps above. Focus on whether the plan meets the object limit, PnP compatibility, and essential clarity.
revised_plan: only if needed; otherwise return the input plan Keep steps PnP-compatible and object-agnostic.

Mini-Example (dolmen, evaluator output):
reasoning:
The plan uses more objects than allowed and includes steps with multiple simultaneous placements, violating pick-and-place constraints. Object features and orientation are generally clear, but minor clarity issues exist in size references.

revised_plan:
Place first upright support of similar height with flat top surface.
Place second upright support of similar height with flat top surface, spaced right from the first to form a gap smaller then capstone's span.
Place a wide, flat capstone horizontally across both supports.

------------
INPUT:
plan_input : {res}

    """

    res2 = llm.invoke([
        SystemMessage(content=system_prompt2),
        HumanMessage(content=user_input)
    ], config=None)
    res = res2.content
    print("res2 is :", res)
    res2_parsed = parse_eval_refine_output_loose(str(res2), ["reasoning", "revised_plan"])
    # print("parsed_plan2 is ",res2_parsed["revised_plan"])
    res2_revised = res2_parsed["revised_plan"]

    # 두번째 계획 받아 위치만 검토하는 에이전트
    prompt_locate = f"""
ROLE:
You are a minimal editor that only verifies and, if necessary, fixes the plan’s LOCATION/PLACEMENT details for a pick-and-place-only robot.

INPUT:
- user_input
- plan: 
{res2_revised}

CONSTRAINTS:
- Do NOT change non-location content (object roles, counts, selection criteria, wording).
- Only check/fix: orientation (upright/horizontal), spacing/span, centering/alignment.
- No physics, no human/tool notes, no new steps unless strictly required.
- If the plan already satisfies the checks, return the original plan unchanged.

CHECKLIST (location only):
- Supports: stated as upright, parallel, flat top surfaces, similar height.
- Capstone: horizontal placement, centered, simultaneous contact with both flat tops.
- Alignment: only small translations/rotations if mentioned.
- Presence test must be explicit: treat an item as PRESENT only if the exact idea is stated in the plan text (do not infer).
- Consider these as explicit signals (examples, not exhaustive)
- gap smaller than (or <) capstone span/length/width

DECISION RULES:
- If ANY checklist item is missing or only implied → decision = "fix-location-only".
- When fixing, edit ONLY the affected location/placement phrase(s); keep all other wording identical (verbatim).
- Keep step numbering and non-location wording verbatim; add only the missing location/placement terms.



EXAMPLE (minimal fix style):
- "with a gap between them." → "with a gap smaller than the capstone’s span."
- "Place the capstone horizontally across the supports." → "Place the capstone horizontally across the supports, centered, and simultaneously contacting both flat tops; make small translations/rotations if needed."


OUTPUT (exactly two sections):
location_analysis:
- State which required items are present/missing (brief bullets).
decision:
- "keep" or "fix-location-only"
final_plan:
- If decision=keep → print the original plan verbatim
- If decision=fix-location-only → minimally edit ONLY the location/placement lines. Keep all other text identical
Note: Output only the final result, not both original and edited versions.
"""

    res3 = llm.invoke([
        SystemMessage(content=prompt_locate),
        HumanMessage(content=user_input)
    ], config=None)
    res3 = res3.content
    print("res3 is :", res3)
    res3_parsed = parse_eval_refine_output_loose(str(res3), ["location_analysis", "decision","final_plan"])
    # print(res3_parsed["final_plan"])


    area_sorted_objects = get_area_sorted_objects(shape_added_object_list)
    similar_height_groups = get_similar_height_groups(shape_added_object_list)

    # plan과 내가 detect를 통해 알아낸 물체 정보를 받아 각각의 plan별 필요한 물체를 매칭시키는 에이전트
    prompt_match_object = f"""
    ROLE:
    You are a minimal editor that matches abstract object descriptors to concrete object IDs.

    INPUT:
    - plan: {res3_parsed["final_plan"]}
    - objects_by_area: {area_sorted_objects}
    - similar_height_groups: {similar_height_groups}

    MATCHING RULES:
    - For "similar height" requirements: use objects from same group in similar_height_groups
    - Objects from different groups cannot be used together for "similar height" requirements
    - For "wide/large/capstone" requirements: use objects from objects_by_area list
    - Consider shape compatibility (rectangular objects are more suitable for flat surfaces)
    - Each object can only be used once
    - If insufficient objects in a single group, remove excess steps rather than mixing groups
    - For distance expressions (e.g., capstone's span), use only the object ID (e.g., capstone → object_id3). Do not use “span” or other descriptors.

    PROCESS (do each step carefully):

    STEP 1: List all abstract descriptors in the plan
    - Find words like "support", "capstone", "wide", "flat", etc.
    - Include possessive forms like "capstone's", "support's"

    STEP 2: Select concrete objects for each descriptor
    - "similar height" → pick from SAME group in similar_height_groups only
    - Do NOT mix objects from different height groups
    - If a group has insufficient objects, skip the excess steps
    - "wide/large/capstone" → pick from top of objects_by_area considering shape
    - "flat" requirements → strongly prefer rectangular shapes

    STEP 3: Create replacement mapping
    - support1 → object_id1
    - support2 → object_id2  
    - capstone → object_id3

    STEP 4: Apply replacements to create final plan
    - Replace every abstract term with its mapped concrete object ID
    - For distance references (e.g., capstone's span), replace with object ID only (e.g., object_id3)
    - Do not use “span”, “length”, or similar terms in the final output

    OUTPUT (natural language format):
    step1_descriptors:
    - List each descriptor on separate lines

    step2_selections:
    - Explain selections in natural language

    step3_mapping:
    - Show mappings as simple text

    final_plan:
    <plan text here>

    IMPORTANT: Do NOT use JSON format. Use natural language.
    """
    res4 = llm.invoke([
        SystemMessage(content = prompt_match_object),
        HumanMessage(content="follow system prompt")
    ], config=None)
    res4 = res4.content
    print("res4 is :", res4)
    res4_parsed = parse_eval_refine_output_loose(str(res4), ["step1_descriptors","step2_selections","step3_mapping","final_plan"])
    # print(res4_parsed["final_plan"])

    # 출력 형식을 파싱하기 쉽게 포멧 변경하는 프롬포트
    format_plan_prompt = f"""
ROLE:
You are an expert that converts natural language robotic assembly plans into structured format.

INPUT:
- plan: Multi-step assembly plan (each step starts with "- Step N:")

OUTPUT STRUCTURE:
For each step, output these 6 keywords:
- target_object: object to be picked and moved
- reference_object: object or location as placement reference  
- relative_position: left, right, front, back, above, below, center
- distance: specific measurement (e.g., 5cm, 10mm), or object-based measurement (e.g., obj_3, obj_1) or None
- object_orientation: vertical, parallel, None
- alignment_reference: reference object for parallel orientation (only when parallel)

Guidelines:
- If spatial relationship is unclear, default to center
- If distance not specified, default to normal
- Each step must handle exactly ONE target object
- If input mentions multiple target objects in one step, create separate steps
- Multiple reference objects are allowed, separate with comma

OUTPUT FORMAT:
STEP N
analysis: Natural language explanation of what this step intends to do
target_object: object_id
reference_object: object_id or location
relative_position: selected_value
distance: measurement or None
object_orientation: vertical or parallel
alignment_reference: object_id (only when parallel)

EXAMPLE:
Input: Place beam1 horizontally across block1_0 and block2_0, spaced from the first to form a gap smaller than the white_box_0.

Output:
STEP 1
analysis: Place beam1 horizontally to bridge across both blocks with white_box_0’s span spacing, creating a dolmen structure
target_object: beam1
reference_object: block1_0,block2_0
relative_position: above
distance: white_box_0
object_orientation: parallel
alignment_reference: block1_0,block2_0

EXAMPLE2:
Input: 
1. Pick black_rectangular_block_1 with a flat bottom and similar height to black_rectangular_block_2, and place it upright on the surface to serve as the first support.
2. Pick black_rectangular_block_2 with a flat bottom and similar height to black_rectangular_block_1, and place it upright on the surface, spaced apart from black_rectangular_block_1 so that the gap is slightly less than black_rectangular_block_0.
3. Pick black_rectangular_block_0 with a wide, flat surface, and place it horizontally across the tops of black_rectangular_block_1 and black_rectangular_block_2, centered and simultaneously contacting both flat tops, to form the capstone.

Output:
STEP 1
analysis: Pick black_rectangular_block_1 and place it upright on the surface to act as the first support.  
target_object: black_rectangular_block_1  
reference_object: None  
relative_position: above  
distance: None  
object_orientation: vertical  

STEP 2
analysis: Pick black_rectangular_block_2 and place it upright on the surface, spaced apart to the right of black_rectangular_block_1 so the gap is slightly less than the length of black_rectangular_block_0.  
target_object: black_rectangular_block_2  
reference_object: black_rectangular_block_1  **Important** Do not add surface on here!! only exact object is accept!!! **
relative_position: right  
distance: slightly less than black_rectangular_block_0  
object_orientation: vertical 

STEP 3  
analysis: Pick black_rectangular_block_0 and place it horizontally across the tops of black_rectangular_block_1 and black_rectangular_block_2, centered and contacting both to form the capstone.  
target_object: black_rectangular_block_0  
reference_object: black_rectangular_block_1, black_rectangular_block_2  
relative_position: above  
distance: None  
object_orientation: parallel  
alignment_reference: black_rectangular_block_1, black_rectangular_block_2

--------------------
INPUT:
plan : {res4_parsed["final_plan"]}

    """
    res5 = llm.invoke([
        SystemMessage(content=format_plan_prompt),
        HumanMessage(content="follow system prompt")
    ], config=None)
    res5 = res5.content
    print("res5 is :", res5)
    res5_parsed = formatted_plan_parser(res5)
    print(res5_parsed)
    converted = convert_plan_to_commands(res5_parsed)

    return{
        **state,
        "user_input_analyzed_by_LLM": converted
    }
    # done = tts.speak_auto_sequential(filtered_obj)
    # done.wait()


