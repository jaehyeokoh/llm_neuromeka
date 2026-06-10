from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from prompts import * # 프롬프트 불러오기
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
from itertools import zip_longest
import time





############################## 상태 정의 (state_schema) ##########################
# 상태 정의 (state_schema)
class AgentState(TypedDict):
    user_input: str # 사용자 입력을 저장하는 필드
    divided_user_input:  Optional[List[str]] = [] # 나눠진 사용자 입력 저장 필드
    all_objects_name: List[str] = []  # 기본값을 빈 리스트로 설정
    matched_object_ids: Optional[List[str]] = []   # 사용자 입력과 매칭된 물체 이름을 저장하는 필드
    detect_failed_objects: Optional[List[str]] = []  # 탐지 실패한 물체 이름을 저장하는 필드
    detect_failed_vlm_feedback: Optional[List[str]] = []  # vlm으로 탐지 실패한거 피드백한 필드
    passed_object: Optional[List[str]] = []  # 재탐지 해도 탐지 못한 물체 저장
    messages: list # 메시지 기록을 저장하는 필드
    information_of_object: Optional[List[Dict[str, Any]]] # 초반 탐지된 이미지속 물체의 정보를 저장하는 필드
    objects_user_input:  Optional[Dict[str, List[float]]] # Userinput과 관련된 물체의 정보만 필터링해 저장하는 필드
    user_input_analyzed_by_LLM: Optional[str] # 사용자 입력을 수행하기 위한 방법을 LLM이 생각해서 저장하는 필드
    regenerated_user_input: Optional[str]  # 사용자 입력을 구체화한 내용을 저장하는 필드
    initial_plan: Optional[str]  # 초기 계획을 저장하는 필드
    plan_with_coordinates: Optional[str]  # 좌표와 함께 계획을 저장하는 필드
    action_num: Optional[int] = 0 # 완료한 액션 번호를 저장하는 필드
    selected_object_id: Optional[str] = None # 현재 액션에 해당하는 물체 이름을 저장하는 필드
    selected_actions: Optional[List[str]] = None # 현재 액션에 해당하는 액션 리스트를 저장하는 필드
    selected_object_location: Optional[Dict[str, float]] = None # 현재 액션에 해당하는 물체의 위치를 저장하는 필드
    selected_target: Optional[str] = None # 현재 액션에 해당하는 목적지 이름을 저장하는 필드
    location_of_target: Optional[Dict[str, float]] = None # 현재 액션에 해당하는 목적지의 위치를 저장하는 필드
    done: Optional[bool] = False  # 작업 완료 여부를 저장하는 필드
    detect_feedback: Optional[bool] = False # detect 단계에서 피드백 스위치
    last_gripped: List[str] = [] # release 했던 물체들 전부 append로 저장
    result_of_divide_seq : Optional[List[str]] = []

initial_state = {
    "user_input": "",
    "divided_user_input": [],
    "all_objects_name": [],
    "matched_object_ids": [],
    "detect_failed_objects": [],
    "detect_failed_vlm_feedback":[],
    "passed_object": [],
    "messages": [],
    "information_of_object": None,
    "objects_user_input": None,
    "regenerated_user_input": None,
    "user_input_analyzed_by_LLM": None,
    "initial_plan": None,
    "plan_with_coordinates": [],
    "action_num": 0,
    "selected_object_id": None,
    "selected_actions": None,
    "selected_object_location": None,
    "location_of_target": None,
    "done": False,
    "detect_feedback": False,
    "last_gripped": [],
    "result_of_divide_seq" : []
}


##################################### 전역 변수 ##################################
grasp_z = 0

############################### LLM 사용하는 에이전트 함수 정의 ##########################
async def user_input_divider_agent(llm, state: AgentState, config=None) -> dict:
    """
    사용자 입력을 받아 
    (~를 ~에 둬라, ~를 가져와라, ~를 따라라, ~를 옮겨라)등의 단순한 동작은 simple_user_input_analyze_agent에게,
    (~모양을 만들어라, ~를 닦아라, ~를 조립해라)등의 복잡한 동작은 hard_user_input_analyze_agent에게 분배하는 에이전트
    """
    user_input = input("Enter user input: ")
    system_prompt = USER_INPUT_DIVIDER_PROMPT
    res = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ], config=config)
    res = res.content
    res = json.loads(res) 
    print("first", res["answer"])
    state["divided_user_input"] = res["answer"]
    return{
        "divided_user_input": res["answer"]
    }





async def vlm_agent(
    llm, state: AgentState, tool_node, config=None, index: int = 0) -> Dict[str, Any]:
    """ 초반에 이미지를 받아와 이미지 속 물체의 이름을 GroundedSAM2가 알아듣기 좋게 
        색상 + 이름으로 반환하는 에이전트,
        
        만약 객체 탐지 실패해서 피드백용으로 다시 탐지 시 
        탐지 실패한 객체가 유효한 객체인지 user input과 관련 있는지 재확인 하는 용도로 프롬포트 변경됨

        출력 형태: ['tan box', 'white box', 'white box'] 이런 식
    """
    # Debug: starting VLM agent
    print("[VLM Agent] Starting execution")
    detect_failed = state.get("detect_failed_objects", [])
    user_input = state.get("user_input", "")

    # 1. Capture image
    print(f"[VLM Agent] Capturing image with index: {index}")
    detect_msg = capture_image_as_jpg_message(index)
    detect_res = await tool_node.ainvoke({"messages": [detect_msg]})

    resp_msgs = detect_res.get("messages", [])
    props = None
    if resp_msgs: # 감지된게 있다면
        content = resp_msgs[0].content
        if content.startswith("data:image/jpg;base64,"):
            props = content

    if not props: # 이미지가 들어오지 않을 때
        print("[VLM Agent] No valid image data, returning empty lists")
        return {
            "all_objects_name": [],
            "detect_failed_objects": [],
            "messages": [
                {"role": "assistant", "content": "No objects detected or invalid response."}
            ]
        }

    # 피드백 루프 아닐때 입력되는 프롬포트
    if not detect_failed:
        prompt = VLM_AGENT_PROMPT

    # detect_failed가 있을때의 피드백용 
    else:
        prompt = VLM_FEEDBACK_FAILED_OBJECTS_PROMPT.format(
            detect_failed_objects=json.dumps(detect_failed, ensure_ascii=False),
            user_input=user_input
        )
        print(f"[VLM Agent] Using feedback prompt for failed objects: {detect_failed}")
        # 3. 피드백용 메세지
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": props}}
        ])
        res = llm.invoke([msg], config=config)
        text = res.content.strip()

        # 4. Parse response: always comma-separated list
        names = [name.strip() for name in text.split(",") if name.strip()]
        print(f"[VLM Agent] Parsed object names: {names}")

        # 5. Update state
        state["detect_failed_vlm_feedback"] = names
        print(f"[VLM Agent] State updated: detect_failed_vlm_feedback={names}")
        return {
        "detect_failed_vlm_feedback": names,
        "messages": [
            {"role": "assistant", "content": f"detect_failed_vlm_feedback : {names}"}
            ]
        }

    # 3. Invoke LLM
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": props}}
    ])
    res = llm.invoke([msg], config=config)
    text = res.content.strip()

    # 4. Parse response: always comma-separated list
    names = [name.strip() for name in text.split(",") if name.strip()]


    # 추가: 사용자 입력과 관련된 이름만 남김 ############################################################# 7_11 추가
    print("text is ",text)
    sys_prompt = USER_INPUT_MATCH_NAME_PROMPT.replace("{objects}", text)
    res = llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_input)
    ], config=config)
    res = res.content
    res = re.sub(r"^```json\s*", "", res.strip(), flags=re.IGNORECASE)
    res = re.sub(r"```$", "", res.strip())
    print("dete디버그",res)
    res = json.loads(res)
    names = res["answer"][0].split(', ')
    print("물체 이름을 추출하는 사고과정:", res["chain_of_thought"])
    print("사용자 입력과 관련된 탐지된 객체들", names)

    #######################################################################################################
    # 5. Update state
    state["all_objects_name"] = names
    state["detect_failed_objects"] = []
    print(f"[VLM Agent] State updated: all_objects_name={names}")

    # 6. Return unified envelope with debug messages
    return {
        "all_objects_name": names,
        "detect_failed_objects": [],
        "messages": [
            {"role": "assistant", "content": f"Detected objects: {names}"}
        ]
    }


async def detector_agent_without_llm(state: AgentState, tool_node) -> Dict[str, Any]:
    """ all_object_name 또는 detect_failed_objects를 기반으로 위치 + 크기 + 면적 + 각도를 탐지
        같은 이름의 객체가 여러개 감지되는 경우를 대비해 객체별로 id를 부여함
         ex) white box가 2개면 white_box_1, white_box_2 이런식으로 """

    ########################## 피드백 발동 시 사용하는 것들 ###################################
    detect_failed = state.get("detect_failed_vlm_feedback", []) # vlm을 거쳐서 나온 탐지 실패 객체 이름
    locations = state.get("information_of_object", []) or [] # 이전에 탐지 한 객체 정보들
    locations = remove_duplicates_by_id(locations)
    detect_failed_original = state.get("detect_failed_objects", []) # vlm 거치기 전 이전 탐지 실패 객체 이름
    detect_feedback = state.get("detect_feedback") # detect feedback 스위치
    ################################################################################################

    # Determine targets
    object_names = state.get("all_objects_name", []) if not detect_failed else detect_failed
    cleaned_names = [n.strip().strip('"').strip("'") for n in object_names if n.strip()]

    # 로봇팔 엔드툴 포즈
    tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    tool_pos = tool_pos[0].content
    tool_pos = list(map(float, eval(tool_pos)))
    print("robot_arm_attached",tool_pos)


    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 3
    attempt = 0
    raw = ""
    while attempt < max_retries:
        print(f"[Detector Agent] Tool call attempt {attempt + 1}/{max_retries}")
        detect_msg = find_object_3d_properties_message(cleaned_names)
        detect_res = await tool_node.ainvoke([detect_msg])
        # print("detect res is : ",detect_res)

        # List[ToolMessage] 반환 가정
        first_msg = detect_res[0]
        raw = first_msg.content or ""
        if raw:
            print("[Detector Agent] Received non-empty response")

            ###################################### 내부에 error나 이상한 값 (넓이가 매우 작다던가) 하는거 날리는 코드###
            # 1단계: 최상위 리스트 문자열 파싱
            try:
                raw_list_strs = json.loads(raw)  # 리스트 안의 문자열들
                print(f"[INFO] Parsed {len(raw_list_strs)} items.")
            except Exception as e:
                print("Top-level JSON parse failed:", e)
                raw_list_strs = []

            # 2단계: 각 문자열 항목 → dict로 파싱 및 필터링
            cleaned_response = []

            for idx, item_str in enumerate(raw_list_strs):
                try:
                    item = json.loads(item_str)  # 문자열을 dict로 파싱

                    if "error" in item:
                        print(f"[{idx}] 필터됨 (error): {item['error']}")
                        continue

                    if item.get("area", 0) <= 0.001:
                        print(f"[{idx}] 필터됨 (area <= 0.001): {item['area']}")
                        continue

                    # 통과된 항목은 원래 포맷으로 다시 저장
                    cleaned_response.append(json.dumps(item, ensure_ascii=False, indent=2))

                except Exception as e:
                    print(f"[{idx}] JSON 항목 처리 실패:", e)
                    print(f"[RAW] {repr(item_str)}")

            raw = cleaned_response
            #########################################################################################################

            break

        # 빈 응답이면 짧게 대기 후 재시도
        attempt += 1
        await asyncio.sleep(0.1)

    if not detect_feedback and not raw:
        # 재시도 하기 전 처음으로 vlm으로부터 받은 이름 입력 시 못찾으면 피드백
        print("[Detector Agent] All retries failed, marking as detect_failed_objects")
        state["detect_failed_objects"] = cleaned_names
        state["detect_feedback"] = True
        return {
            "information_of_object": locations,
            "detect_failed_objects": cleaned_names,
            "detect_feedback": True,
            "messages": [
                AIMessage(
                    content=f"Failed to detect 3D properties for {cleaned_names} after {max_retries} attempts.",
                    name="detector_agent_without_llm"
                )
            ]
        }
    try:
        # 이후 기존 로직대로 raw → parse_to_obj → det_list 처리
        # 만약 빈 값이 들어오면 except로 빠짐
        parsed_raw = parse_to_obj(raw) # 이후 파싱
        det_list = ensure_list_of_dicts(parsed_raw)
        # det_list 안에 아직 문자열(JSON) 요소가 남아 있을 경우, dict로 파싱
        det_list = [json.loads(item) if isinstance(item, str) else item for item in det_list]

    except:
        
        print("can't find object", detect_failed_original)
        state["detect_failed_objects"] = []
        state["detect_feedback"] = False
        state["passed_object"] = detect_failed_original
        return {
            "detect_failed_objects": [],
            "detect_feedback": False,
            "passed_object":  state["passed_object"]
        }

    # 피드백 루프용 코드
    if detect_feedback:
        print("[Detector Agent] Matching with existing locations")
        
        added = False
        for single_det in det_list:
            new_obj = match_or_create_object(prev_list=locations, new_det=single_det, tool_pos=tool_pos) # 기존에 감지된 물체와 증복되는지 확인 용
            print(f"[Detector Agent] match_or_create_object result: {new_obj}")
            
            if new_obj: # 만약 기존과 다른 user  input을 만족하는 새로운 물체 발견 시 발동
                locations.append(new_obj)
                added = True

        if added:
            locations = remove_duplicates_by_id(locations)
            # 2) 새로 추가된 객체가 하나라도 있으면 상태 갱신
            state["information_of_object"] = locations
            state["detect_failed_objects"] = []
            state["detect_failed_vlm_feedback"] = []
            state["detect_feedback"] = False
            print(f"[Detector Agent] Updated locations: {locations}")

            return {
                "information_of_object": locations,
                "detect_failed_objects": [],
                "detect_feedback": False,
                "messages": [
                    AIMessage(
                        content=json.dumps(locations, ensure_ascii=False, indent=2),
                        name="detector_agent_without_llm"
                    )
                ]
            }
        else: # 피드백 해도 아무것도 안들어오면 그냥 그 물체 무시, passed_object에 저장 -> 물체 없으므로 그냥 종료
            print("passed_object:", detect_failed_original)
            state["passed_object"] = detect_failed_original
            state["detect_failed_objects"] = []
            state["detect_failed_vlm_feedback"] = []
            state["detect_feedback"] = False
            return{
                "detect_failed_objects" : [],
                "detect_failed_vlm_feedback" : [],
                "detect_feedback": False,
                "passed_object": detect_failed_original
            }

    new_failed = []
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
            dims = det["dimensions"]
            x, y, z = [round(c, 3) for c in pos]
            L, W    = dims

            x,y,w = tool_to_world(tool_pos, x,y, det["rotation_degree"]) # 월드 좌표계로 전환

            obj = {
                "id": obj_id,
                "name": name,
                "position": {"x": x, "y": y, "z": z},
                "dimensions": {"L": L, "W": W},
                "area": det["area"],
                "rotation_degree": w
            }
            print("obj is",obj)

            locations.append(obj)

        except Exception as e:
            new_failed.append(cleaned_names[idx])
            print(f"[Detector Agent] Failed parsing for {cleaned_names[idx]}: {e}")
    locations = remove_duplicates_by_id(locations)
    state["information_of_object"] = locations
    state["detect_failed_objects"] = new_failed
    
    return {
        "information_of_object": locations,
        "detect_failed_objects": new_failed,
        "messages": [AIMessage(content=json.dumps(locations, indent=2), name="detector_agent_without_llm")]
    }



async def user_input_matching_object_agent(llm, state: AgentState, config=None) -> dict:
    """ 객체 리스트(ID+name)와 사용자 입력을 받아, 객체의 ID 단위로 매칭
        예를 들어 가장 작은 물건부터 순서대로 잡아줘 라는 입력 시 
        작은 물건: obj_id_1, obj_id_2,...,이런 식 
        사용자가 물건의 이름이 아닌 모양의 특징(가장 작은거 주워줘, 가장 큰거 주워줘)같은 거 말하면 그거 찾아주는 거임"""
        
    try:
        # 1) detector가 만든 리스트(각 항목에 'id'/'name' 포함)를 가져옴
        info_list = state.get("information_of_object", []) or []
        if not info_list:
            raise ValueError("No detected objects in state.")

        # 2) 프롬프트에 넣을 ID: name 목록 생성
        all_objects_text = "\n".join(
            f"- {item['id']}: {item['name']},size:{item['area']}"
            for item in info_list
        )
        formatted_prompt = USER_INPUT_MATCH_PROMPT.replace("{all_objects}", all_objects_text)
        # 3) 사용자 입력 확보
        user_input = state.get("user_input", "").strip()
        # 4) LLM 호출
        res = llm.invoke([
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=user_input)
        ], config=config)
        res = res.content
        print("match original",res)
        res = re.sub(r"^```json\s*", "", res.strip(), flags=re.IGNORECASE)
        res = re.sub(r"```$", "", res.strip())
        res = json.loads(res) 
        print("chain of thought :", res)
        print("answer:",res["answer"])
        parsed = parse_to_obj(res["answer"])
        print("parsed")

        # 5) matched_ids / alias_mapping 파싱
        if isinstance(parsed, dict):
            matched_ids   = parsed.get("matched_ids", [])
            missing = parsed.get("missing", [])
        else:
            raise ValueError("LLM output is not a dict with matched_ids")

        if missing:
            state["detect_failed_objects"] = missing
            state["detect_feedback"] = True
            state["user_input"] = user_input
            print(state.get("user_input"))
            return{
                "detect_failed_objects": missing,
                "detect_feedback": True,
                "user_input": user_input
            }

        # 6) state에 저장
        state["matched_object_ids"] = matched_ids

        # 7) objects_user_input: info_list 중 ID가 매칭된 항목만 필터링
        objects_user_input = [
            item
            for item in info_list
            if item["id"] in matched_ids
        ]
        state["objects_user_input"] = objects_user_input

        return {
            "matched_object_ids": matched_ids, # 사용자 입력과 매칭되는 id
            "user_input":         user_input, # 사용자 입력
            "objects_user_input": objects_user_input, # 매칭된 물건의 상세 정보(넓이, 위치, 각도 등등)
            "messages": [
                AIMessage(
                    content=(
                        f"Matched IDs: {matched_ids}, "
                        f"objects_user_input IDs: {[o['id'] for o in objects_user_input]}"
                    ),
                    name="user_input_matching_object_agent"
                )
            ]
        }

    except Exception as e:
        return {
            "matched_object_ids": [],
            "user_input":         state.get("user_input",""),
            "objects_user_input": [],
            "messages": [
                AIMessage(content=f"Error: {str(e)}", name="user_input_matching_object_agent")
            ]
        }




async def simple_user_input_analyze_agent(llm, state: AgentState, config=None) -> dict:
    """
    사용자 입력을 받아 어떻게 수행할지를 LLM이 판단해서 순서를 정하는 에이전트
    단순 pick n place 뿐 아니라 더 다양한 동작에 대해서도 액션을 구체적으로 나눔
    사용자 입력인 user_input만 받음 -> 물체의 이름만 받아 토큰 단순화
    출력은 1. action 1, 2. action2, 3.~ 이런식으로 할 예정
    """
    user_input = state.get("user_input")
    objects_user_input = state.get("objects_user_input")
    objects_user_input = [obj['id'] for obj in sorted(objects_user_input, key=lambda o: o['area'])] # 사용자 입력과 관련된 물건을 미리 넓이가 작은 순서로 넣어준다.
    objects_user_input = str(objects_user_input)
    system_prompt = SIMPLE_USER_INPUT_ANALYZE_PROMPT.replace("{objects}",objects_user_input)


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

    blocks = [
        block.strip().splitlines()
        for block in res.content.strip().split('\n\n')
        if block.strip()
    ]

    # 2. 출력은 [[액션1],[액션2],[]...] 이런 형식
    cleaned_blocks = [
        [re.sub(r'^\d+\.\s*', '', line) for line in block]
        for block in blocks
    ]
    print("user_input_analyze_agent:",res.content)
    print("blocks:", cleaned_blocks)
    state["user_input_analyzed_by_LLM"] = cleaned_blocks
    return{
        "user_input_analyzed_by_LLM": cleaned_blocks
    }


############################################################################################################################################################################ action supervisor
async def analyzed_user_input_regenerate_supervisor(llm, state: AgentState, tool_node, config=None) -> dict:
    """
    user_input_analyzed_by_LLM 받아 액션 시컨스 별로 나누고 정규화 시키는 에이전트
    user_input_analyzed_by_LLM리스트를 받음 -> 리스트를 action_num번째를 뽑아서 실행
    """

    user_input_analyzed_by_LLM = state.get("user_input_analyzed_by_LLM")
    user_input_analyzed_by_LLM = user_input_analyzed_by_LLM
    print("user input analyzed by llm:",user_input_analyzed_by_LLM,"len",len(user_input_analyzed_by_LLM) )

    action_num = state.get("action_num")
    
    # 액션 다 끝나면 종료
    if action_num >= len(user_input_analyzed_by_LLM):
        state["done"] = True
        print("done")
        return{
            "done" : True
        }
    
    print("supervisor start with action num:", action_num)
    
    system_prompt = ANALYZED_USER_INPUT_REGENERATE_SIMPLE_PROMPT
    user_input = user_input_analyzed_by_LLM[action_num]
    user_input = str(user_input)
    res = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ], config=config)
    
    output = str(res.content)
    output = output.strip()  # 맨 앞뒤 공백·개행 제거
    if output.startswith("```") or output.endswith("```"):
        output = output[3:-3].strip()  # 맨 앞뒤 ```만 잘라내고 다시 trim

    keys = ("agent", "action", "target") # 각각 사용할 agnet, 수행할 action, 타겟 물체 or 장소
    parsed = [
        dict(zip(keys, (part.strip() for part in line.split(":"))))
        for line in output.strip().splitlines()
    ]

    print("parsed is:", parsed)

    #출력을 파싱 -> curr : 현재 입력, nxt : 다음스탭 입력
    for curr, nxt in zip_longest(parsed, parsed[1:]):

        if curr["agent"] == "move":
            if nxt is not None and nxt["agent"] == "grip": # move다음에 grip이면 이거로 해서 grip하기 전에 그 물체로 바로 가지 않게
                result = await move_agent(llm, state, tool_node, curr, grip = True)
            elif nxt is not None and nxt["agent"] == "release":
                print("move before release")
                result = await move_agent(llm, state, tool_node, curr, release = True)
            else:
                result = await move_agent(llm, state, tool_node,curr)

        elif curr["agent"] == "grip":
            result = await grip_agent(llm, state, tool_node,curr)
            
        elif curr["agent"] == "release":
            print(curr["target"])
            items = [item.strip() for item in curr["target"].split(',')]  # 쉼표로 나눈 뒤 양쪽 공백 제거
            print("item first:", items[0])
            state["last_gripped"].append(items[0]) # 마지막으로 잡은 물체 append하는거 근데 이거 이름만 하는거라서 나중에 위치도 업데이트 해야함
            # result = await grip_agent(llm,curr,tool_node)
            result_msg = await tool_node.ainvoke({"messages": [gripper_message("Release")]})



    state["action_num"] = action_num +1

    return await analyzed_user_input_regenerate_supervisor(llm,state,tool_node)


############################################################################################################################################################################ move agent
async def move_agent(llm, state:AgentState , tool_node, move_plan, config=None, grip = False, release = False) -> dict:
    """
    analyzed_user_input_regenerate_supervisor가 호출 시 기본적인 move와 관련된 명령(move_plan)을 수행
    목표는 정확한 좌표로 이동한 후 이동완료 신호를 보내는것
    이동할때는 다른 물건과 충돌이 없도록 일단은 수직, 수평이동으로 함
    앞서 받은 액션이 단순한 move같은 거면 llm없이 실행, 복잡한 액션이면 llm 실행
    """
    # 로봇팔 엔드툴 포즈
    tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    tool_pos = tool_pos[0].content
    tool_pos = list(map(float, eval(tool_pos)))
    print("robot_arm_attached",tool_pos)

    plan = move_plan["action"]
    target = move_plan["target"]
    last_gripped = state["last_gripped"]
    action_num = state.get("action_num")

    if grip or plan == "move to above" and not release: # 잡을 때나 놓을 때는 오프셋 적용
        height_offset = 0.33
        x_offset = -0.1
    elif release:
        print("releasing!!!!!!!!!!!!!!!!!!!!!!!!!")
        global grasp_z # 마지막으로 파지한 위치
        height_offset = grasp_z+0.02
        x_offset = 0

    else: # 그냥 이동일 때는 해당 위치로 이동
        height_offset = 0

    print("move plan:", plan)
    print("target:", target)
    print("action_num", action_num)
    objects = state.get("objects_user_input")
    print("objects is ", objects)
    print("target =", repr(target))
    # 정상적인 action이 들어왔을 때 = llm 사용 안함
    # if (grip or plan in ["move to", "move to above"]) and "home" not in str(target).strip():
    if "home" not in str(target).strip():
        print("moving before grip or release")
        obj = next(item for item in objects if item['id'] == str(target))
        if obj: # 타겟 물체가 저장되어 있다면
            print("obj is:", obj)
            if action_num == 0: # 첫번째 액션이라면
                # msg = moveL_message([obj["position"]["x"]-0.1,obj["position"]["y"],obj["position"]["z"]+height_offset,0])
                msg = moveL_rpy_message([obj["position"]["x"]+x_offset,obj["position"]["y"],obj["position"]["z"]+height_offset,tool_pos[3],tool_pos[4],tool_pos[5]])
                print("msg is:", msg)
                result_msg = await tool_node.ainvoke({"messages": [msg]})
                print("result move", result_msg)
                if not result_msg:
                    print("경로로 못감")

            else: # 첫번째 액션이 아니라면
                if obj["id"] in last_gripped: # 이전에 잡아서 옮겨서 release까지 끝낸 객체를 다시 잡는다면
                    print(f"{obj['id']}를 이전에 이미 집은 적이 있습니다. 옮겼던 위치로 이동합니다.")

                    """이동 후 탐지 -> find_closest_detection로 그 물체 확인, 만약 발견 하면 그 위치로 이동, 발견 못하면 피드백 루프"""
                    # msg = find_object_3d_properties_message(obj["name"])
                    # result_msg = await tool_node.ainvoke({"messages": [msg]})
                    # 여기에 탐지 실패 코드 넣을것
                else:
                    # msg = moveL_message([obj["position"]["x"]-0.1,obj["position"]["y"],obj["position"]["z"]+height_offset,0])
                    msg = moveL_rpy_message([obj["position"]["x"]+x_offset,obj["position"]["y"],obj["position"]["z"]+height_offset,tool_pos[3],tool_pos[4],tool_pos[5]])
                    result_msg = await tool_node.ainvoke({"messages": [msg]})
                    print("result move2", result_msg)
                    if not result_msg:
                        print("경로로 못감")
        else:
            print("no match obj")
            # if target == 좌표:
            #   msg = moveL_message(좌표 넣을것)
            # #만약 target이 좌표가 아니라 객체 이름인데 감지된게 아니라면    
            # vlm단계부터 피드백 하는 코드 넣을것

    else: # 이상한 action이 들어왔을 때 -> llm 사용
        # llm 사용 코드 넣을 것
        print("go to home pose")
        # tool_pos[0],tool_pos[1],tool_pos[2] = 0.4,0.15,0.3
        # msg = moveL_rpy_message(tool_pos)

        result_msg = await tool_node.ainvoke({"messages": [robot_move_home_message()]})
        msg = moveL_message([0.4,0.15,0.45,0])
        print("msg is:", msg)
        result_msg = await tool_node.ainvoke({"messages": [msg]})
        return





    # system_prompt = MOVE_AGENT_PROMPT
    # # LLM 메시지 구성
    # msgs = [
    #     SystemMessage(content=system_prompt),
    #     HumanMessage(content=move_plan)
    # ]
    
    # # LLM 응답 호출
    # res = llm.invoke(msgs, config=config)


    # print("move agent output:", res.content)
    return

############################################################################################################################################################################ grip agent
async def grip_agent(llm, state:AgentState , tool_node, move_plan, config=None) -> dict:
    """
    물체 잡으러 가는 자세, 잡는 파지점, 잡아서 올리는 거 까지 책임지는 에이전트
    입력 : 잡을 물체 이름
    내부 추론 : 물체 이름 GroundedSAM2에 넣어 정보 얻음-> 피드백 구조 만들 것!,
    물체 정보 LLM에 프롬포트와 함께 전달
    단계를 2단계로 나눌거임
    1. 물체 잡기 전까지 파지점과 잡는 거 까지로
    2. 물체 잡기 시도 후 실패, 성공 유무 확인
    3. 실패 시 -> return에 이 에이전트 다시 호출 -> 최대 3번 시도 후 안되면 pass
    성공 시 -> 안전 높이까지 올린 후 return
    """
    # 로봇팔 엔드툴 포즈
    tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    tool_pos = tool_pos[0].content
    tool_pos = list(map(float, eval(tool_pos)))
    print("robot_arm_attached",tool_pos)



    target_object = move_plan["target"]
    target_name = re.sub(r'_[0-9]+$', '', target_object) # 타겟에서 id뒤에있는 _숫자 이거 없애는 코드
    print("target_objext is:",target_name)
    # 객체 탐지 -> 수직하게 봐서 정확한 위치 확인
    msg = find_object_3d_properties_message(target_name)
    detect_res = await tool_node.ainvoke({"messages": [msg]})

    # 탐지 못할 경우의 피드백 루프 넣을것
    detect_msg = detect_res["messages"][0].content
    print("detect_msg is:", detect_msg)
    parsed_raw = parse_to_obj(detect_msg) # 이후 파싱
    target_obj = ensure_list_of_dicts(parsed_raw)
    # det_list 안에 아직 문자열(JSON) 요소가 남아 있을 경우, dict로 파싱
    target_obj = [json.loads(item) if isinstance(item, str) else item for item in target_obj]
    # print("det list is:", target_obj[0])
    target_obj = target_obj[0] # 탐지된 물체 정보


    objects = state.get("objects_user_input") # 물체 정보들이 저장된 리스트
    last_gripped = state["last_gripped"] # 이전에 옮긴 적이 있던 객체 리스트

    obj = next(item for item in objects if item['id'] == str(target_object))
    if obj: # 타겟 물체가 저장되어 있다면
        if obj["id"] in last_gripped: # 이전에 잡아서 옮겨서 release까지 끝낸 객체를 다시 잡는다면
            print(f"{obj['id']}를 이전에 이미 집어서 옮긴 적이 있습니다. 옮긴 위치 기준으로 객체를 감지합니다.")
            # 여기에 이전에 옮긴 위치 정보를 받아오는 코드 추가 -> release_agent 에서 업데이트 해주면 될거같음
            # closest_obj = find_closest_detection(detect_msg, obj["position"]) # 옮긴 기존 물체와 가장 가까운 물체를 선택, 리턴은 물체 모든 정보



        else:
            closest_obj = find_closest_detection(detect_msg, obj["position"]) # 초반과 가장 가까운 물체를 선택, 리턴은 물체 모든 정보

######################################## VLM 영역 ############################################3
    capture_msg = capture_image_as_jpg_message(0)
    capture_res = await tool_node.ainvoke({"messages": [capture_msg]})

    resp_msgs = capture_res.get("messages", [])
    props = None
    if resp_msgs: # 사진을 base64로 잘 반환 했다면
        content = resp_msgs[0].content
        if content.startswith("data:image/jpg;base64,"):
            props = content
    a = re.sub(r'_[0-9]+$', '', target_object) # 타겟에서 id뒤에있는 _숫자 이거 없애는 코드
    print("target is is is is:", a)
    prompt = GRIPPER_VLM_SHAPE_AGENT_PROMPT.replace("{object}",a) # grip 하기 전 이미지 상으로 객체 모양 및 파지 방향 결정 프롬포트
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": props}}
    ])
    vlm_res = llm.invoke([msg], config=config)
    vlm_text = vlm_res.content.strip()
    print("vlm출력이 이겁니다ㅣ:", vlm_text)

    features = {
        line.split(':', 1)[0].strip(): line.split(':', 1)[1].strip()
        for line in vlm_text.splitlines() if ':' in line
    }
    print("features is:", features)
    print("featu:",features["shape"])


    # 타겟 좌표계 변경 (엔드툴에 카메라 달았다는 가정 하에)
    target_obj["position"][0],target_obj["position"][1],target_obj["rotation_degree"] = tool_to_world(tool_pos, target_obj["position"][0],target_obj["position"][1],target_obj["rotation_degree"]) # 월드 좌표계로 전환
 ##################################### vlm 출력 파싱? 해서 잡을 방향 변수에 지정 ##################
    grip = None

    if features["shape"] == "cylinder" and features["orientation"] == "lying": # 누운 실린더면
        grip = "vertical"

    elif features["shape"] == "rectangular_prism" or features["shape"] == "cube":
        # if det_list["position"][2] >= det_list["dimensions"][1]: # 박스인데 높이보다 단축이 더 짧을 때
        grip = "vertical"
        # else:
        #     grip = "side"
    else: 
        grip = "side"

    print("grip is", grip)

    ############################################## 안전장치 #########################################
    if target_obj["position"][2] <= 0.04: # 4cm보다 물체 낮으면 그냥 멈춤
        return

############################################### 수직방향 그립 액션 ##################################
    if grip == "vertical":
        print("수직으로 잡는 동작 실행")



        pos1 = [target_obj["position"][0],target_obj["position"][1],target_obj["position"][2]+0.05,target_obj["rotation_degree"]] # 먼저 5cm 위로 감
        pos2 = [target_obj["position"][0],target_obj["position"][1],target_obj["position"][2]-0.015,target_obj["rotation_degree"]] # 그다음 내려감
        await tool_node.ainvoke({"messages": [moveL_message(pos1)]})
        await tool_node.ainvoke({"messages": [moveL_message(pos2)]})
        await tool_node.ainvoke({"messages": [gripper_message("Grab")]}) # 그 후에 잡음

        pos3 = [target_obj["position"][0],target_obj["position"][1],target_obj["position"][2]+0.1,target_obj["rotation_degree"]] # 그다음 10cm위로 올라감
        await tool_node.ainvoke({"messages": [moveL_message(pos3)]})

########################################## 수평 방향 그립 액션 ######################################
    else:
        print(" 옆면 잡는 동작 실행")
        prompt = GRIPPER_VLM_GRASP_DETECT_AGENT_PROMPT.replace("{object}",a) # grip 하기 전 이미지 상으로 목표 대상 주변 다른 장애물의 이름 추출
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": props}}
        ])
        vlm_res = llm.invoke([msg], config=config)
        print("vlm 파지 할때 출력", vlm_res)
        vlm_res = vlm_res.content
        res = json.loads(vlm_res)
        vlm_text = res["answer"]
        # vlm_text = vlm_res.content.strip()
        
        print("vlm 파지 이름 출력이 이겁니다ㅣ:", vlm_text)
        # names = [name.strip() for name in vlm_text.split(",") if name.strip()]
        # cleaned_names = [n.strip().strip('"').strip("'") for n in names if n.strip()]
        cleaned_names = [n.strip().strip('"').strip("'") for n in (vlm_text if isinstance(vlm_text, list) else vlm_text.split(","))]

        print("cleaned_names: ", cleaned_names)
        detect_msg = find_object_3d_properties_message(cleaned_names)
        detect_res = await tool_node.ainvoke([detect_msg])
        print("detected is ",detect_res[0].content)
        parsed_raw = parse_to_obj(detect_res[0].content) # 이후 파싱
        det_list = ensure_list_of_dicts(parsed_raw)
        

        det_list = update_world_coordinates(tool_pos, det_list) # 월드 좌표 기준으로 치환
        print("detlist is ",det_list)

####################################### 파지 방향 벡터 및 rpy 추출 ####################################
        target = normalize_object(target_obj, threshold = 0.0) # threshold = 탐지 제한 높이, 기본값:0.02 = 2cm
        print("타겟:",target)
        normalized_obstacles = [
            normalize_object(s if isinstance(s, dict) else json.loads(s))
            for s in det_list
        ]
        normalized_obstacles = [item for item in normalized_obstacles if item is not None]
        print("노말라이즈드", normalized_obstacles)
        if normalized_obstacles != None:
            dx, dy, status = compute_best_approach_direction(target, normalized_obstacles, offset=0.005) # 충돌 안하는 방향으로의 방향벡터 추출 함수
            # print("거의다 했다",find_wide_open_spans(target, normalized_obstacles, offset=0.005, threshold=100))
            print("리설트",dx,dy,status)
        else:
            dx,dy = 0,1
        import random
        async def check_ik(pos):
            pos_json = json.dumps(pos)   
            res = await tool_node.ainvoke([check_IK_message(pos_json)])
            return res[0].content == "success"

################################ pos1 IK 재시도 (순차 보정) + tilt 재시도 추가 #########################
        
        tilt = -15
        dx_init, dy_init = dx, dy
        initial_tilt = tilt  # -20였던 초깃값
        tilt_angles = list(range(-15, -41, -5)) # -15 ~ -40도로 시도
        max_offsets = 15
        offsets = [ -0.02 + i * (0.04 / (max_offsets - 1)) for i in range(max_offsets) ]

        success = False

        for t in tilt_angles:
            # print(f"=== tilt={t}° 시도 ===")
            for trial, off in enumerate(offsets, start=1):
                # print(f"  --- pos1 IK 재시도 {trial}/{max_offsets} (offset={off:.4f}) ---")
                dx = dx_init + off
                dy = dy_init + off
                # 방향벡터·기울기 계산
                rpy, _ = get_alignment_and_compensation(dx, dy, t)
                # 웨이포인트 계산
                waypoint = compute_waypoints(
                    dx, dy,
                    target_obj["position"],
                    target_obj["dimensions"][1] / 2,
                    base_offset=0.1
                )
                pos1 = waypoint["start_point"] + list(rpy)

                if await check_ik(pos1):
                    print("    ✓ pos1 IK 성공")
                    tilt = t   # 사용된 tilt 값으로 업데이트
                    success = True
                    break

            if success:
                break

        if not success:
            print("ERROR: 모든 tilt & offset 재시도에서 pos1 IK 실패, 동작 중단")
            # 마지막으로 초기값 복원
            tilt = initial_tilt

            rpy, _ = get_alignment_and_compensation(dx_init, dy_init, tilt)
            if target_obj["dimensions"][1] / 2>= 0.04:
                radius = target_obj["dimensions"][1] / 2
            else:
                radius = 0

            waypoint = compute_waypoints(
                dx_init, dy_init,
                target_obj["position"],
                radius,
                base_offset=0.07
            )
            pos1 = waypoint["start_point"] + list(rpy)
            print("dx,dy:",dx_init,dy_init)



        # 2) pos1으로 이동
        msg1 = moveL_rpy_message(pos1)
        movel_res1 = await tool_node.ainvoke([msg1])
        print("movel pos1:", movel_res1)

        # 3) 그리퍼 평형 및 rpy 갱신
        new_rpy = await tool_node.ainvoke([gripper_rotate_message()])
        rpy_list = json.loads(new_rpy[0].content)
        rpy = [float(x) for x in rpy_list]

        # 4) pos2 생성 및 IK 체크 (루프 밖에서 한 번만)
        pos2 = waypoint["target_point"] + list(rpy)
        print("최종 pos2:", pos2)

        # 추가: 잡았을 때의 오프셋
        global grasp_z
        grasp_z = pos2[2]

        # 5) pos2으로 이동 및 그립
        msg2 = moveL_rpy_message(pos2)
        movel_res2 = await tool_node.ainvoke([msg2])
        print("movel pos2:", movel_res2)

        grip_res = await tool_node.ainvoke([gripper_message("Grab")])
        time.sleep(0.5)
        print("그립 완료:", grip_res)

        # 로봇팔 엔드툴 포즈
        tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
        tool_pos = tool_pos[0].content
        tool_pos = list(map(float, eval(tool_pos)))

        tool_pos[2] += 0.2
        msg = moveL_rpy_message(tool_pos)
        result_msg = await tool_node.ainvoke({"messages": [msg]})

        return


async def release_agent(llm, state:AgentState , tool_node, move_plan, config=None) -> dict:
    """
    그리퍼 릴리즈 하는 에이전트
    보통은 llm 없이 그냥 지정 위치가 맞는지 확인 후 놓는거로
    에러상황 발생시 llm호출하는 거로
    """



