from typing import TypedDict, List, Optional, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json
from prompts import * # 프롬프트 불러오기
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
from itertools import zip_longest
import time
import open3d as o3d




############################## 상태 정의 (state_schema) ##########################
# 상태 정의 (state_schema)
class AgentState(TypedDict):
    user_input: str # 사용자 입력을 저장하는 필드
    divided_user_input:  Optional[List[str]] = [] # 나눠진 사용자 입력 저장 필드
    all_object_name: List[str] = []  # 기본값을 빈 리스트로 설정
    object_user_want_name: List[str] = []  # 기본값을 빈 리스트로 설정
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
    "all_object_name": [],
    "object_user_want_name": [],
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

debug = True
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
            "object_user_want_name": [],
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
        text = parse_llm_non_json2json_output_for_vlm(res.content)
        print(f"[VLM Agent] feedbacked object names: {text}")

        # 4. Parse response: always comma-separated list
        names = text["answer"]
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
    # res = re.sub(r"^```json\s*", "", res.strip(), flags=re.IGNORECASE)
    # res = re.sub(r"```$", "", res.strip())
    print("물체 이름 추출 파싱 전 디버그",res)
    # res = json.loads(res)
    res = parse_llm_non_json2json_output_for_vlm(res)
    print("물체 이름을 추출 결과:", res)

    names = res["answer"]
    print("물체 이름을 추출하는 사고과정:", res["reasoning"])
    print("사용자 입력과 관련된 탐지된 객체들", names)

    #######################################################################################################
    # 5. Update state
    state["object_user_want_name"] = names
    state["detect_failed_objects"] = []
    state["all_object_name"] = text
    
    print(f"[VLM Agent] State updated: object_user_want_name={names}")

    # 6. Return unified envelope with debug messages
    return {
        "object_user_want_name": names,
        "all_object_name": text,
        "detect_failed_objects": [],
        "messages": [
            {"role": "assistant", "content": f"Detected objects: {text}"}
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
    object_names = state.get("object_user_want_name", []) if not detect_failed else detect_failed
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
                    
                    print(f"[INFO] Parsed {len(raw_list_strs)} items for '{name}'")
                    
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
                        # print(f"[SUCCESS] '{name}' processed successfully. Found {len(cleaned_response)} valid items.")
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

    # 피드백 로직 수정
    if not detect_feedback and failed_objects:  # raw가 아닌 failed_objects 체크
        print("[Detector Agent] Some objects failed detection, marking as detect_failed_objects")
        state["detect_failed_objects"] = failed_objects  # 실패한 객체들만
        state["detect_feedback"] = True
        return {
            "information_of_object": locations,
            "detect_failed_objects": failed_objects,
            "detect_feedback": True,
            "messages": [
                AIMessage(
                    content=f"Failed to detect 3D properties for {failed_objects} after {max_retries} attempts. Successfully detected: {len(cleaned_names) - len(failed_objects)} objects.",
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
            x, y, z = [round(c, 3) for c in pos]

            x,y,_ = tool_to_world(tool_pos, x,y,0) # 월드 좌표계로 전환

            obj = {
                "id": obj_id,
                "name": name,
                "position": {"x": x, "y": y, "z": z},
                "area": det["area"],
            } # 7_24 변경점: rotation degree, dimention 제거
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
        # print("match original",res)
        res = parse_llm_non_json2json_output_for_input_matching(res) # LLM이 출력한 json이 아닌 str로 출력되므로 파싱
        print("***user_input_matching_agent parsed*** :", res)
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
            print("grasp_z is ", grasp_z)
        elif curr["agent"] == "release":
            print(curr["target"])
            items = [item.strip() for item in curr["target"].split(',')]  # 쉼표로 나눈 뒤 양쪽 공백 제거
            print("item first:", items[0])
            state["last_gripped"].append(items[0]) # 마지막으로 잡은 물체 append하는거 근데 이거 이름만 하는거라서 나중에 위치도 업데이트 해야함
            # result = await grip_agent(llm,curr,tool_node)
            await tool_node.ainvoke({"messages": [gripper_message("Release")]})
            time.sleep(0.5) # release는 바로 안되니까 0.5초 정도 기다림
            tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
            tool_pos = tool_pos[0].content
            tool_pos = list(map(float, eval(tool_pos)))
            tool_pos[2] += 0.2
            await tool_node.ainvoke({"messages": [moveL_rpy_message(tool_pos)]})
            tool_pos[0], tool_pos[1] = 0.45, 0.15 # release 후에는 툴을 원점으로 이동
            # await tool_node.ainvoke({"messages": [moveL_rpy_message(tool_pos)]})
            await tool_node.ainvoke({"messages": [robot_move_home_message()]}) # 툴 포즈 다시 받아오기



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
        height_offset = 0.33 # 물체 위 30cm에서 바라봄
        x_offset = -0.1 # 카메라 오프셋 적용 (yaw가 180이라는 가정 즉, 정방향)
    elif release:
        print("releasing!!!!!!!!!!!!!!!!!!!!!!!!!")
        global grasp_z # 마지막으로 파지한 위치
        height_offset = grasp_z+0.02 # 마지막으로 파지한 위치에서 2cm위에서 놓음
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
    -> 리턴 : 타겟 정보 ( pos, mask, mask_shape)

    1. 타겟 포함 주변 물체 vlm으로 감지해서 옆면이 보간된 point cloud를 생성 및 anygrasp 서버 실행해 파지점 추출
    2. 파지점의 rpy를 분석 후 수직 그랩이 많으면 수직으로, 수평 그랩이 많으면 수평으로 취급 -> 이건 테스트 해봐야 할듯 아니면 윗면 넓이와 높이 보고 추측하던가
    3. 수평이면 각도 제한을 시도(먼저 수평면으로부터 10~20도 기울어진거-> 없으면 20~30-> 없으면 30~40 이런순서로.. 절대 -각도가 안나오게)-> 만약 정 없으면 그나마 되는것중 점수 높은거로
    성공 시 -> 안전 높이까지 올린 후 return
    실패 시 -> 일단 실패 감지 로직을 만들어야 함
    """

    ##################### State로부터의 값  #################
    objects = state.get("objects_user_input") # 물체 정보들이 저장된 리스트
    last_gripped = state["last_gripped"] # 이전에 옮긴 적이 있던 객체 리스트
    target_object = move_plan["target"] # supervisor로부터 전달받은 타겟 물체 id
    all_object_names = state.get("all_object_name")
    print("all object gripper go :",all_object_names)


    ########################################################


    ################# 로봇팔 엔드툴 포즈 추출 ################
    tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    tool_pos = tool_pos[0].content
    tool_pos = list(map(float, eval(tool_pos)))
    print("엔드툴 pos : ",tool_pos)
    ########################################################


    ################ 전역 변수 #############################
    global grasp_z # 마지막으로 파지한 위치


    #######################################################
    

#### 1. 타겟 포함 주변 물체 vlm으로 감지해서 옆면이 보간된 point cloud를 생성 및 anygrasp 서버 실행해 파지점 추출 #################################
    
    # 1.1 이미지 캡쳐 및 vlm 전송
    capture_msg = capture_image_as_jpg_message(0)
    capture_res = await tool_node.ainvoke({"messages": [capture_msg]})

    resp_msgs = capture_res.get("messages", [])
    props = None

    if resp_msgs: # 사진을 base64로 잘 반환 했다면
        content = resp_msgs[0].content
        if content.startswith("data:image/jpg;base64,"):
            props = content

    # 입력으로 전달받은 타겟 id를 추출함
    target_name = re.sub(r'_[0-9]+$', '', target_object) # 타겟에서 id뒤에있는 _숫자 이거 없애는 코드 (GroundedSAM2에는 id를 포함해서 넣으면 못찾으니까 이름을 넣음)
    print("타겟 이름은:",target_name)

    prompt = GRIPPER_VLM_GRASP_DETECT_AGENT_PROMPT.replace("{object}",str(all_object_names)) # grip 하기 전 이미지 상으로 목표 대상 주변 다른 장애물의 이름 추출
    msg = HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": props}}
    ])
    vlm_res = llm.invoke([msg], config=config)
    print("vlm의 파지점 주변 물체들 감지 raw 출력 ", vlm_res)
    vlm_res = vlm_res.content
    res = parse_llm_non_json2json_output_for_vlm(vlm_res)
    print("vlm의 파지점 주변 물체들 감지 파싱된 출력:", res)
    vlm_text = res["answer"] # 출력은 2가지로 나옴 1. reasoning(추론단계 -> 이게 있어야 성능 향상), 2. answer -> 이게 최종 출력
    print("vlm의 파지점 주변 물체들 감지 출력:", vlm_text)


    # vlm 출력에서 쓸데없는 기호들 제거하는 코드
    cleaned_names = [
        n.strip()
        .removeprefix("```json")
        .removeprefix("```")
        .removesuffix("```")
        .strip('"')
        .strip("'")
        for n in (
            sum([s.split(",") for s in vlm_text], [])
            if isinstance(vlm_text, list)
            else vlm_text.split(",")
        )
    ]
    print("cleaned_names: ", cleaned_names)
    target_name = "can"  # 네가 잡으려는 물체

    if target_name in cleaned_names:
        cleaned_names.remove(target_name)
        cleaned_names.append(target_name)


######################################################################################################3
    max_retries = 3
    all_results = []  # 모든 객체의 결과를 합칠 리스트
    failed_objects = []  # 감지에 실패한 객체들

    for name in cleaned_names:
        print(f"[그리퍼 Agent] Processing object: '{name}'")
        
        attempt = 0
        raw = ""
        object_success = False  # 현재 객체의 성공 여부
        
        while attempt < max_retries:
            print(f"[그리퍼 Agent] '{name}' attempt {attempt + 1}/{max_retries}")
            
            # 단일 객체로 호출
            detect_msg = find_object_3d_properties_message([name])
            detect_res = await tool_node.ainvoke([detect_msg])
            
            first_msg = detect_res[0]
            raw = first_msg.content or ""
            
            if raw:
                print(f"[그리퍼 Agent] Received non-empty response for '{name}'")
                
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
                    
                    print(f"[INFO] Parsed {len(raw_list_strs)} items for '{name}'")
                    
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
                        # print(f"[SUCCESS] '{name}' processed successfully. Found {len(cleaned_response)} valid items.")
                        break  # 성공했으면 재시도 루프 탈출
                    else:
                        print(f"[WARNING] '{name}' - No valid items after filtering")
                        
                except Exception as e:
                    print(f"Top-level JSON parse failed for '{name}':", e)
            
            # 빈 응답이거나 파싱 실패시 재시도
            attempt += 1
            if attempt < max_retries:
                print(f"[그리퍼 Agent] Retrying '{name}' in 0.1 seconds...")
                await asyncio.sleep(0.1)
        
        # 재시도 후에도 실패했으면 실패 목록에 추가
        if not object_success:
            failed_objects.append(name)
            print(f"[FAILED] '{name}' could not be detected after {max_retries} attempts")

    # 최종 결과
    raw = all_results  # 모든 객체의 결과가 합쳐진 리스트
    print(f"[그리퍼 Agent] Total processed items: {len(raw)}")
    print(f"[그리퍼 Agent] Successfully detected: {len(cleaned_names) - len(failed_objects)}/{len(cleaned_names)} objects")
    if failed_objects:
        print(f"[Detector Agent] Failed objects: {failed_objects}")

######################################################################################################3
    parsed_raw = parse_to_obj(raw) # 이후 파싱
    print("parsed raw is ", parsed_raw)
    det_list = ensure_list_of_dicts(parsed_raw) # 리스트 형식으로 출력된 물체 정보들 [ {obj1 id:, obj name:, ....}, {....}]
    

    # 월드 좌표 기준으로 Groundedsam2로 추출한 물체 좌표 변환
    det_list = update_world_coordinates(tool_pos, det_list) 
    print("감지된 물체의 정보들: ",det_list)

    # 이미 이전에 감지한 타겟위치와 비슷한 위치의 물체를 타겟으로 지정 (같은 물체가 여러개 있을 경우를 대비), 타겟 id를 통해 위치 추출해서 같은물체 여러개 있어도 정확한 위치 기준으로 탐지함
    obj = next(item for item in objects if item['id'] == str(target_object))
    if obj: # 타겟 물체가 저장되어 있다면
        if obj["id"] in last_gripped: # 이전에 잡아서 옮겨서 release까지 끝낸 객체를 다시 잡는다면
            print(f"{obj['id']}를 이전에 이미 집어서 옮긴 적이 있습니다. 옮긴 위치 기준으로 객체를 감지합니다.")
            # 여기에 이전에 옮긴 위치 정보를 받아오는 코드 추가 -> release_agent 에서 업데이트 해주면 될거같음
            # closest_obj = find_closest_detection(detect_msg, obj["position"]) # 옮긴 기존 물체와 가장 가까운 물체를 선택, 리턴은 물체 모든 정보

        else:
            closest_obj = find_closest_detection(det_list, obj["position"]) # 초반과 가장 가까운 물체를 선택, 리턴은 물체 모든 정보

    print("gripper's closest obj", closest_obj, "gripper's closest obj end")

    filtered = remove_target_from_detections(detections=det_list, target_obj=closest_obj)
    print("filtered is:", filtered)


######################################################## 마스크  추출 및 point cloud 추출 ##############################3
    mask_base64_list = []
    mask_shape_list = []
    obj_height_list = []

    # 타겟 제외 나머지의 마스크와 높이 추출
    for obj in filtered:
        mask_base64_list.append(obj['mask_base64'])
        mask_shape_list.append(list(obj['mask_shape']))  # mask_shape가 이미 list면 굳이 변환 안 해도 됨
        obj_height_list.append(obj['position'][2])       # position[2] == z축 높이

    # 타겟의 마스크와 높이 추출해서 맨 뒤로 보냄
    mask_base64_list.append(closest_obj['mask_base64'])
    mask_shape_list.append(list(closest_obj['mask_shape'])) 
    obj_height_list.append(closest_obj['position'][2])   
    
    
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
    print("cloud res is ",cloud_res[:100])
    try:
        cloud_res = json.loads(cloud_res)  # 리스트 안의 문자열들
        target_boundary_info = cloud_res["target_boundary_info"]

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
    max_retries = 3
    attempt = 0
    try:
        print("target side pts cloud is ", target_boundary_info[:200])
        print(f"[anygrasp Agent] Tool call attempt {attempt + 1}/{max_retries}")

        grasp_msg = anygrasp_message(cloud_res,target_boundary_info, debug)

        grasp_res = await tool_node.ainvoke([grasp_msg])
        # print("grasp res is : ",grasp_res[0].content[:2000])

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
        # print("gg is ", gg[0:5])
        target_x = closest_obj["position"][0] # 타겟 중심 위치
        target_y = closest_obj["position"][1]

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

            # x,y,w = tool_to_world(tool_pos,target_x,target_y, 0) # 월드 좌표계로 전환 , x,y 위치는 GroundedSAM2로 구했음 (캔같은 원통형은 이거로 하고 박스는 위에 주석으로 할것)
            # print("x,y in sam", x,y)
            x = target_x
            y = target_y
            # if closest_obj["position"][2] >= 0.12:
            #     grasp_height = closest_obj["position"][2] - 0.04
            # else:
            grasp_height = closest_obj["position"][2]

            g = None
            pos1 = None  
            approach_pos = None

            # 1단계: 그래스프 포인트들을 기울기에 따라 분류
            low_tilt_grasps = []  # x도 미만 (수평에 가까운)
            high_tilt_grasps = [] # x도 이상 (수직에 가까운)
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
                
                if abs(tilt_from_xy) <= tilt_deg_threshold:
                    low_tilt_grasps.append(grasp_data)
                else:
                    high_tilt_grasps.append(grasp_data)

            print(f"수평에 가까운 그래스프 (≤{tilt_deg_threshold}°): {len(low_tilt_grasps)}개")
            print(f"수직에 가까운 그래스프 (>{tilt_deg_threshold}°): {len(high_tilt_grasps)}개")


            # 2단계: 우선순위 결정 및 높이 오프셋 설정
            if len(low_tilt_grasps) >= len(high_tilt_grasps):
                primary_list = low_tilt_grasps
                secondary_list = high_tilt_grasps
                # 수평(기울기 작음): 4cm 아래 (단, 최소 높이 보장)
                primary_grasp_height = grasp_height - 0.045 if grasp_height >= 0.11 else grasp_height - 0.02
                # 수직(기울기 큼): 3cm 아래 (단, 최소 높이 보장)
                secondary_grasp_height = grasp_height - 0.01 if grasp_height >= 0.09 else grasp_height - 0.01
            else:
                primary_list = high_tilt_grasps
                secondary_list = low_tilt_grasps
                # 수직(기울기 큼): 3cm 아래 (단, 최소 높이 보장)
                primary_grasp_height = grasp_height - 0.01 if grasp_height >= 0.09 else grasp_height - 0.01
                # 수평(기울기 작음): 4cm 아래 (단, 최소 높이 보장)
                secondary_grasp_height = grasp_height - 0.045 if grasp_height >= 0.11 else grasp_height - 0.02

            async def try_grasp_list(grasp_list, list_name, adjusted_height):
                """그래스프 리스트를 시도하고 성공한 그래스프 데이터를 반환"""
                print(f"\n=== {list_name} 시도 중 ===")
                
                for grasp_data in grasp_list:
                    i = grasp_data['grasp']
                    rpy_deg = grasp_data['rpy_deg']
                    tool_z = grasp_data['tool_z']
                    tilt_from_xy = grasp_data['tilt']
                    
                    print(f"기울기 {tilt_from_xy:.2f}° 그래스프 시도 중...")
                    
                    if tilt_from_xy > -0.1: # 기울기가 역방향이면 건너뛰기
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

            # 먼저 primary 리스트 시도
            if len(primary_list) > 0:
                successful_grasp, g, adjusted_height  = await try_grasp_list(primary_list, "우선순위 그래스프 리스트",primary_grasp_height )

            # primary 리스트가 모두 실패하면 secondary 리스트 시도
            if successful_grasp is None and len(secondary_list) > 0:
                print(f"\n우선순위 리스트 실패, 백업 리스트로 전환...")
                successful_grasp, g, adjusted_height = await try_grasp_list(secondary_list, "백업 그래스프 리스트", secondary_grasp_height)

            if successful_grasp is None:
                print("\n❌ 모든 그래스프 포인트에서 IK 체크 실패")
                # 여기서 에러 처리 또는 다른 대안 로직
                raise Exception("사용 가능한 그래스프 포인트가 없습니다.")
            else:
                print(f"\n✅ 최종 선택된 그래스프 기울기: {successful_grasp['tilt']:.2f}°")
                
                try:
                    # 이후 실제 동작 실행
                    tilt_from_xy = successful_grasp['tilt']
                    tool_z = successful_grasp['tool_z']
                    pos1 = [x, y, adjusted_height, *successful_grasp['rpy_deg']]
                    
                    # 만약 접근 위치의 z가 x 이하면 +5cm 해줌
                    offset = -0.08 * tool_z
                    if pos1[2] + offset[2] <= 0.11:
                        height_offset = 0.05
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
                    
                    resmove1 = await tool_node.ainvoke([moveL_rpy_message(approach_pos1)])
                    print("resmove1 is", resmove1)
                    
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
                    resmove2 = await tool_node.ainvoke([moveL_rpy_message(approach_pos2)])
                    print("resmove2 is", resmove2)

                    # 최종 목표 위치 이동 (정렬된 RPY 적용)
                    target_pos = [pos1[0], pos1[1], pos1[2], *new_rpy]
                    await tool_node.ainvoke([moveL_rpy_message(target_pos)])
                    print("movel pos1:", target_pos)

                    grip_res = await tool_node.ainvoke([gripper_message("Grab")])
                    time.sleep(0.5)
                    print("그립 완료:", grip_res)
                    
                    # 마지막으로 잡은 높이 저장
                    grasp_z = pos1[2]

                    target_pos2 = [pos1[0], pos1[1], pos1[2] +0.1, *new_rpy]
                    await tool_node.ainvoke([moveL_rpy_message(target_pos2)])
                    print("movel pos2:", target_pos2)

                except Exception as e:
                    print("robot error is", e)

        except Exception as e:
            print("robot error is", e)
        
    except Exception as e:
        print("robot error is", e)



async def release_agent(llm, state:AgentState , tool_node, move_plan, config=None) -> dict:
    """
    그리퍼 릴리즈 하는 에이전트
    보통은 llm 없이 그냥 지정 위치가 맞는지 확인 후 놓는거로
    에러상황 발생시 llm호출하는 거로
    """



