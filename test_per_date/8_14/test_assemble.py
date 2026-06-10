from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Dict, Any
import json
from utils import * # 파싱, MCP 설정, 액션 포맷팅 같은 함수 불러오기
import asyncio
import open3d as o3d
import time
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# import logging

# #  루트 로거에 달려 있는 기존 핸들러 모두 제거
# logging.getLogger().handlers.clear()

# #  전파 방지 (중복 로깅 방지)
# logging.getLogger().propagate = False

# #  기본 로그 설정 복구 (필요에 따라 파일 또는 콘솔 설정)
# logging.basicConfig(level=logging.WARNING)  # WARNING 이상 로그만 출력

from kokoro_tts import KokoroEnglishTTS
############################## 상태 정의 (state_schema) ##########################
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
## 뉴로메카
llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key="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")

## 내꺼
# llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key="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")

## 구글 제미나이
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key="AIzaSyCquWt4KjYq0hCveRvEVikNssifzKa3ZYc")


# openrouter 사용
# llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1",model="mistralai/mistral-small-3.2-24b-instruct:free", temperature=0, api_key="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") 


#arlaili llm 사용
# llm = ChatOpenAI(base_url="https://api.arliai.com/v1",model="Gemma-3-27B-ArliAI-RPMax-v3", temperature=0, api_key="efe3eb2a-20b3-4d38-b657-5cb911cd09f8") 


import warnings

# 특정 경고만 억제
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

# 또는 카테고리별 억제
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

tts = KokoroEnglishTTS()
tts.start_initialization()



# def main():
#     tts = KokoroEnglishTTS()
#     if not tts.initialize_pipeline():
#         return

#     long_text = "System starting. Initializing modules. Please wait. Done. dararsdradsfaasa ."

#     done = speak_auto_sequential(tts, long_text)

#     print("➡ 코드 즉시 실행")
#     for i in range(5):
#         print(f"Working {i}")
#         # time.sleep(0.2)

#     done.wait()
#     print("✅ 모든 재생 완료")











# 상태 정의 (state_schema)
class AgentState(TypedDict):
    user_input: str # 사용자 입력을 저장하는 필드
    divided_user_input:  Optional[List[str]] = [] # 나눠진 사용자 입력 저장 필드
    all_objects_name: List[str] = []  # 기본값을 빈 리스트로 설정

initial_state = {
}
# object_names = [ "white box", "black box", "can" ] # 마지막에 잡을 대상 넣으면 됨.
object_names = ["white box", "blue silver can",  "black cube" ]
target_name = "blue silver can"  # 네가 잡으려는 물체

if target_name in object_names:
    object_names.remove(target_name)
    object_names.append(target_name)

print("타겟 물건 이름",target_name)

debug = False

def add_shape_from_id(objects, name_shape_map=None):
    """
    객체 리스트에 'shape' 항목을 추가합니다.
    객체의 'id'에 포함된 키워드를 기반으로 도형을 추론합니다.
    
    Parameters:
        objects (list of dict): 각 객체는 'id', 'height', 'dimensions', 'area' 등을 포함
        name_shape_map (dict): 키워드 → shape 매핑. None이면 기본값 사용.

    Returns:
        list of dict: 각 객체에 'shape' 키가 추가된 리스트
    """
    if name_shape_map is None:
        name_shape_map = {
            "box": "rectangular",
            "can": "cylinder",
            "slab": "flat",
            "cube": "rectangular",
            "cylinder": "cylinder",
            "brick": "rectangular",
            "plate": "flat",
            "ball": "sphere",
            "block": "rectangular",
        }

    def extract_shape_flexible(object_id):
        object_id_lower = object_id.lower()
        for keyword, shape in name_shape_map.items():
            if keyword in object_id_lower:
                return shape
        return "unknown"

    for obj in objects:
        obj["shape"] = extract_shape_flexible(obj["id"])

    return objects


def filter_objs_dimensions_height_name_area(objs):
    g = lambda d,*ks: next((d[k] for k in ks if k in d), None)
    pos = lambda p: ([float(p[0]),float(p[1]),float(p[2])] if isinstance(p,(list,tuple)) and len(p)>=3
                     else ([float(p['x']),float(p['y']),float(p['z'])] if isinstance(p,dict) and all(k in p for k in('x','y','z')) else None))
    nums = lambda v: [float(x) for x in v] if isinstance(v,(list,tuple)) else None

    out=[]
    for o in objs:
        if not isinstance(o,dict): continue
        p = pos(g(o,'position','pos','center'))
        d = nums(g(o,'dimensions','dimension','size'))
        a = g(o,'area','surface','area_m2')
        out.append({
            'id':   g(o,'id','object_id','uuid'),
            # 'name': g(o,'id','object_name'),
            # 'height': p[2],
            'dimensions': [min(d[0], d[1], p[2]), max(d[0], d[1], p[2])],

            'area': float(a)
        })
    return out

def get_area_sorted_objects(normalized_objs):
    """넓이 순으로 정렬된 물체 리스트 반환 (id, dimensions만)"""
    # 넓이 순으로 정렬 (큰 것부터)
    sorted_objs = sorted(normalized_objs, key=lambda x: x['area'], reverse=True)
    
    # id와 dimensions만 추출
    area_sorted = []
    for obj in sorted_objs:
        area_sorted.append({
            'id': obj['id'],
            'dimensions': obj['dimensions'],  # [shortest, longest]
            'shape': obj['shape']
        })
    
    return area_sorted

def get_similar_height_groups(normalized_objs, threshold=0.15):
    """비슷한 높이의 물체들을 그룹으로 묶어서 반환"""
    # 높이 정보 추출 (dimensions의 longest axis만 고려)
    objects_with_height = []
    for obj in normalized_objs:
        longest_axis = obj['dimensions'][1]  # longest axis = height
        objects_with_height.append({
            'id': obj['id'],
            'longest_axis': longest_axis,
            'shape': obj['shape']  # shape 정보 추가
        })
    
    # 높이 순으로 정렬 (longest axis 기준)
    objects_with_height.sort(key=lambda x: x['longest_axis'])
    
    # 비슷한 높이끼리 그룹화
    groups = []
    used_objects = set()
    
    for i, obj1 in enumerate(objects_with_height):
        if obj1['id'] in used_objects:
            continue
            
        # 새 그룹 시작
        group = [obj1]
        used_objects.add(obj1['id'])
        
        # 비슷한 높이의 다른 물체들 찾기
        for j, obj2 in enumerate(objects_with_height[i+1:], i+1):
            if obj2['id'] in used_objects:
                continue
                
            # 장축만 비교
            longest_diff = abs(obj1['longest_axis'] - obj2['longest_axis']) / max(obj1['longest_axis'], obj2['longest_axis'])
            
            # 장축 차이만 threshold 이내면 similar로 판정
            if longest_diff <= threshold:
                group.append(obj2)
                used_objects.add(obj2['id'])
        
        # 그룹이 1개 이상의 물체를 포함할 때만 추가
        if len(group) >= 1:  # 단일 물체도 그룹으로 포함
            group_objects = []
            for obj in group:
                group_objects.append({
                    'id': obj['id'],
                    'shape': obj['shape']
                })
            
            group_info = {
                'group_id': len(groups) + 1,
                'objects': group_objects  # id와 shape만 포함
            }
            groups.append(group_info)
    
    return groups




def parse_eval_refine_output_loose(text, keywords):
    """
    텍스트에서 지정된 키워드 섹션들을 파싱하여 딕셔너리로 반환
    
    Args:
        text: 파싱할 텍스트 (str, bytes, None)
        keywords: 파싱할 키워드 리스트 (예: ["evaluation", "revised_plan"])
    
    Returns:
        dict: {키워드: 내용} 형태의 딕셔너리
    
    Raises:
        ValueError: 키워드 섹션을 찾을 수 없거나 비어있을 때
    """
    s = "" if text is None else (text.decode(errors="replace") if isinstance(text, bytes) else str(text))
    
    # 메타데이터 제거: 'additional_kwargs=' 이후 잘라냄
    meta_idx = s.find("additional_kwargs=")
    if meta_idx != -1:
        s = s[:meta_idx]

    # 마크다운 코드블록 제거
    s = re.sub(r"^\s*```.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"^\s*```$", "", s, flags=re.MULTILINE)

    # 줄바꿈 정규화
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    
    flags = re.I | re.S
    result = {}
    
    for i, keyword in enumerate(keywords):
        if i == len(keywords) - 1:  # 마지막 키워드
            pattern = rf"{keyword}:\s*(?P<content>.*)\Z"
        else:
            next_keyword = keywords[i + 1]
            pattern = rf"{keyword}:\s*(?P<content>.*?)(?:\n\s*###\s*\n\s*)?{next_keyword}:"
        
        m = re.search(pattern, s, flags)
        if not m:
            raise ValueError(f"Could not find {keyword} section.")
        
        content = m.group("content").strip()
        if not content:
            content = "No content found."
        
        # 따옴표 제거
        if content[:1] in "\"'" and content[-1:] == content[:1]:
            content = content[1:-1]

        # 이스케이프 문자 처리
        content = (content.replace("\\r\\n", "\n").replace("\\r", "\n")
                            .replace("\\n", "\n").replace("\\t", "\t")
                            .replace("\r\n", "\n").replace("\r", "\n").strip())
        
        result[keyword] = content
    
    return result


def formatted_plan_parser(text: Union[str, bytes, None]) -> List[Dict[str, str]]:
    """
    LLM 출력에서 로봇 계획을 파싱하여 구조화된 데이터로 변환
    
    Args:
        text: 파싱할 텍스트 (str, bytes, None)
        
    Returns:
        List[Dict]: 각 스텝별 파싱된 데이터
        [
            {
                'step': '1',
                'analysis': '...',
                'target_object': '...',
                'reference_object': '...',
                'relative_position': '...',
                'distance': '...',
                'object_orientation': '...',
                'alignment_reference': '...'  # optional
            },
            ...
        ]
        
    Raises:
        ValueError: 파싱 실패 시
    """
    # 입력 전처리
    if text is None:
        raise ValueError("Input text is None")
    
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    
    text = str(text).strip()
    
    # 코드 블록 제거
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # STEP으로 분할
    step_pattern = r"STEP\s+(\d+)"
    steps = re.split(step_pattern, text)[1:]  # 첫 번째 빈 부분 제거
    
    if len(steps) % 2 != 0:
        raise ValueError("Invalid step format: steps and content mismatch")
    
    results = []
    required_fields = ['target_object', 'reference_object', 'relative_position', 
                      'distance', 'object_orientation']
    
    for i in range(0, len(steps), 2):
        step_num = steps[i].strip()
        step_content = steps[i + 1].strip()
        
        step_data = {'step': step_num}
        
        # 각 필드 파싱
        fields_to_parse = required_fields + ['analysis', 'alignment_reference']
        
        for field in fields_to_parse:
            pattern = rf"{field}:\s*(.+?)(?=\n\w+:|$)"
            match = re.search(pattern, step_content, re.DOTALL)
            
            if match:
                value = match.group(1).strip()
                step_data[field] = value
            elif field in required_fields:
                raise ValueError(f"Required field '{field}' not found in step {step_num}")
        
        # alignment_reference는 object_orientation이 parallel일 때만 필요
        if step_data.get('object_orientation') == 'parallel' and 'alignment_reference' not in step_data:
            raise ValueError(f"alignment_reference is required when object_orientation is parallel in step {step_num}")
        
        results.append(step_data)
    
    return results


async def assemble_agent_llm(detect_data, tool_node,llm=llm):
    filtered_obj = filter_objs_dimensions_height_name_area(detect_data)
    print("object that llm got :", filtered_obj)
    # done = tts.speak_auto_sequential(filtered_obj)
    user_input = " 앞에 보이는 물건들로 고인돌 만들어"
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
    - capstone's → object_id3's

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
- Step 1: Place block1 vertically on table
- Step 2: Place block2 vertically on table
- Step 3: Place beam1 horizontally across block1 and block2

Output:
STEP 1
analysis: Place block1 in upright position on table surface
target_object: block1
reference_object: table
relative_position: above
distance: None
object_orientation: vertical

STEP 2
analysis: Place beam1 horizontally across block1 and block2
target_object: beam1
reference_object: block1, block2
relative_position: center
distance: None
object_orientation: parallel
alignment_reference: block1, block2

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


    done = tts.speak_auto_sequential(filtered_obj)
    done.wait()



async def detector_agent_without_llm(state: AgentState, tool_node) -> Dict[str, Any]:
    """ all_object_name 또는 detect_failed_objects를 기반으로 위치 + 크기 + 면적 + 각도를 탐지
        같은 이름의 객체가 여러개 감지되는 경우를 대비해 객체별로 id를 부여함
         ex) white box가 2개면 white_box_1, white_box_2 이런식으로 """

    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 3
    attempt = 0
    raw = ""
    print(f"[Detector Agent] Tool call attempt {attempt + 1}/{max_retries}")
    # 로봇팔 엔드툴 포즈
    tool_pos = await tool_node.ainvoke([robot_get_tcp_pose_message()])
    tool_pos = tool_pos[0].content
    tool_pos = list(map(float, eval(tool_pos)))
    cleaned_names = [n.strip().strip('"').strip("'") for n in object_names if n.strip()]

    # 결과 리스트 초기화
    detect_list = []


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
            # print("detection_res is :", detect_res)
            first_msg = detect_res[0]
            raw = first_msg.content or ""
            # print("[DETECTOR] raw is :", detect_res)
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
                        detect_list.extend(cleaned_response)
                        object_success = True
                        print("[Detector Agent] found :",cleaned_response)
                        break  # 성공했으면 재시도 루프 탈출
                    else:
                        print(f"[WARNING] '{name}' - No valid items after filtering")
                        
                except Exception as e:
                    print(f"Top-level JSON parse failed for '{name}':", e)
            
                # 빈 응답이거나 파싱 실패시 재시도
                if attempt < max_retries:
                    print(f"[Detector Agent] Retrying '{name}' in 0.1 seconds...")
                    await asyncio.sleep(0.1)
            attempt += 1
        # 재시도 후에도 실패했으면 실패 목록에 추가
        if not object_success:
            print(f"[FAILED] '{name}' could not be detected after {max_retries} attempts")

    # 최종 결과
    raw = detect_list  # 모든 객체의 결과가 합쳐진 리스트
    try:
        # 이후 기존 로직대로 raw → parse_to_obj → det_list 처리
        # 만약 빈 값이 들어오면 except로 빠짐
        parsed_raw = parse_to_obj(raw) # 이후 파싱
        det_list = ensure_list_of_dicts(parsed_raw)
        # det_list 안에 아직 문자열(JSON) 요소가 남아 있을 경우, dict로 파싱
        det_list = [json.loads(item) if isinstance(item, str) else item for item in det_list]
        print("[detector agent] raw output :", det_list)
        det_list = deduplicate_by_position(det_list)
    except Exception as e:
        print(f"[Detector Agent] Error parsing raw data: {e}")

    # 1) 루프 시작 전에 counts dict 초기화
    counts: Dict[str, int] = {}
    locations = []
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
                "dimensions": det["dimensions"],
                "rotation_degree": det["rotation_degree"],
                "mask_base64" : det["mask_base64"],
                "mask_shape" : det["mask_shape"],
                "score" : det["score"]

            } # 7_24 변경점: rotation degree, dimention 제거
            # print("obj is",obj)

            locations.append(obj)

        except Exception as e:
            print(f"[Detector Agent] Failed parsing for {cleaned_names[idx]}: {e}")
    locations = remove_duplicates_by_id(locations)

    print("[Detector Agent] output objects are :", locations)



    return await assemble_agent_llm(locations,tool_node)

# Graph 생성
async def create_graph(tools, selected_tool):
    graph = StateGraph(state_schema=AgentState)  # 상태 스키마 전달
    tool_node = ToolNode(tools)  # 도구 노드 생성
    # llm_assemble = llm.bind_tools(selected_tool)
    # 비동기 함수로 추가
    graph.add_node("detectorAgent", wrap_async(detector_agent_without_llm, tool_node=tool_node))
    graph.add_node("anygraspAgent", wrap_async(assemble_agent_llm, tool_node=tool_node, llm = llm))
    graph.set_entry_point("detectorAgent")  # 시작 지점을 설정
    
    # 비동기적으로 그래프를 컴파일할 수 있도록 확인
    compiled_graph = graph.compile()  # 비동기적으로 컴파일
    return compiled_graph


async def find_tools_by_names(all_tools, tool_names):
    """2. 도구 이름으로 도구 찾기"""
    found_tools = [tool for tool in all_tools if tool.name in tool_names]
    return found_tools

async def main():
    mcp_server_configs = create_server_config()  # MCP 서버 설정 불러오기

    print("--- MCP Client Initializing ---")
    try:
        async with MultiServerMCPClient(mcp_server_configs) as client: # mcp 서버에 연결 및 실행상태 유지
            print("--- MCP Client Initialized ---")
            tools = client.get_tools() # MCP 서버에서 도구 가져오기
            # print("사용 가능 도구:",tools)
            selected_tool = await find_tools_by_names(tools, "check_collision")
            compiled_graph = await create_graph(tools, selected_tool)  # 그래프 비동기적으로 생성

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

