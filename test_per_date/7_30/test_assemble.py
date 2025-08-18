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

# # ✅ 루트 로거에 달려 있는 기존 핸들러 모두 제거
# logging.getLogger().handlers.clear()

# # ✅ 전파 방지 (중복 로깅 방지)
# logging.getLogger().propagate = False

# # ✅ 기본 로그 설정 복구 (필요에 따라 파일 또는 콘솔 설정)
# logging.basicConfig(level=logging.WARNING)  # WARNING 이상 로그만 출력


############################## 상태 정의 (state_schema) ##########################
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key="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")
# 상태 정의 (state_schema)
class AgentState(TypedDict):
    user_input: str # 사용자 입력을 저장하는 필드
    divided_user_input:  Optional[List[str]] = [] # 나눠진 사용자 입력 저장 필드
    all_objects_name: List[str] = []  # 기본값을 빈 리스트로 설정

initial_state = {
}
# object_names = [ "white box", "black box", "can" ] # 마지막에 잡을 대상 넣으면 됨.
object_names = ["white box", "brown box", "black cube", "blue silver can" ]
target_name = "blue silver can"  # 네가 잡으려는 물체

if target_name in object_names:
    object_names.remove(target_name)
    object_names.append(target_name)

print("타겟 물건 이름",target_name)

debug = False

def filter_objs(objs):
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
            # 'id':   g(o,'id','object_id','uuid'),
            'name': g(o,'name','object_name'),
            'position': p,
            # 'dimensions': d,
            'area': float(a)
        })
    return out

async def assemble_agent_llm(detect_data, tool_node,llm=llm):
    # end = detect_data[0]["position"][:2]
    # print("[assemble_agent_without_llm] end position:",end)
    # msg = check_collision_message(detect_data[-1],end,detect_data)
    # collision_res = await tool_node.ainvoke([msg])
    # collision_raw = json.loads(collision_res[0].content)
    # print("[assemble_agent_without_llm] ourput: ",collision_raw)

    # collided = collision_raw["collided"] # True, False
    # z_margin = collision_raw["z_margin"] # z_margine

    # print("[assemble_agent] collided:", collided)
    # print("[assemble_agent] z_margin:", z_margin)
    filtered_obj = filter_objs(detect_data)
    # llm_with_tools = llm.bind_tools(tools)
    save_obs_msg = save_obstacles2assemble_server(filtered_obj)
    save_obs_res = await tool_node.ainvoke([save_obs_msg])
    print(save_obs_res[0].content)


    get_obs_res = await tool_node.ainvoke([get_obstacles2assemble_server()])
    print("get_obs_res is ",get_obs_res[0].content)

    system_prompt = f"""
you are a agent with tool.
your goal is to check collision while moving object to other object with tool.
respond which object can move and which object collision when moving to other object with using tool.
plan your own.

the objects information is {filtered_obj}

RULES:
DO not determine whether collision will occur by your own. ALWAYS determine it by tool
IF YOU DIDN'T use the tool, then answer the reason why you didn't use it
IF YOU USED IT, DESCRIBE THE  INPUT PARAMETERS you input to tool.

output formatt:

reasoning:
1.describe the objects
2.describe the tool and how to use it.
3.reasoning for answer
4.answer exact the return of tool
###
answer:
result

    """
    res = llm.invoke([
        SystemMessage(content=system_prompt)
    ], config=None)
    print("res is :", res)



async def detector_agent_without_llm(state: AgentState, tool_node) -> Dict[str, Any]:
    """ all_object_name 또는 detect_failed_objects를 기반으로 위치 + 크기 + 면적 + 각도를 탐지
        같은 이름의 객체가 여러개 감지되는 경우를 대비해 객체별로 id를 부여함
         ex) white box가 2개면 white_box_1, white_box_2 이런식으로 """

    # 툴 호출 + 재시도 로직 (빈 값을 리턴하면 3번까지 재시도)
    max_retries = 3
    attempt = 0
    raw = ""
    print(f"[Detector Agent] Tool call attempt {attempt + 1}/{max_retries}")

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

    except Exception as e:
        print(f"[Detector Agent] Error parsing raw data: {e}")


    print("[detector agent] output :", det_list)
    return await assemble_agent_llm(det_list,tool_node)

# Graph 생성
async def create_graph(tools, selected_tool):
    graph = StateGraph(state_schema=AgentState)  # 상태 스키마 전달
    tool_node = ToolNode(tools)  # 도구 노드 생성
    llm_assemble = llm.bind_tools(selected_tool)
    # 비동기 함수로 추가
    graph.add_node("detectorAgent", wrap_async(detector_agent_without_llm, tool_node=tool_node))
    graph.add_node("anygraspAgent", wrap_async(assemble_agent_llm, tool_node=tool_node, llm = llm_assemble))
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

