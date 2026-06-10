from agents import *  # 에이전트 함수 불러오기
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END


# 전역 변수 설정
actions_divide_num = 0


# Graph 생성
async def create_graph(llm, tools):
    graph = StateGraph(state_schema=AgentState)  # 상태 스키마 전달
    tool_node = ToolNode(tools)  # 도구 노드 생성
    
    # 비동기 함수 정의
    async def vlm_agent_node(state, config):
        # 비동기적으로 실행되는 vlm_agent 호출
        return await vlm_agent(llm, state, tool_node, config)

    def route_divide(state: AgentState):
        """
        divide된 인풋 분배
        """
        global actions_divide_num
        print("actions_divide_num",actions_divide_num)
        user_input = state["user_input"]
        print("user input is",user_input)

        len_of_seq = len(state["divided_user_input"])
        print("len of seq",len_of_seq)
        if actions_divide_num == len_of_seq:
            return {"result_of_divide_seq":{"result":"done"}}



        agent = state["divided_user_input"][actions_divide_num]["agent"]
        user_input_divide = state["divided_user_input"][actions_divide_num]["action"]
        print("actiuon is ", user_input_divide)
        print("agent is ", agent)
        state["user_input"] = user_input_divide

        actions_divide_num += 1
        print(actions_divide_num)
        return {"result_of_divide_seq": {"result":"not_done", "agent":agent}, "user_input": user_input_divide}


    # 비동기 함수로 추가
    graph.add_node("DivierAgent", wrap_async(user_input_divider_agent, llm=llm))  # 비동기적으로 실행되는 노드 추가
    graph.add_node("VLMAgent", vlm_agent_node)  # 비동기적으로 실행되는 노드 추가
    graph.add_node("detectorAgent", wrap_async(detector_agent_without_llm, tool_node=tool_node))
    graph.add_node("userInputMatchingObjectAgent", wrap_async(user_input_matching_object_agent, llm=llm))  # 사용자 입력 매칭 에이전트 추가
    graph.add_node("userInputAnalyzeateAgent", wrap_async(simple_user_input_analyze_agent, llm=llm))  # 사용자 입력 분석 에이전트 추가
    graph.add_node("plan_execute_supervisor", wrap_async(analyzed_user_input_regenerate_supervisor, llm=llm, tool_node=tool_node))  # plan을 실행시키는 슈퍼바이저 추가
    graph.add_node("move_agent", wrap_async(move_agent, llm=llm, tool_node=tool_node))  # 단순 이동만 담당하는 에이전트
    graph.add_node("grip_agent", wrap_async(grip_agent, llm=llm, tool_node=tool_node))  # 단순 이동만 담당하는 에이전트
    graph.add_node("route_divide", route_divide)

    graph.add_edge("VLMAgent", "detectorAgent") 

    def route_based_on_detectioon(state: AgentState) -> str:
        if state.get("detect_feedback",True): # 탐지 실패한 객체가 있으면
            return "detect_failed"
        elif state.get("passed_object", []):
            return "cant_find_object"
        return "detect_success"
    
    # 이게 위에서 언급한 조건부 엣지 추가 부분임
    graph.add_conditional_edges(
        "detectorAgent", # detect 이후
        route_based_on_detectioon,
        {
            "detect_failed": "VLMAgent",  # 결과가 detect_failed이면 VLMAgent로 감
            "detect_success": "userInputMatchingObjectAgent",  # 결과가 detect_success이면 userInputMatchingObjectAgent 가서 작업 계속
            "cant_find_object": END # 객체 끝끝내 못찾으면 종료
        }
    )

    graph.add_conditional_edges(
        "userInputMatchingObjectAgent", # regenerate 이후
        route_based_on_detectioon,
        {
            "detect_failed": "VLMAgent",  # 결과가 detect_failed이면 VLMAgent로 감
            "detect_success": "userInputAnalyzeateAgent",  # 결과가 detect_success이면 initial_planner_agent 가서 작업 계속
            "cant_find_object": END # 객체 끝끝내 못찾으면 종료
        }
    )

    graph.add_edge("userInputAnalyzeateAgent","plan_execute_supervisor")

    # action_supervisor에서 엑션이 끝나면 END로, 아니면 action_executor_agent로 가도록 조건부 엣지 추가
    def route_based_on_classification(state: AgentState) -> str:
        if state.get("done", True):  # done이 True면 작업이 끝났다고 판단
            return "done"
        return "not_done"
    















    graph.set_entry_point("DivierAgent")  # 시작 지점을 설정
    def divide_classification(state: AgentState) -> str:
        print(state["result_of_divide_seq"])
        if state["result_of_divide_seq"]["result"] == "done":  # done이 True면 작업이 끝났다고 판단
            return "done"
        elif state["result_of_divide_seq"]["agent"] == "simple_agent":
            return "simple"
        return "hard"
    


    graph.add_edge("DivierAgent","route_divide")
    graph.add_conditional_edges(
        "route_divide",
        divide_classification,
        {
            "done": END,  # 결과가 detect_failed이면 VLMAgent로 감
            "simple": "VLMAgent",  # 결과가 detect_success이면 initial_planner_agent 가서 작업 계속
            "hard":"route_divide"
        }
    )
    graph.add_conditional_edges(
        "plan_execute_supervisor",
        route_based_on_classification,
        {
            "done": "route_divide",  # 결과가 detect_failed이면 VLMAgent로 감 
            "not_done":"plan_execute_supervisor"
        }
    )
    # 비동기적으로 그래프를 컴파일할 수 있도록 확인
    compiled_graph = graph.compile()  # 비동기적으로 컴파일
    # print((compiled_graph).get_graph().draw_mermaid()) # 그래프 구조를 mermaid 형식으로 출력 (디버깅용)

    return compiled_graph
