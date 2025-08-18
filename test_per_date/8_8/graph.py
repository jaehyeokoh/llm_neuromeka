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
        # objects = []
        global actions_divide_num
        print("actions_divide_num",actions_divide_num)
        user_input = state["user_input"]
        print("user input is",user_input)

        len_of_seq = len(state["divided_user_input"])
        print("[route_divide] divided_user_input is", state["divided_user_input"])
        print("len of seq",len_of_seq)
        if actions_divide_num >= len_of_seq:
            return {"result_of_divide_seq":{"result":"done"}, "done":True}

        agent = state["divided_user_input"][actions_divide_num]["agent"]
        user_input_divide = state["divided_user_input"][actions_divide_num]["action"]
        # if agent == "simple_agent":
            # objects = state["divided_user_input"][actions_divide_num]["objects"]
        #     print("[route_divide] 관련 물체 : ", objects)
        # print(f"[Route_divider]임무가 배정된 에이전트:{agent}, 임무 내용 : {user_input_divide}, 관련 물체 : {objects}")

        actions_divide_num += 1
        return {"result_of_divide_seq": {"result":"not_done", "agent":agent}, "one_of_divided_user_input": user_input_divide, "done": False} # done도 False로 틀어야 다음에 실행됨

    # 비동기 함수로 추가
    graph.add_node("DivierAgent", wrap_async(user_input_divider_agent, llm=llm, tool_node = tool_node))  # 비동기적으로 실행되는 노드 추가
    graph.add_node("HardAgent", wrap_async(hard_agent_supervisor, llm=llm, tool_node = tool_node))  # 비동기적으로 실행되는 노드 추가
    graph.add_node("VLMAgent", vlm_agent_node)  # 비동기적으로 실행되는 노드 추가
    graph.add_node("detectorAgent", wrap_async(detector_agent_without_llm, tool_node=tool_node))
    graph.add_node("userInputAnalyzeateAgent", wrap_async(simple_user_input_analyze_agent, llm=llm))  # 사용자 입력 분석 에이전트 추가
    graph.add_node("simple_plan_editor_agent", wrap_async(simple_plan_editor_agent, llm=llm, tool_node=tool_node))  # plan을 실행시키는 슈퍼바이저 추가
    graph.add_node("plan_execute_supervisor", wrap_async(analyzed_user_input_regenerate_supervisor, llm=llm, tool_node=tool_node))  # plan을 실행시키는 슈퍼바이저 추가
    graph.add_node("move_agent", wrap_async(move_agent, llm=llm, tool_node=tool_node))  # 단순 이동만 담당하는 에이전트
    graph.add_node("grip_agent", wrap_async(grip_agent, llm=llm, tool_node=tool_node))  # 단순 이동만 담당하는 에이전트
    graph.add_node("route_divide", route_divide)
    graph.add_node("detect_failed_feedback_agent", wrap_async(detect_failed_feedback_agent, llm=llm, tool_node=tool_node))  # detect_failed_feedback_agent
    graph.add_node("assembly_agent", wrap_async(Assembly_agent, llm=llm, tool_node=tool_node))  # detect_failed_feedback_agent


    graph.add_edge("VLMAgent", "detectorAgent") 
    graph.add_edge("detectorAgent", "userInputAnalyzeateAgent")  # detectorAgent에서 userInputAnalyzeateAgent 연결
    graph.add_edge("userInputAnalyzeateAgent", "simple_plan_editor_agent")  # userInputAnalyzeateAgent에서 simple_plan_editor_agent 연결
    # action_supervisor에서 엑션이 끝나면 END로, 아니면 action_executor_agent로 가도록 조건부 엣지 추가
    def route_based_on_classification(state: AgentState) -> str:
        if state.get("done", True):  # done이 True면 작업이 끝났다고 판단
            return "done"
        return "not_done"
    
    graph.add_conditional_edges(
        "simple_plan_editor_agent", # regenerate 이후
        route_based_on_classification,
        {
            "not_done": "plan_execute_supervisor",  # 결과가 detect_success이면 initial_planner_agent 가서 작업 계속
            "done": END # 객체 끝끝내 못찾으면 종료
        }
    )
    graph.set_entry_point("DivierAgent")  # 시작 지점을 설정
    def divide_classification(state: AgentState) -> str:
        print(state["result_of_divide_seq"])
        if state["result_of_divide_seq"]["result"] == "done":  # done이 True면 작업이 끝났다고 판단(agent state의 done과 다른거임!!)
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
            "hard":"HardAgent" # 결과가 어려우면 hard agent에 매칭
        }
    )
    graph.add_edge("HardAgent","assembly_agent") # 일단 바로 종료하게 함 나중에 수정할 것 
    graph.add_edge("assembly_agent","plan_execute_supervisor")

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
