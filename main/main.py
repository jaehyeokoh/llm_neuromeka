import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from graph import * # 에이전트 함수 불러오기
import warnings
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_NEUROMEKA_API") # api key 불러오기

# 경고 억제
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
# from langchain_core.globals import set_debug, set_verbose

# # 디버그 모드 끄기
# set_debug(False)
# set_verbose(False)


async def main():

    llm = ChatOpenAI(
    model="gpt-5",
    use_responses_api=True,                 # Responses API로 강제 -> 이거 하면 openai에서 제공하는 설정들 (verbosity,..)사용 가능
    output_version="responses/v1",          # 블록형 출력 (openai에서 추천하는 설정)
    extra_body={
        "text":{"verbosity": "low"},                 # low | medium | high
        "reasoning": {"effort": "minimal"},     # minimal | low | medium | high
    },
    api_key=api_key,
    )


    llm = TextLLM(llm) # 바로 text 출력 나오게 하는 래퍼
    mcp_server_configs = create_server_config()  # MCP 서버 설정 불러오기

    print("--- MCP Client Initializing ---")
    try:
        async with MultiServerMCPClient(mcp_server_configs) as client: # mcp 서버에 연결 및 실행상태 유지
            print("--- MCP Client Initialized ---")
            tools = client.get_tools() # MCP 서버에서 도구 가져오기
            compiled_graph = await create_graph(llm, tools)  # 그래프 생성

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

