import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from graph import * # 에이전트 함수 불러오기
import warnings
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_MINE_API") # 내 계정
# api_key = os.getenv("OPENAI_NEUROMEKA_API") # 뉴로메카 계정

# from langchain_google_genai import ChatGoogleGenerativeAI
# 특정 경고만 억제
warnings.filterwarnings("ignore", message=".*dropout option adds dropout.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*")

# 또는 카테고리별 억제
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
# from langchain_core.globals import set_debug, set_verbose

# # 디버그 모드 끄기
# set_debug(False)
# set_verbose(False)


async def main():

    llm = ChatOpenAI(model="gpt-4.1", temperature=0, api_key = api_key)

    # llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key="AIzaSyCquWt4KjYq0hCveRvEVikNssifzKa3ZYc")
    # llm = ChatOpenAI(base_url="https://openrouter.ai/api/v1",model="mistralai/mistral-small-3.2-24b-instruct:free", temperature=0, api_key="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") # openrouter 사용
    mcp_server_configs = create_server_config()  # MCP 서버 설정 불러오기

    print("--- MCP Client Initializing ---")
    try:
        async with MultiServerMCPClient(mcp_server_configs) as client: # mcp 서버에 연결 및 실행상태 유지
            print("--- MCP Client Initialized ---")
            tools = client.get_tools() # MCP 서버에서 도구 가져오기
            compiled_graph = await create_graph(llm, tools)  # 그래프 비동기적으로 생성

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

