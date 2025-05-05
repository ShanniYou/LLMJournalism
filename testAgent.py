# Code for testing GPT access with API keys
# A question to ask: why is the agent keep repeating while the chain already successfully ended.
from langchain.globals import set_verbose
from langchain.agents import initialize_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model = "gpt-4", max_retries = 0, temperature=0)

@tool
def hello_tool(name: str) -> str:
    """
    This is a function for saying hello
    """
    print(f"{tool} hello_tool called with input: {name}")
    return f"hello, {name}"

agent = initialize_agent(
    tools = [hello_tool],
        llm = llm,
        agent = "chat-zero-shot-react-description",
        verbose=True,
        max_iterations = 3,
        max_execution_time = 10
    )
        
def main():
    set_verbose(True)

    # log the output
    try:
        response = agent.invoke({"input": "say hello to Shanni!"})  # Cost Warming: docker is still running, need to kill it mannually or it will cost more money
        
    except KeyboardInterrupt:
        print("Interrupted! Shutting down...")
    finally:
        print("Cleaning finished. Exiting.")


if __name__ == '__main__':
    main()


