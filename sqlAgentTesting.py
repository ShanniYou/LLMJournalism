# Shanni You, 04/11/2025
# Sql Agent

from sqlUtils import *
import getpass
import os

from typing import Any
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langgraph.prebuilt import ToolNode
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
import requests
from typing import Annotated, Literal
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain.globals import set_verbose
from langchain.agents import initialize_agent



# Setup API keys: already put into the .env file

db = SQLDatabase.from_uri('sqlite:///Chinook.db')  # Test database
print(db.dialect)
print(db.get_usable_table_names())
print(db.run('SELECT * FROM Artist LIMIT 10;'))
    
# Define Utilities Fun


toolkit = SQLDatabaseToolkit(db = db, llm = ChatOpenAI(model = "gpt-4", max_retries = 3, temperature=0))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == 'sql_db_list_tables')
get_schema_tool = next(tool for tool in tools if tool.name == 'sql_db_schema')

print(list_tables_tool.invoke(""))
print(get_schema_tool.invoke('Artist'))

# Utilities to handle errors


def create_tool_node_with_fallback(tools: list) -> RunnableWithFallbacks[Any, dict]:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key = 'error'
    )

def handle_tool_error(state) -> dict:
    error = state.get('error')
    tool_calls = state['messages'][-1].tool_calls
    return {
        'messages':[
            ToolMessage(
                content=f'Error: {repr(error)}\n please fix the problems.',
                tool_call_id = tc['id'],
            )
            for tc in tool_calls
        ]
    }

@tool
def db_query_tool(query:str) -> str:
    """
    Execute a SQL query against the database and get back the result.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """
    result = db.run_no_throw(query)
    print(f"This is a checking point: {tool} called with input: {query}")
    if not result:
        return "Error: Query failed. Please rewrite your query and try again. "
    return result

print(db_query_tool.invoke("SELECT * FROM Artist LIMIT 10;"))

# Promt the LLM to check for common mistakes in the query and later add this as a node in the workflow

query_check_system = """ You are a SQL expert with a strong attention to detail. 
    Double check the SQLite query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exlusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguements for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the orginal query.

    You will call the appropriate tool to execute the query after running this check.    
    """

query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("placeholder", "{messages}")]
    )

query_check = query_check_prompt | ChatOpenAI(model = "gpt-4", max_retries = 3, temperature=0).bind_tools(
        [db_query_tool], tool_choice = "required"
    ) # need to check if the query is a valid SQL query
    
    #api_key = os.getenv('OPENAI_API_KEY')
    #print('open ai Key', api_key)
    #print(query_check.invoke({"messages": [("user", "SELECT * FROM Artist LIMIT 10")]}))  # This step cost money



# Query
def testDBloading():
    # Using the test dataset
    url = 'https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db'
    response = requests.get(url)
    if response.status_code == 200:
        # Open a local file in binary write mode
        with open("Chinook.db", "wb") as file:
            file.write(response.content)
        print("File downloaded and saved as Chinook.db")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

# Define the workflow for the agent


# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Define a new graoh
workflow = StateGraph(State)

# Add a node for the first tool call
def first_tool_call(state: State) -> dict[str, list[AIMessage]]:
    return {
        "messages":[
            AIMessage(
                content = "",
                tool_calls = [
                    {
                        "name": "sql_db_list_tables",
                        "args": {},
                        "id": "tool_abcd123",
                    }
                ],
            )
        ]
    }

def model_check_query(state: State) -> dict[str, list[AIMessage]]:
    return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}

workflow.add_node("first_tool_call", first_tool_call)

# add nodes for the first two tools
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# add a node for a model to choose the relevant tables based on the question and available tables
model_get_schema = ChatOpenAI(model = 'gpt-3.5-turbo', temperature=0).bind_tools([get_schema_tool])
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)

# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    final_answer: str = Field(..., description = "The final answer to the user")

# Add a node for a model to generate a query based on the question and schema
query_gen_system = """You are a SQL expert with a strong attention to detail. 
Given an input question, output a synactically correct SQLite query to run, then look at the results of the query and return the answer.
Do NOT call any tool besides SubmitFinalAnswer to submit the final answer.
When generating the query:
Output the SQL query that answers the input question without a tool call.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns of a specific table, only ask for the relevant columns given the question.

If you get an error while excuting a query, rewrite the query and try again.

If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
Never make stuff up if you don't have enough information to answer the query... just say you don't have enough information.

If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.

Do not make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

"""

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatOpenAI(model = "gpt-4", max_retries = 3, temperature=0).bind_tools(
    [SubmitFinalAnswer]
)

def query_gen_node(state: State):
    message = query_gen.invoke(state)

    # Sometime, the LLM will hallucinate and call the wrong tool. We need to catch this and return an error message.
    tool_messages = []
    if message.tool_calls:
        for tc in message.tool_calls:
            if tc["name"] != "SubmitFinalAnswer":
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call. ",
                        tool_call_id = tc["id"],
                    )
                )
    else:
        tool_messages = []
    return {'messages': [message] + tool_messages}

workflow.add_node("query_gen", query_gen_node)

# add a node for the model to check the query before executing it
workflow.add_node("correct_query", model_check_query)

# add a node for executing the query
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))

# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there is a tool call, then we finish
    if getattr(last_message, "tool_calls", None):
        return END
    if last_message.content.startswith("Error:"):
        return "query_gen"
    else:
        return "correct_query"

# Specify the edges between the node:
workflow.add_edge(START, "first_tool_call")
workflow.add_edge("first_tool_call", "list_tables_tool")
workflow.add_edge("list_tables_tool", "model_get_schema")
workflow.add_edge("model_get_schema", "get_schema_tool")
workflow.add_edge("get_schema_tool", "query_gen")
workflow.add_conditional_edges("query_gen", should_continue,)
workflow.add_edge("correct_query", "execute_query")
workflow.add_edge("execute_query", "query_gen")

# complie the workflow into a runnable
app = workflow.compile()


def main():
    
    # to save the workflow image
    with open("workflow.png", "wb") as f:
        f.write(
            app.get_graph().draw_mermaid_png(
                draw_method = MermaidDrawMethod.API,
            )
        )
    print('Save graph to workflow.png')

    
    # Run the Agent
    messages = app.invoke(
        {"messages": [("user", "which sales agent made the most in sales in 2009?")]}
    )
    #json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
    #print(json_str)

    # print the event
    #for event in app.stream(
    #    {"messages": [("user", "which sales agent made the most in sales in 2009?")]}
    #):
    #    print(event)

    

if __name__ == '__main__':
    main()




