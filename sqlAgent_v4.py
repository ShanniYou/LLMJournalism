# Shanni You, @04/17/2025
# Implemented with stateGraph, a package from langchain, to define the agent
# LLM: Chatgpt-4, API stored in .env file
# Add feature: use sql_RAG to retrieve the relevant tables only
# Add feature: only use subset of SQL tables to run the agent

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableWithFallbacks
from langchain_core.messages import ToolMessage
from langchain_core.messages import AIMessage

from typing import Any
from typing import Annotated, Literal
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display
from pydantic import BaseModel, Field
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain.globals import set_verbose
from dotenv import load_dotenv
load_dotenv()
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
collector = RunCollectorCallbackHandler()


#llmmodel = "gpt-4o"  # gpt-4o is the model that is used for the agent
set_verbose(True)
#llmmodel = "mistral"
llmmodel = "gpt-4o"  # gpt-4o is the model that is used for the agent
model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name = model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)
user_query = "I want to know what high school students are studing by the intel competition information."

def related_sql_tables(user_query):
    # Load the vector store
    print("Reloading the vector store...")
    db = Chroma(collection_name = "hf_schema_all", 
                        embedding_function=hf,
                        persist_directory="./chroma_schema_hf_db",)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    docs = retriever.get_relevant_documents(user_query)
    
    table_lists = []
    for doc in docs:
        #print(doc.page_content)
        
        table_lists.append(doc.metadata['table_name'])
    
    return table_lists

class SubsetSQLDatabase(SQLDatabase):
    """A subset of SQLDatabase that only includes specified tables."""
    
    def get_table_names(self):
        return self._include_tables or []

    def get_table_info(self, table_names = None):

        if table_names is None:
            tables = self._include_tables
        else:
            tables = [t for t in table_names if t in self._include_tables]
        return super().get_table_info(table_names=tables)


##############    Pre-processing for SQL tables      ############
# SQL database plug in:
#db = SQLDatabase.from_uri('sqlite:///Chinook.db')  # Test database
table_lists = related_sql_tables(user_query)
print("Here are the related SQL tables: ", table_lists)
db = SubsetSQLDatabase.from_uri('mysql+mysqlconnector://web:JAzv3WHQh-y@newsdayinteractive.cr2zrybivkdw.us-east-1.rds.amazonaws.com:3306/daily',
include_tables = table_lists,
sample_rows_in_table_info = 3)  # Test database
#print(db.dialect)
#print(db.get_usable_table_names())
print(db.get_table_info())
#print(db.run('SELECT * FROM Artist LIMIT 10;'))

# Utilities getting the table name and the schema:
toolkit = SQLDatabaseToolkit(db = db, llm = ChatOpenAI(model = llmmodel, max_retries = 3, temperature=0))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == 'sql_db_list_tables')
get_schema_tool = next(tool for tool in tools if tool.name == 'sql_db_schema')

print(list_tables_tool.invoke(""))  # manually calling the table names
print(get_schema_tool.invoke('120111_intel_semifinalists')) # DDL for Artist
#print(get_schema_tool.invoke('130123_intel')) # DDL for Artist

##############     Utilities to handle errors     ###############
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

####################   Tool for the agent workflow    ##################
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

#print(db_query_tool.invoke("SELECT * FROM Artist LIMIT 10;"))

#################  Building Agent Workflow  ###############
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
# Copied
query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("placeholder", "{messages}")]
    )

query_check = query_check_prompt | ChatOpenAI(model = llmmodel, max_retries = 3, temperature=0).bind_tools(
        [db_query_tool], tool_choice = "required"
    ) # need to check if the query is a valid SQL query

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
model_get_schema = ChatOpenAI(model = llmmodel, temperature=0).bind_tools([get_schema_tool])
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
Do NOT call any tool besides SubmitFinalAnswer to submit the final answer. Remember, the SQL query is not the final answer yet.
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

""".format(dialect = db.dialect, top_k = 5)
# Copied

query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", query_gen_system), ("placeholder", "{messages}")]
)
query_gen = query_gen_prompt | ChatOpenAI(model = llmmodel, temperature=0).bind_tools(
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

# Draw the workflow

def draw_agent_worflow():
    with open("workflow.png", "wb") as f:
        f.write(
            app.get_graph().draw_mermaid_png(
                draw_method = MermaidDrawMethod.API,
            )
        )
    print('Save graph to workflow.png')

#################  Now is the time to evaluate the agent to the reference answer   ###############
import json

def predict_sql_agent_answer(example: dict):
    '''use this for answer evaluation'''
    msg = {"messages": ("user", example["input"])}
    #msg = {"messages": ("user", example)}
    messages = app.invoke(msg)
    json_str = messages["messages"][-1].tool_calls[0]["args"]
    response = json_str["final_answer"]
    return {"response": response}

from langchain import hub
from langchain_openai import ChatOpenAI

# Grade prompt:
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")

def answer_evaluator(run, example) -> dict:
    '''
    A simple evaluator for RAG answer accuracy
    '''
    print('See what is inside the run param', run.outputs)
    # Get question, ground truth answer, chain
    input_question = example.inputs["input"]
    reference = example.outputs["output"]
    prediction = run.outputs["response"]

    # LLM grader
    llm = ChatOpenAI(model = llmmodel, temperature = 0)

    # Structured Prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # RUN evaluator
    score = answer_grader.invoke(
        {
            "question": input_question,
            "correct_answer": reference,
            "student_answer": prediction,
        }
    )
    score = score["Score"]
    return {'key': 'answer_v_reference_score', "score": score}

# Trajectory:

def predict_sql_agent_messages(example: dict):
    """ Use this for answer evaluation """
    msg = {"messages": ("user", example["input"])}
    messages = app.invoke(msg)
    return {"response": messages}

from langsmith.schemas import Example, Run
def find_tool_calls(messages):
    '''
    Find all tool calls in the messages returned
    '''
    tool_calls = [
        tc["name"] for m in messages["messages"] for tc in getattr(m, "tool_calls", [])
    ]
    return tool_calls

def contains_all_tool_calls_in_order_exact_match(
    root_run: Run, example: Example
) -> dict:
    """
    Check if all expected tools are called in exact order and without any additional tool calls
    """
    expected_trajectory = [
        "sql_db_list_tables", 
        "sql_db_schema",
        "db_query_tool",
        "SubmitFinalAnswer",
    ]
    messages = root_run.outputs["response"]
    tool_calls = find_tool_calls(message)

    # Print out the tool calls for debugging
    print("Here are my tool calls")
    print(tool_calls)

    # Check if the tool calls match the expected trajectory exactly
    if tool_calls == expected_trajectory:
        score = 1
    else:
        score = 0
    
    return {"score": int(score), "key": "multi_tool_call_in_exact_order"}

def contains_all_tool_calls_in_order(root_run: Run, example: Example) -> dict:
    """
    Check if all expected tools are called in order,
    but it allows for other tools to be called in bewtween the expected ones.
    """
    messages = root_run.outputs["response"]
    tool_calls = find_tool_calls(messages)

    print("Here are my tool calls: ")
    print(tool_calls)

    it = iter(tool_calls)
    if all(elem in it for elem in expected_trajectory):
        score = 1
    else:
        score = 0
    return {"score": int(score), "key": "multi_tool_call_in_order"}


def main():
    
    # Run the Agent
    messages = app.invoke(
        {"messages": [("user", user_query)]}
    )

    json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
    print('######## Here is the output of agent  ########## ')
    print('Checking the final answer: ', json_str)
    #print(messages["usage"])

    '''
    for step in messages["messages"]:
        print(step)
    '''

    
    for event in app.stream(
        {"messages": [("user", "What does student from California study?")]}
    ):
        print(event)
    
    

if __name__ == '__main__':
    main()
