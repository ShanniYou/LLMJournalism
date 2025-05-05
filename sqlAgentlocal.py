# Shanni You, @04/17/2025
# Implemented with stateGraph, a package from langchain, to define the agent
# LLM: Chatgpt-4, API stored in .env file
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
#from langchain_openai import ChatOpenAI
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
import json

##############     Decide the llm to be used       ###############
from langchain_community.chat_models import ChatOllama
# Link to local ollama
llm = ChatOllama(
    base_url = "http://ollama:11434",
    model = "mistral",
    temperature = 0,
)

##############    Pre-processing for SQL tables      ############
# SQL database plug in:
db = SQLDatabase.from_uri('sqlite:///Chinook.db')  # Test database
print(db.dialect)
print(db.get_usable_table_names())
print(db.run('SELECT * FROM Artist LIMIT 10;'))

# Utilities getting the table name and the schema:
toolkit = SQLDatabaseToolkit(db = db, llm = llm)
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == 'sql_db_list_tables')
get_schema_tool = next(tool for tool in tools if tool.name == 'sql_db_schema')

#print(list_tables_tool.invoke(""))  # manually calling the table names
#print(get_schema_tool.invoke('Artist')) # DDL for Artist

##############     Utilities to handle errors     #################
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

print(db_query_tool.invoke("SELECT * FROM Artist LIMIT 10;"))

#################  Building Agent Workflow  ###############
# Promt the LLM to check for common mistakes in the query and later add this as a node in the workflow
# Also need to tell it to use JSON to return tool call

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

    If there are any of the above mistakes, rewrite the query and wrap it in a JSON like:
    {{ "tool": "db_query_tool",
     "sql": "SELECT * FROM ..." }}
    
    If there are no mistakes, warp the original query in the same JSON format.

    Only return JSON. Do not add explanation.

    You will call the appropriate tool to execute the query after running this check.    
    """

query_check_prompt = ChatPromptTemplate.from_messages(
        [("system", query_check_system), ("human", "{query}")]
    )
'''
For GPT
query_check = query_check_prompt | llm.bind_tools(
        [db_query_tool], tool_choice = "required"
    ) # need to check if the query is a valid SQL query
'''

query_check_chain = query_check_prompt | llm


def query_check(user_query: str):
    response = query_check_chain.invoke({"query": user_query, "tool": "db_query_tool"})  # to complie with the json format
    print(" LLM Response:\n", response.content)

    parsed = json.loads(response.content)
    print('debug', parsed)

    try:
        parsed = json.loads(response.content)
        print('debug', parsed)
        if parsed.get("tool") == "db_query_tool":
            return db_query_tool.invoke(parsed["sql"])
        else:
            return "! Invalid action in response"
    
    except Exception as e:
        return f"! Failed to parse tool call: {e}"

#print(query_check.invoke({"messages": [("user", "SELECT * FROM Artist LIMIT 10;")]}))
print(query_check("SELECT * FROM Artist LIMIT 10;"))



# Define the state for the agent
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int = 0

# Define a new graoh
workflow = StateGraph(State)
#workflow.add_state("iteration_count", int, default = 0)

# Increase the iteration
def increment_iteration(state):
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    return {**state}

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
    #return {"messages": [query_check.invoke({"messages": [state["messages"][-1]]})]}
    return {"messages": [query_check.invoke(state["messages"][-1])]}

workflow.add_node("first_tool_call", first_tool_call)


# add nodes for the first two tools
workflow.add_node(
    "list_tables_tool", create_tool_node_with_fallback([list_tables_tool])
)
workflow.add_node("get_schema_tool", create_tool_node_with_fallback([get_schema_tool]))

# add a node for a model to choose the relevant tables based on the question and available tables
"""
model_get_schema = ChatOpenAI(model = 'gpt-3.5-turbo', temperature=0).bind_tools([get_schema_tool])
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [model_get_schema.invoke(state["messages"])],
    },
)
"""
# this is adapted for the local llm
schema_check_system = """ You are a sql schema specialist. Based on the user's question and list of available tables, choose
the most relevant tables and call the following tools: 

Tool: get_schema_tool
Arguments: 
    - table_name: name of the relevant table

Only respond in valid JSON format like this:
{{
"tool": "get_schema_tool",
"table_name": "Album" 
}}
"""

schema_check_prompt = ChatPromptTemplate.from_messages([("system", schema_check_system), ("human", "{query}")])
model_get_schema_chain = schema_check_prompt | llm
# ! This outcome is not being parsed:
workflow.add_node(
    "model_get_schema",
    lambda state: {
        "messages": [
            # Call LLM with user messages
            model_get_schema_chain.invoke({
                "query": state["messages"]
            }),
        ]
    },
)




# Describe a tool to represent the end state
class SubmitFinalAnswer(BaseModel):
    final_answer: str = Field(..., description = "The final answer to the user")

# Add a node for a model to generate a query based on the question and schema
query_gen_system = """You are a SQL expert with a strong attention to detail. 
Given an input question, output a synactically correct SQLite query to run, then look at the results of the query and return the answer.
Do NOT call any tool besides SubmitFinalAnswer to submit the final answer.

And you will only submit the final answer using this format:

{{
"tool": "SubmitFinalAnswer",
"output": "Your answer here"
}}

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
#query_gen = query_gen_prompt | llm.bind_tools(
#    [SubmitFinalAnswer]
#)
# Adapted for local llms:
query_gen = query_gen_prompt | llm

'''
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
'''
# for local llms:
def query_gen_node(state: State):
    response = query_gen.invoke(state)
    tool_messages = []
    try:
        parsed = json.loads(response.content)

        if parsed.get("tool") == "SubmiFinalAnswer":
            tool_messages


workflow.add_node("query_gen", query_gen_node)

# add a node for the model to check the query before executing it
workflow.add_node("correct_query", model_check_query)

# add a node for executing the query
workflow.add_node("execute_query", create_tool_node_with_fallback([db_query_tool]))


# To set a iteration increamental 
MAX_ITERATIONS = 5
# Define a conditional edge to decide whether to continue or end the workflow
def should_continue(state: State) -> Literal[END, "correct_query", "query_gen"]:
    state = increment_iteration(state)
    print('state info', state["iteration_count"])
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return END

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
    #llm = ChatOpenAI(model = "gpt-4-turbo", temperature = 0)

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

expected_trajectory = [
        "sql_db_list_tables",       # first: list_tables_tool node
        "sql_db_schema",            # second: get_schema_tool node
        "db_query_tool",            # third: execute_query node
        "SubmitFinalAnswer",         # fourth: query_gen
    ]

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
    set_verbose(True)
    # Run the Agent
    messages = app.invoke(
        {"messages": [("user", "which sales agent made the most in sales in 2009?")]}
    )

    json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
    print('######## Here is the output of agent  ########## ')
    print('Checking the final answer: ', json_str)
    

    '''
    ############  Create a Evaluation Dataset   ##############
    from langsmith import Client
    client = Client()
    # Create a dataset
    examples = [
        #("Which country's customers spent the most? And how much did they spend?", "The country whose customers spent the most is the USA, with a total expenditure of $523.06"),
        #("What was the most purchased track of 2013?", "The most purchased track of 2013 was Hot Girl."),
        #("How many albums does the artist Led Zeppelin have?", "Led Zeppelin has 14 albums."),
        #("What is the total price for the album 'Big Ones'?", "The total price for the album 'Big Ones' is 14.85"),
        ("Which sales agent made the most in sales in 2009?", "Steve Johnson made the most most sales in 2009"),
    ]
    dataset_name = "SQL Agent Response"
    if not client.has_dataset(dataset_name = dataset_name):
        dataset = client.create_dataset(dataset_name = dataset_name)
        inputs, outputs = zip(
            *[({"input": text}, {"output": label}) for text, label in examples]
        )
        client.create_examples(inputs=inputs, outputs=outputs, dataset_id = dataset.id)

    from langsmith.evaluation import evaluate

    #print('Input user prompt: ', examples[0][0])
    #print(predict_sql_agent_answer(examples[0][0]))
    
    dataset_name = "SQL Agent Response"
    try:
        experiment_results = evaluate(
            predict_sql_agent_answer,
            data = dataset_name,
            evaluators = [answer_evaluator],
            num_repetitions = 3,
            experiment_prefix = "sql-agent-multi-step-response-v-reference",
            metadata = {'version': "Chinook, gpt-4o multi-step-agent"},
        )
    except:
        print('Pleaset set up LangSmitch')
    
    ########### Evaluate Trajectory #################
    # These are the tools that we expect the agent to use
    expected_trajectory = [
        "sql_db_list_tables",       # first: list_tables_tool node
        "sql_db_schema",            # second: get_schema_tool node
        "db_query_tool",            # third: execute_query node
        "SubmitFinalAnswer",         # fourth: query_gen
    ]

    try: experiment_results = evaluate(
        predict_sql_agent_messages,
        data = dataset_name,
        evaluators = [
            contains_all_tool_calls_in_order,
            contains_all_tool_calls_in_order_exact_match,
        ],
        num_repetitions=3,
        experiment_prefix="sql-agent-multi-step-tool-calling-trajectory-in-order",
        metadata={"version": "Chinook, gpt-4o multi-step-agent"},
    )
    except:
        print("Please setup LangSmith")
    '''

if __name__ == '__main__':
    main()


