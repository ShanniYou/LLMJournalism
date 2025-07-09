# Shanni You, @04/17/2025
# Implemented with stateGraph, a package from langchain, to define the agent
# LLM: Chatgpt-4, API stored in .env file
# Add feature: use sql_RAG to retrieve the relevant tables only
# Add feature: only use subset of SQL tables to run the agent
# Add feature: set up recursive limit to address the infinite loop issue
# Update: 05/29/2025 Wrapp it into a chain so then the router can call it

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

from langgraph.graph import END, StateGraph, START, MessagesState
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
import json

def related_sql_tables(user_query):
    model_name = "BAAI/bge-small-en"   # Embedding model for RAG semantic search
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(model_name = model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)
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



# Draw the workflow

def draw_agent_worflow():
    with open("workflow.png", "wb") as f:
        f.write(
            app.get_graph().draw_mermaid_png(
                draw_method = MermaidDrawMethod.API,
            )
        )
    print('Save graph to workflow.png')

def promptDicts():
    # need to load queries from json file

    with open('PromptLibs.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

##################  Building Agent Workflow  ###############
    # Promt the LLM to check for common mistakes in the query and later add this as a node in the workflow


def sql_chain(user_query, comLLM):
    promptLibs = promptDicts()
    table_lists = related_sql_tables(user_query)
    print("Related SQL tables:", table_lists)
    db = SubsetSQLDatabase.from_uri('mysql+mysqlconnector://web:JAzv3WHQh-y@newsdayinteractive.cr2zrybivkdw.us-east-1.rds.amazonaws.com:3306/daily',
        include_tables = table_lists)
        #sample_rows_in_table_info = 3)  # Test database
    llm = ChatOpenAI(model = comLLM, max_retries = 3, temperature=0)
    toolkit = SQLDatabaseToolkit(db = db, llm = llm)
    tools = toolkit.get_tools()

    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    get_schema_node = ToolNode([get_schema_tool], name = "get_schema")

    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    run_query_node = ToolNode([run_query_tool], name = "run_query")

    def list_tables(state: MessagesState):
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "abc123",
            "type": "tool_call",
        }
        tool_call_message = AIMessage(content = "", tool_calls = [tool_call])

        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")

        return {"messages": [tool_call_message, tool_message, response]}

    def call_get_schema(state: MessagesState):
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice = "any")
        response = llm_with_tools.invoke(state['messages'])

        return {"messages": [response]}

    generate_query_system_prompt = promptLibs['sql_query_gen_system'].format(dialect = db.dialect, top_k = 5)

    def generate_query(state: MessagesState):
        system_message = {
            "role": "system",
            "content": generate_query_system_prompt,
        }
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state['messages'])
        return {"messages": [response]}

    check_query_system_prompt = promptLibs['sql_query_check_system'].format(dialect = db.dialect)

    def check_query(state: MessagesState):
        system_message = {
            "role": "system",
            "content": check_query_system_prompt,
        }
        # generate artificial user message to check
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call['args']['query']}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice = "any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
        message = state["messages"]
        last_message = message[-1]
        if not last_message.tool_calls:
            return END
        else:
            return "check_query"
    
    builder = StateGraph(MessagesState)
    builder.add_node(list_tables)
    builder.add_node(call_get_schema)
    builder.add_node(get_schema_node, "get_schema")
    builder.add_node(generate_query)
    builder.add_node(check_query)
    builder.add_node(run_query_node, "run_query")

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges(
        "generate_query",
        should_continue,
    )
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")

    app = builder.compile()

    return app
    

def main():
    print("Routing to SQL Agent...")
    #user_query = "Which people have the same last name in the Village of Hempstead payroll?"  # Example user query
    user_query = "Could you pull out 10 people in the village of hempstead payroll and give me the sum of their salaries?"
    comLLM = "gpt-4o"  # Commercial LLM for accurate and reliable responses
    sql_app = sql_chain(user_query, comLLM)


    response = sql_app.invoke({"messages": [("user", user_query)]})
    print("SQL Agent Response:", response["messages"][-1].content)
    '''
    for step in sql_app.stream(
        {"messages": [{"role":"user", "content":user_query}]},
        stream_mode = "values",
    ):
        step["messages"][-1].pretty_print()
    
    '''
if __name__ == '__main__':
    main()
