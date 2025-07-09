# Shanni You, @04/17/2025
# Implemented with stateGraph, a package from langchain, to define the agent
# LLM: Chatgpt-4, API stored in .env file
# Add feature: use sql_RAG to retrieve the relevant tables only

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
include_tables = table_lists)
#sample_rows_in_table_info = 0)  # Test database
#print(db.dialect)
print(db.get_usable_table_names())
#print(db.get_table_info())
#print(db.run('SELECT * FROM Artist LIMIT 10;'))

# Utilities getting the table name and the schema:
toolkit = SQLDatabaseToolkit(db = db, llm = ChatOpenAI(model = llmmodel, max_retries = 3, temperature=0))
tools = toolkit.get_tools()

list_tables_tool = next(tool for tool in tools if tool.name == 'sql_db_list_tables')
get_schema_tool = next(tool for tool in tools if tool.name == 'sql_db_schema')

print(list_tables_tool.invoke(""))  # manually calling the table names
print(get_schema_tool.invoke('120111_intel_semifinalists')) # DDL for Artist
