# By Shanni You, 05/20/2025
# This is a RAG pipeline to load the SQL table schema and put it into the vector database
# The goal is so it can retrieve the related table from the database

from langchain_community.utilities import SQLDatabase
from langchain.globals import set_verbose
import requests
import re
import json
import tqdm
from time import sleep
from populateRAG import *
from RAG import *
from langchain.schema import Document
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
load_dotenv()



model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name = model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)

#################### Setting Constants ######################
llmmodel = "mistral"
set_verbose(True)
sql_address = 'mysql+mysqlconnector://web:JAzv3WHQh-y@newsdayinteractive.cr2zrybivkdw.us-east-1.rds.amazonaws.com:3306/'

#################### Testing the prompt for Ollama ##########
def query_to_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": llmmodel,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 1000,
        },
    )
    return response.json()['choices'][0]['message']['content']
print("######################### This is Testing for Mistral #########################")
print(query_to_ollama("What is the capital of France?"))  # Expected output: Paris
print("######################### Generating JSON for Table Schema #########################")

def db_preprocess(db_name):
    db = SQLDatabase.from_uri(sql_address + db_name)
    table_names = db.get_usable_table_names()
    print("Table names: ", type(table_names), table_names[:10])
    # Loop through each table and get the schema
    for table_name in tqdm(table_names):
        json_temp = table_to_dict(db, table_name)
        with open("schema_" + db_name + ".json", "a", encoding = "utf-8") as f:
            f.write(json.dumps(json_temp) + '\n')
            #f.write(json_temp + "\n")
            #json_string = json.dumps(json_template, ensure_ascii=False, indent=4)
        print("Table: ", table_name, " has been processed and saved to "+ "schema_" + db_name + ".json")
        #break

def table_to_dict(db, table_name):
    # Process 1 table and convert it to a dictionary according to the json template
    #print("Processing table: ", table_name)
    table_info = db.get_table_info([table_name])
    #print(table_info)
    #print(type(table_info))
    # Generate the json template for the table
    json_template = {
        "table_name": table_name,
        "description": "",
        "columns": [],
        "tags": []
        #"source": ""
    }
    column_info = column_parser(table_info)
    json_template["columns"].append(column_info)

    # Add the description and tags
    json_template["description"] = query_to_ollama(f"Please provide a description for the table {table_name}. This is the columns {column_info} that it has. You need to understand the whole picture of the table and can you limit it to up to 5 sentences? don't just put the table column.")
    json_template["tags"] = query_to_ollama(f"Please provide a list of tags for the table {table_name}. This is the columns {column_info} that it has. You need to understand the whole picture of table and can you limit it to 5 tages? don't just put the table column.")
    #json_template["source"] = query_to_ollama(f"Please provide the source for the table {table_name}")
    # Convert the dictionary to a json string
    
    return json_template


def column_parser(table_info):
    # Table info is a string, and we want to get the table schema
    match = re.search(r"CREATE TABLE `(\w+)` \((.+?)\)", table_info, re.DOTALL)
    if not match:
        return None
    
    table_name = match.group(1)
    columns = match.group(2).split(",\n")[0]
    #print("Columns: ", columns, type(columns))
    #print("Table name: ", table_name)

    column_info = []
    cols = columns.split(",")
    #print("Column: ", cols, type(cols))
    for col in cols:
        # Clean it first
        col = col.replace("\ufeff", "").replace("\n", "").replace("\t", "").strip()
        #print("Column: ", col)
        parts = col.strip().split()
        #print("Parts: ", parts, len(parts))
        if len(parts) >= 2:
            column_name = parts[0]
            data_type = parts[1]
            is_nullable = "NULL" in col
            description = query_to_ollama(f"Please provide a description for the column {column_name} in the table {table_name}, and keep it short within 2 sentences.")
            #print("Description: ", description)
            column_info.append({
                "column_name": column_name,
                "description": description,
                "data_type": data_type,
                "is_nullable": is_nullable
            })
        else:
            print("Column not found: ", col)
        #break
    #print("Column info: ", column_info)
    return column_info

def schema_to_docs(json_file):
    # Load the json file and convert it to a list of Document objects
    docs = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            data = json.loads(line)
            #print("Processing line: ", data)
            doc = Document(
                page_content=data['description'],
                metadata={
                    "table_name": data["table_name"],
                    "description": data["description"],
                    "columns": data["columns"],
                    "tags": data["tags"]
                }
            )
            docs.append(doc)
    return docs

def main():
    '''
    db_name = 'geography'  # Change this to the database you want to process
    
    # Newsday Records Service that have valid record:
    db_lists = ['baseball_salaries', 'bootleaf_maps', 'circulation', 'daily', 'daycare', 'entertainment', 'geography', 'health', 'job_salaries',
    'luxury_living', 'payrolls', 'politics', 'regions', 'schools', 'shop', 'sports', 'crime_reports', 'dangerous_roads', 'education', 'events',
    'limarathon', 'lisports', 'opinion', 'polling', 'restaurants', 'services', 'social']
      # Change this to the database you want to process
    db_preprocess(db_name)
    # Note:
    # DB processed: daily, dangerous_roads, baseball_salaries, bootleaf_maps, circulation, daycare, entertainment, geopraphy
    
    # Try embedding the table schema into the vector database: schema.json
    # 1. Load the schema.json file, 2. Convert it to a list of Document objects, 3. Sanitize the metadata, 4. Chunk the documents, 5. Embed the documents into the vector database
    
    
    json_file = "schema_" + db_name + ".json"
    print("########################## Loading the JSON file #########################")
    docs = schema_to_docs(json_file)
    print("Number of raw documents: ", len(docs))
    print("########################### Sanitizing the metadata #########################")
    docs = sanitize_metadata_fields(docs)
    print("Number of documents after sanitizing: ", len(docs))

    print("######################## Embedding and storing in Chroma ########################")
    # Embedding
    
    vectorstore = Chroma(collection_name = "hf_schema_all", 
                        #embedding_function=OpenAIEmbeddings(),
                        embedding_function = hf,
                        persist_directory="./chroma_schema_hf_db",)

    # Batch embedding:
    batch_size = 128
    total_docs = len(docs)
    for i in tqdm(range(0, total_docs, batch_size), desc="Embedding table schema", unit="batch"):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)

    print("######################## Finished embedding and storing ########################")
    '''
    # Load the vector store
    print("Reloading the vector store...")
    db = Chroma(collection_name = "hf_schema_all", 
                        embedding_function=hf,
                        persist_directory="./chroma_schema_hf_db",)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    
    print("######################## This is what we are loading ########################")
    docs = db.get(include=["documents"])
    #print(docs["documents"])    # Check if the overlap is too high
    documents = docs["documents"]
    print(len(documents[0].split()), documents[0])
    print("######################## New Docs ########################")
    print(len(documents[1].split()), documents[1])
    print("######################## New Docs ########################")
    print(len(documents[2].split()), documents[2])
    print("######################## New Docs ########################")
    
    ########## Retrieval and Generation ##########
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    print("Prompt: ", prompt)
    # LLM
    llm = ChatOpenAI(model = "gpt-4o", temperature=0)
    #system_prompt = "You are a SQL expert. You have access to the sql table schema, when user ask you a question, you want to find out the schema so then you can help generate sql query later"
    # QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        callbacks=[collector]
    )

    #docs = retriever.get_relevant_documents("what is the consequence of delaying in county subsidies that Nassau day care worry about?")
    response = chain.invoke("I want to know what high school students are studing by the intel competition information.")
    save_rag_to_json(response, filepath='sql_schema_response.json')
    '''



if __name__ == '__main__':
    main()






