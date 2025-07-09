# Shanni You, testing for reloading RAG.py

#from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import datetime
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import requests
import re
import json
from langchain_community.callbacks import get_openai_callback
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
collector = RunCollectorCallbackHandler()
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
load_dotenv()


model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name = model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)

def validate_docs(docs, max_total_tokens=3000, max_chunk_tokens=500, model="gpt-4o"):
    from tiktoken import encoding_for_model
    enc = encoding_for_model(model)

    total = 0
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc.page_content)
        if len(tokens) > max_chunk_tokens:
            print(f"Document {i} exceeds max chunk size: {len(tokens)} tokens")
        total += len(tokens)

    if total > max_total_tokens:
        print(f"Total tokens exceed max size: {total} tokens")
        return False
    
    print(f"Total tokens: {total}, looks good!")
    return True

# Save to json to check the result and the workflow
def save_rag_to_json(response, filepath='rag_response.json'):
    result = {
        "result": response['result'],
        "source_documents": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in response['source_documents']
        ]
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

def main():
    # Load the vector store
    print("Reloading the vector store...")
    db = Chroma(collection_name = "hf_articles_all", 
                        embedding_function=hf,
                        persist_directory="./chroma_articles_hf_db",)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    '''
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
    print(len(documents[3].split()), documents[3])
    print("######################## New Docs ########################")
    print(len(documents[4].split()), documents[4])
    print("######################## New Docs ########################")
    print(len(documents[5].split()), documents[5])
    print("Number of documents: ", len(documents))
    #print("Total docs lens: ", len(docs['ids']))

    
    docs = retriever.get_relevant_documents("Is there story about car accident or payrolls?")

    for doc in docs:
        
        print("Doc: ", doc.page_content)
        print("Meta: ", doc.metadata)
    
    '''
    ########## Retrieval and Generation ##########
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    print("Prompt: ", prompt)
    # LLM
    llm = ChatOpenAI(model = "gpt-4o", temperature=0)

    # QA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        callbacks=[collector]
    )

    #docs = retriever.get_relevant_documents("what is the consequence of delaying in county subsidies that Nassau day care worry about?")
    response = chain.invoke("what is the consequence of delaying in county subsidies that Nassau day care worry about?")
    save_rag_to_json(response, filepath='rag_response.json')
    

   
if __name__ == '__main__':
    main()
