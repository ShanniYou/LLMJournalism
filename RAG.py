# Shanni You, testing for reloading RAG.py
# Add feature: passing metadata to the chain @ 06/17/2025

#from langchain_community.vectorstores import Chroma
#from langchain.text_splitter import RecursiveCharacterTextSplitter # Character Text Splitter
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

def format_docs(docs):
    """
    Re-formatting the documents to include metadata to contextualize the query
    Article RAG we want the published date and the source URL
    """
    context = []
    for doc in docs:
        # Extracting metadata
        published_date = doc.metadata.get('publish_date', 'Unknown date')
        source_url = doc.metadata.get('source_url', 'Unknown source')
        
        # Formatting the content
        formatted_content = f"""
        [Published Date]: {published_date}
        [Source URL]: {source_url}
        [Content]: {doc.page_content}"""
        context.append(formatted_content)
    return context

def article_chain(user_query, comLLM):
    # Load the vector store
    print("Reloading the vector store...")
    db = Chroma(collection_name = "hf_articles_all", 
                        embedding_function=hf,
                        persist_directory="./chroma_articles_hf_db",)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model = comLLM, temperature=0)
    docs = retriever.get_relevant_documents(user_query)
    ########## Retrieval and Generation ##########
    # Re-formatting the documents to include metadata to contextualize the query
    # Article RAG we want the published date and the source URL
    #context = format_docs(docs)
    #print("Formatted context for the query:", context)
    #custom_prompt = f""" You are a new assistant that helps users to answer their questions based on the context provided.
    #Context: {context}
    #User Query: {user_query}
    #Answer the question based on the context provided. Keep track of the source_url, and also the published date of the article when you are generating the answer. 

    #Here is the answer format:
    #[Answer]: <Your answer here>
    #[Source URL]: <Source URL here>
    #[Published Date]: <Published date here>

    #if there are multiple sources, please list them all in the answer.
    #"""
    #answer = llm.invoke(custom_prompt)
    #print("Answer from the LLM:", answer.content)
    return docs


def main():
    user_query = "What are new york college students really paying?"
    article_chain(user_query, "gpt-4o")

if __name__ == "__main__":
    main()