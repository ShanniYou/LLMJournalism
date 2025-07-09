# Written by Shanni You, 05/05/2025
# Imports

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import datetime
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
#from bs4 import BeautifulSoup
import requests
import re
import json
from langchain_community.callbacks import get_openai_callback
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
collector = RunCollectorCallbackHandler()
from algoliasearch.search.client import SearchClientSync
from json import loads
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# Packages for local embedding from HuggingFace
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

##############          Constants         ##############
APP_ID = 'IZXD45EXH7'
API_KEY = '1380221e0fdaf65262fc627c43ab1069'
INDEX_NAME = 'prod_ace_automations'

def articleViews():
    # This function is to struture and view how the articles are stored in Algolia

    client = SearchClientSync(APP_ID, API_KEY)
    response = client.browse(INDEX_NAME)
    hits = []
    for hit in response:
        hits.append(hit)

    recordObjects = hits[30]               # identify as the article structure from Algolia
    number_hit = len(recordObjects[1])      # the number of total articles within 1 hit = 1000, just so for validation
    print("Total articles per hit: ", number_hit)

    print(recordObjects[1][1])   # first article to figure out the structure

    print(type(recordObjects[1][1])) # objectID is the unique identifier for each article

    recordObject = dict(recordObjects[1][1])
    '''
    for key, value in recordObject.items():
        print(key, ":", value)
    '''

    print("Page content: ", recordObject['body'])
    print("Title: ", recordObject['headline'])
    print("Authors: ", recordObject['authors'])
    print("Publish date: ", recordObject['publishedDate'])
    print("Top image URL: ", recordObject['topElement']['baseUrl'])
    print("Top image caption: ", recordObject['topElement']['caption'])
    print("Keywords: ", recordObject['parent']['title'] + ", " + " ".join(recordObject['search_tags']))
    print("Source URL: ", recordObject['url'])
    print("Location: ", recordObject['location'])
    
    return recordObject

def get_article_algolia():
    pass
    
################       Indexing        #############
def algolia_to_docs(recordObject):
    # This is a function to convert the Algolia article to a Document object
    print("Type name:", recordObject.get('__typename'))
    '''
    print("Testing if indexing is working")
    print("body", recordObject.get('body'))
    print("authors", recordObject['authors'])
    print("publishedDate", recordObject['publishedDate'])
    print("top_image_url", (recordObject.get('topElement') or {}).get('baseUrl'))
    print("top_image_caption", (recordObject.get('topElement') or {}).get('caption'))
    print("keywords", (recordObject.get('parent') or {}).get('title') + ", " + " ".join(recordObject['search_tags']))
    print("url", recordObject['url'])
    print("location", recordObject.get('location')[0])
    print("location", type(recordObject.get('location')[0]))
    print("location", (recordObject.get('location') or [{}])[0].get('name'))
    '''
    if recordObject.get('__typename') == 'Article':
        try:
            return Document(
                page_content=recordObject['body'],
                metadata={
                    "title": recordObject.get('headline') or None,
                    "authors": recordObject.get('authors') or None,
                    "publish_date": recordObject.get('publishedDate') or None,
                    "top_image_url": (recordObject.get('topElement') or {}).get('baseUrl') or None,
                    "top_image_caption": (recordObject.get('topElement') or {}).get('caption') or None,
                    "keywords": (recordObject.get('parent') or {}).get('title') + ", " + " ".join(recordObject.get('search_tags')) or None,
                    "source_url": recordObject.get('url') or None,
                    "location": (recordObject.get('location') or [{}])[0].get('name') or None
                }
            )
        except Exception as e:
            print(f"Error processing {recordObject}: {e}")
            print("Error: ", e)
            print("Article not added to docs")
            return None

###### Sanitize and Check the metadata ######
def sanitize_metadata_fields(documents: list[Document]) -> list[Document]:
    # Clean and standardize metadata fields to ensure Chroma compatibility
    cleaned_docs = []
    for doc in documents:
        '''
        if len(doc.page_content.split()) >= 8000:
            print("Page content is too long, skipping...")
            print("Page content: ", len(doc.page_content))
            #print("Page content: ", doc.metadata['source_url'])
            continue
        
        if len(doc.page_content.split()) <= 20:
            print("Page content is empty, skipping...")
            print("Page content: ", doc.page_content)
            continue
        '''
        cleaned_metadata = {}
        for key, value in doc.metadata.items():
            # 1. skip None, [], {}, and empty strings
            if value in [None, "", [], {}]:
                continue
            
            # 2. list -> string
            elif isinstance(value, list):
                joined = ", ".join(str(v) for v in value)
                if joined:
                    cleaned_metadata[key] = joined
            
            # 3. datetime -> string
            elif isinstance(value, datetime.datetime):
                cleaned_metadata[key] = value.isoformat()

            # 4. Other types -> string
            elif not isinstance(value, str):
                cleaned_metadata[key] = str(value)

            else:
                cleaned_metadata[key] = value
        cleaned_docs.append(Document(page_content=doc.page_content, metadata=cleaned_metadata))
    return cleaned_docs

def docs_from_doc(recordObjects):
    docs = []
    for recordObject in recordObjects:
        #print("Record Object: ", recordObject)
        article = algolia_to_docs(recordObject)
        #print("Article: ", article)
        if article is not None:
            docs.append(article)
            #print("Article added to docs", docs[-1])
        #break
    return docs

def json_to_recordObjects():
    recordObjects = []
    with open('algolia_article.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split(',', maxsplit=3)
            recordObject = json.loads(line[-1])
            recordObjects.append(recordObject)
        
    return recordObjects

class hf_embedding_function:
    def __init__(self, model_name="BAAI/bge-small-en"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def embed_documents(self, texts:list[str]) -> list[list[float]]:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy().tolist()

    def embed_query(self, text:str) -> list[float]:
        return self.embed_documents([text])[0]

import time

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name = model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)
def main_hf_test():
    # So this works, to test for the HuggingFace embedding function
    
    print("Testing the HuggingFace embedding function")

    vectors = hf.embed_documents(["This is a test document.", "This is another test document."])
    print("Vectors: ", vectors)
    
    print("######################## Embedding and storing in Chroma ########################")
    # Embedding
    load_dotenv()
    vectorstore = Chroma(collection_name = "hf_articles_all", 
                        #embedding_function=OpenAIEmbeddings(),
                        embedding_function = hf,
                        persist_directory="./chroma_articles_hf_db",)


    # Test embedding function
    print("Testing the HuggingFace embedding function with Chroma")
    test_docs = [Document(page_content="This is a test document.", metadata={"source": "test1"}),
                 Document(page_content="This is another test document.", metadata={"source": "test2"})]
    
    vectorstore.add_documents(documents = test_docs,
                            ids = ["docs1", "docs2"],)
    print("Test documents added to Chroma")
    

def main():

    print("######################## This is what we are loading ########################")
    recordObjects = json_to_recordObjects()
    print(recordObjects[0])
    print("######################## Passing down to docs        ########################")
    docs0 = docs_from_doc(recordObjects)
    print("######################## Cleaning the docs   ########################") 
    print("Number of Documents before filtered: ", len(docs0))
    docs = sanitize_metadata_fields(docs0)
    print("Number of Documents after filtered: ", len(docs))
    print("######################## Check the length of the docs ########################")
    article_length = []
    for doc in docs:
        #print("Page content: ", doc.page_content)
        #article_length.append(len(doc.page_content))   # This only count the characters
        article_length.append(len(doc.page_content.split()))   # This count the words
        #print("Page content: ", len(doc.page_content.split()), " words")
        #print("Page content: ", article_length[-1], " words")
        #print("Page content: ", doc.page_content)
        #break
    
    print("Number of articles: ", len(article_length))
    print("Average article length: ", np.mean(article_length))
    print("Max article length: ", np.max(article_length))
    print("Min article length: ", np.min(article_length))
    print("Standard deviation of article length: ", np.std(article_length))
    print("Median article length: ", np.median(article_length))


    '''
    plt.hist(article_length, bins=20, edgecolor='black')
    plt.xlabel('Article Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Article Lengths')
    plt.show()
    '''


    #print('######################## Chunking and splitting ########################')
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
    #chunks = text_splitter.split_documents(doc)
    #print("Number of chunks: ", len(chunks))
    
    print("######################## Embedding and storing in Chroma ########################")
    # Embedding
    load_dotenv()
    vectorstore = Chroma(collection_name = "hf_articles_all", 
                        #embedding_function=OpenAIEmbeddings(),
                        embedding_function = hf,
                        persist_directory="./chroma_articles_hf_db",)

    # Batch embedding:
    batch_size = 128
    total_docs = len(docs)
    for i in tqdm(range(0, total_docs, batch_size), desc="Embedding articles", unit="batch"):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        # Optional: Add a sleep to avoid hitting rate limits
        #time.sleep(1)
    #vectorstore.add_documents(chunks)
    #vectorstore.persist()  
    print("######################## Finished embedding and storing ########################")
    
if __name__ == '__main__':
    main()













