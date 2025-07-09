# Written by Shanni You, 05/05/2025
# Imports
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import datetime
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from bs4 import BeautifulSoup
import requests
import re
import json
from langchain_community.callbacks import get_openai_callback
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from dotenv import load_dotenv
collector = RunCollectorCallbackHandler()

############ INDEXING ############
# Get the url of all newsday articles
# Load Documents and Wrap With Metadata

def get_article_urls(base_url, pages):
    urls = []
    for i in pages:
        page_url = f"{base_url}/{i}"
        html = requests.get(page_url).text
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all("a", href = True):
            href = a['href']
            #print(href)
            if href.startswith("/") and re.search(r'-[a-z0-9]{8}$', href):
                full_url = "https://www.newsday.com" + href
                #print('full link:', full_url)
                urls.append(full_url)
    return list(set(urls))

def url_to_article(url):
    #print("Processing URL: ", url)
    try:
        #print("Downloading article...")
        article = Article(url)
        article.download()
        article.parse()
        return Document(
            page_content=article.text,
            metadata={
                "title": article.title,
                "authors": article.authors,
                "publish_date": article.publish_date,
                "top_image": article.top_image,
                "movies": article.movies,
                "keywords": article.keywords,
                "source_url": article.source_url
            }
        )
    except Exception as e:
        print(f"Error processing {url}: {e}")
        print("Error: ", e)
        print("Article not added to docs")
        return None

def docs_from_url(urls):
    docs = []
    for url in urls:
        article = url_to_article(url)
        #print("Article: ", article)
        if article is not None:
            docs.append(article)
            #print("Article added to docs", docs[-1])
    return docs

###### Sanitize and Check the metadata ######
def sanitize_metadata_fields(documents: list[Document]) -> list[Document]:
    # Clean and standardize metadata fields to ensure Chroma compatibility
    cleaned_docs = []
    for doc in documents:
        cleaned_metadata = {}
        for key, value in doc.metadata.items():
            # 1. skip None, [], {}, and empty strings
            if value in [None, "", [], {}]:
                continue
            
            # 2. list -> string
            elif isinstance(value, list):
                joined = ", ".join(value)
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





def main():
    base_url = "https://www.newsday.com/"
    pages = ['long-island', 'sports', 'news/nation', 'lifestyle/restaurants', 'opinion', 'business', 'lifestyle', 'long-island/education']
    urls = get_article_urls(base_url, pages=pages)
    print("Number of URLs: ", len(urls))
    docs = docs_from_url(urls)
    print("Number of Documents: ", len(docs))
    # Sanitize and Check the metadata
    doc = sanitize_metadata_fields(docs)

    # Chunking and splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    chunks = text_splitter.split_documents(doc)
    print("Number of chunks: ", len(chunks))

    load_dotenv()
    # Embedding
    vectorstore = Chroma(collection_name = "newsday", 
                        embedding_function=OpenAIEmbeddings(),
                        persist_directory="./chroma_langchain_db",)


    vectorstore.add_documents(chunks)
    #vectorstore.persist()  
    



if __name__ == '__main__':
    main()













