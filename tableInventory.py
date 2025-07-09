# Shanni You, 06/11/2025
# This is a file to check the database and then provide additional table information

from langchain_community.utilities import SQLDatabase
from langchain.globals import set_verbose
import requests
import re
import json
import tqdm
from time import sleep
from dotenv import load_dotenv
load_dotenv()
from populateRAG import *
import requests
from newspaper import Article
from langchain.text_splitter import TokenTextSplitter

##################### Setting Constants ######################
sql_address = 'mysql+mysqlconnector://web:JAzv3WHQh-y@newsdayinteractive.cr2zrybivkdw.us-east-1.rds.amazonaws.com:3306/'



def table_inventory_framework_creation(db_name):
    
    print(("######################### This is Table Inventory for Database: ", db_name, " #########################"))
    db = SQLDatabase.from_uri(sql_address + db_name)
    table_names = db.get_usable_table_names()
    print("Table names: ", type(table_names), table_names[:10])
    all_tables = {}
    for table_name in table_names:
        all_tables[table_name] = {"database": db_name}

    with open("table_inventory.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(all_tables) + '\n')
    print("Table inventory for database", db_name, "has been saved to table_inventory.json")

def table_inventory_building(db_name):
    with open("table_inventory.json", "r", encoding="utf-8") as f:
        all_tables = json.load(f)
    
    print(list(all_tables.keys())[:10])

import pandas as pd
def excelExtract():
    excel_Data = pd.read_excel("nextLi.xlsx", sheet_name=None)
    cSheet = excel_Data['Inventory Existing']

    # Need to filter out the Table Name that have no SQL structure yet
    cSheet_filtered = cSheet[~cSheet['TABLE NAME'].str.contains('no sql', case = False, na = False) & ~cSheet['TABLE NAME'].str.contains('table', case = False, na = False) & ~cSheet['TABLE NAME'].str.contains('carto', case = False, na = False)]
    # filter out never published tables
    cSheet_filtered = cSheet_filtered[~cSheet_filtered['NOTES'].str.contains('never published', case = False, na = False) & ~cSheet_filtered['NOTES'].str.contains('not published yet', case = False, na = False)]
    #print(cSheet_filtered['TABLE NAME'].head(10))
    #print("Total number of tables in the sheet:", len(cSheet_filtered['TABLE NAME']))
    '''
    table = '061418_college_costs'
    
    '''
    return cSheet_filtered

def table_puller(table_name):
    print("pulling table information for:", table_name)
    cSheet = excelExtract()
    # See if it is in the table inventory
    table_check = cSheet[cSheet['TABLE NAME'].str.contains(table_name)]
    #print("Table check result:", table_check)
    #print("Columns in the table:", table_check.columns.tolist())
    
    url = table_check['URL'].values[0] if 'URL' in table_check.columns and not table_check.empty else None
    if table_check.empty:
        #print(f"Table {table_name} not found in the inventory.")
        return None, ""
    elif url.startswith('http'):
        try:
            final_article = ""
            article = Article(url)
            print("Downloading article from URL:", url)
            article.download()
            article.parse()
            final_article = article.text
            return table_check, final_article
        except Exception as e:
            print(f"Error downloading article from {url}: {e}")
            return table_check, ""
    else:
        return table_check, ""

def schema_to_docs(json_file):
    # Load the json file and convert it to a list of Document objects
    # This modify version is to combine the schema json file (already processed by llm to generate description)
    # And add on accurate description of the table from the excel file (data inventory from NextLi)
    # This current function is for the daily schema process
    docs = []
    with open(json_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            data = json.loads(line)
            #print("Processing line: ", data)
            # Here should goes with a function to insert information from the excel file to add on
            excel_data, article = table_puller(data['table_name'])
            if excel_data is None:
                print(f"Table {data['table_name']} not found in the inventory.")
                doc = Document(
                    page_content=data['description'] + article,
                    metadata={
                        "table_name": data["table_name"] or None,
                        "description": data["description"] or None,
                        "columns": data["columns"] or None,
                        "tags": data["tags"] or None,
                        "data_category": None,
                        "source": None,
                        "method": None,
                        "topic": None,
                        "last_updated": None,
                        "url": None,
                        "associate_article": ""
                    }
                )
            else:
                doc = Document(
                    page_content=data['description'] + article,
                    metadata={
                        "table_name": data["table_name"] or None,
                        "description": data["description"] or None,
                        "columns": data["columns"] or None,
                        "tags": data["tags"] or None,
                        "data_category": excel_data['DATA CATEGORY'].values[0] if 'DATA CATEGORY' in excel_data.columns else None,
                        "source": excel_data['SOURCE'].values[0] if 'SOURCE' in excel_data.columns else None,
                        "method": excel_data['METHOD'].values[0] if 'METHOD' in excel_data.columns else None,
                        "topic": excel_data['TOPIC'].values[0] if 'TOPIC' in excel_data.columns else None,
                        "last_updated": excel_data['LAST UPDATED'].values[0] if 'LAST UPDATED' in excel_data.columns else None,
                        "url": excel_data['URL'].values[0] if 'URL' in excel_data.columns else None,
                        "associate_article": article
                    }
                )
            #print(doc.metadata['accocaiate_article'])
            #print(doc.page_content[:200])  # Print first 200 characters of the content
            #break
            docs.append(doc)
    return docs

from bs4 import BeautifulSoup
def wordPress():
    base_url = "https://stage.next.newsday.com/wp-json/wp/v2/posts"
    params = {
        'per_page': 100,  # Number of posts per page
        'page': 1         # Start from the first page
    }
    all_posts = []

    while True:
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching posts: {response.status_code}")
            break
        
        posts = response.json()
        if not posts:
            break
        all_posts.extend(posts)
        params['page'] += 1
        '''
        print("post id:", posts[0]["id"])
        print("Date:", posts[0]["date"])
        print("Title:", posts[0]["title"]["rendered"])
        #print("Content:", posts[0]["content"]["rendered"])
        raw_html = posts[0]["content"]["rendered"]
        soup = BeautifulSoup(raw_html, 'html.parser')
        text_content = soup.get_text()

        print("Text Content:", text_content)  # Print first 200 characters of text content
        print("Link:", posts[0]["link"])
        print("Tags:", posts[0].get("tags", []))
        print("Categories:", posts[0].get("categories", []))
        print("Author:", posts[0]["author"])
        break
        '''

    print(f"Total posts fetched: {len(all_posts)}")

    return all_posts

def wordPress_to_docs(posts):
    docs = []
    for post in tqdm(posts):
        raw_html = post["content"]["rendered"]
        soup = BeautifulSoup(raw_html, 'html.parser')
        text_content = soup.get_text()

        doc = Document(
            page_content=text_content,
            metadata={
                "title": post["title"]["rendered"] or None,
                "author": post["author"] or None,
                "published_date": post["date"] or None,
                "top_image_url": None,
                "top_image_caption": None,
                "keywords": post.get("tags", []) or None,
                "source_url": post["link"] or None,
                "location": None
            }
        )
        docs.append(doc)
    return docs

def WP_article_preprocess():
    # This is a function to embed the WordPress articles and store them in Chroma
    WPosts = wordPress()
    docs0 = wordPress_to_docs(WPosts)
    print("Total documents created from WordPress posts:", len(docs0))
    print("########################### Cleaning up the docs ###########################")
    docs = sanitize_metadata_fields(docs0)
    print("Total documents after cleaning:", len(docs))
    print("######################## Embedding and storing in Chroma ########################")
    # Embedding
    load_dotenv()
    vectorstore = Chroma(collection_name = "hf_articles_all", 
                        #embedding_function=OpenAIEmbeddings(),
                        embedding_function = hf,
                        persist_directory="./chroma_articles_hf_db",)
    # Batch embedding:
    batch_size = 12
    total_docs = len(docs)
    for i in tqdm(range(0, total_docs, batch_size), desc="Embedding articles", unit="batch"):
        batch = docs[i:i + batch_size]
        vectorstore.add_documents(batch)
        # Optional: Add a sleep to avoid hitting rate limits
        #time.sleep(1)
    #vectorstore.add_documents(chunks)
    #vectorstore.persist()  
    print("######################## Finished embedding and storing ########################")



def main():

    db_lists = ['baseball_salaries', 'bootleaf_maps', 'circulation', 'daily', 'daycare', 'entertainment', 'geography', 'health', 'job_salaries',
    'luxury_living', 'payrolls', 'politics', 'regions', 'schools', 'shop', 'sports', 'crime_reports', 'dangerous_roads', 'education', 'events',
    'limarathon', 'lisports', 'opinion', 'polling', 'restaurants', 'services', 'social']

    db_name = 'daily'  # Change this to the database you want to process
    #table_inventory_framework_creation("daily")
    #table_inventory_building("daily")
    #excel_info = excelExtract()
    
    json_file = "schema_" + db_name + ".json"
    docs0 = schema_to_docs(json_file)
    print("Number of raw documents: ", len(docs0))
    print("########################### Sanitizing the metadata #########################")
    docs = sanitize_metadata_fields(docs0)
    print("Number of documents after sanitizing: ", len(docs))

    # Chunking and splitting
    print("########################### Chunking and splitting the documents #########################")
    text_splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=50)  # Chunk by tokens not by characters
    chunks = text_splitter.split_documents(docs)
    print("Number of chunks after splitting: ", len(chunks))
    #print("Example chunk content: ", chunks[0].page_content[:200])  
    print("######################## Embedding and storing in Chroma ########################")
    # Embedding
    
    vectorstore = Chroma(collection_name = "hf_schema_all", 
                        #embedding_function=OpenAIEmbeddings(),
                        embedding_function = hf,
                        persist_directory="./chroma_schema_hf_db",)

    # Batch embedding:
    batch_size = 12
    total_docs = len(chunks)
    for i in tqdm(range(0, total_docs, batch_size), desc="Embedding table schema", unit="batch"):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)

    print("######################## Finished embedding and storing ########################")
    
def main_test():
    db_name = 'daily'  # Change this to the database you want to process
    json_file = "schema_" + db_name + ".json"
    docs0 = schema_to_docs(json_file)


if __name__ == "__main__":
    main()