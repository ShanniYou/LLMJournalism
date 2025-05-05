# Shanni You, 2025/04/09

import chromadb
from langchain.vectorstores import Chroma
from chromadb.utils import embedding_functions
import mysqlDBUtils
import llmUtil
from ObjectDef import ArticleHealth
import time

TEST_COLLECTION = 'Test_Collection'
LOCAL_PATH = '/Users/syou/Codes'

# This is the class for a single collection to group all the functions.
class ChromaClientND:
    def __init__(self):
        self.client = chromadb.PersistentClient(path = LOCAL_PATH) # Need to fill in the local path
        client.heartbeat()  # Check for the client running
        self.emFun = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = 'all-mpnet-base-v2')
        self.cName = None

    def getCollection(collection_name):
        # Params: embedding function
        #         collection name
        self.cName = collection_name
        collection = self.client.get_or_create_collection(name = collection_name, embedding_function = emFun)
        return collection




