# Shanni You, to test out the Algolia through Python 

from algoliasearch.search.client import SearchClientSync
from json import loads
import json
import gc
import time

APP_ID = 'IZXD45EXH7'
API_KEY = '1380221e0fdaf65262fc627c43ab1069'
INDEX_NAME = 'prod_ace_automations'
#INDEX_NAME = 'prod_ace'



client = SearchClientSync(APP_ID, API_KEY)
response = client.browse(INDEX_NAME)
    
hits = []
for hit in response:
    hits.append(hit)
    
    cursor = hits[-1][-1]
    aritcle_num = 0
    
while cursor:
    print("Cursor: ", cursor, 'total hits:', aritcle_num)
    response = client.browse(INDEX_NAME, {'cursor': cursor})
    if response:
        hits = []
        for hit in response:
            hits.append(hit)

        cursor = hits[-1][-1]
        aritcle_num += 1
        
        time.sleep(0.1)
    else:
         break


print("Total objects: ", aritcle_num)

