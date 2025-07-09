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

# Crashed at: Cursor:  AgC8AgEcY29udGVudGlkL2FydGljbGUuMS4yMDIzODAxNg== total hits: 322

# This is the funtion to turn algolia article to json file:
def algolia_to_json():
    with open('algolia_article.json', 'a', encoding='utf-8') as f:
        cursor = 'AgC8AgEcY29udGVudGlkL2FydGljbGUuMS4yMDIzODAxNg=='
        aritcle_num = 323

        client = SearchClientSync(APP_ID, API_KEY)
        #response = client.browse(INDEX_NAME)
        response = client.browse(INDEX_NAME, {'cursor': cursor})

        hits = []
        for hit in response:
            hits.append(hit)

        cursor = hits[-1][-1]
        

        for i in range(len(hits[30][1])):
            try:
                f.write(str(aritcle_num) + ',' + str(i) + ',' + cursor + ',' + json.dumps(dict(hits[30][1][i])) + '\n')
            except:
                pass

        while cursor:
            #print("Cursor: ", cursor, 'total hits:', len(objects))
            print("Cursor: ", cursor, 'total hits:', aritcle_num)
            response = client.browse(INDEX_NAME, {'cursor': cursor})

            if response:
                hits = []
                for hit in response:
                    hits.append(hit)

                #objects.append(hits[30])
                cursor = hits[-1][-1]
                aritcle_num += 1

                for i in range(len(hits[30][1])):
                    try:
                        f.write(str(aritcle_num) + ',' + str(i) + ',' + cursor + ',' + json.dumps(dict(hits[30][1][i])) + '\n')
                    except:
                        pass
            else:
                break

        print("Current page: ", aritcle_num)

def json_to_recordObjects():
    data = []
    with open('algolia_article.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split(',', maxsplit=3)
            recordObject = json.loads(line[-1])
            print(recordObject)
            break
        

    #print(type(data[0]))
    #print(len(data[0]))
    #print(data)


def main():
    json_to_recordObjects()

if __name__ == "__main__":
    main()
