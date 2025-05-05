# Shanni You, 04/10/2025
# Funcitionality including:
#                           - Loading tables from SQL
#                           - Option to preprocessing as text

import mysql.connector
import chromadb
import json


def table(tableName = None):
    # If tableName == None, we look through the table structure
    results = [] # store all the row info
    colName = None
    # Connection config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Load Query from SQL
    with open('query.sql', 'r') as f:
        query = f.read()


    #print('print out the query, just for checking \n', query)

    # Connect to the server
    try:
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()

        # Run a query
        # Placeholder
        #with conn.cursor() as cursor:
        cursor.execute(query)
        
        results.append(cursor.fetchall())
        #print(results, '1')
        while cursor.nextset():
            if cursor.with_rows:
                colName = [col[0] for col in cursor.description]
            results.append(cursor.fetchall())
            #print(results)
        
    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close() 

    return results, colName
    #print("Hello World!")

#Helper functions to get from the Company database
#cTable, colName = table() # Currently fetching all the information
    #print(type(cTable[-1]))
    #print(colName, cTable[-1][:10])

    #tableName = 'car accidents'
    #paramID = colName[0]
    
    #print(db_query_tool(cTable[-1][:10]))
    # Just modify the query file
    

    #db = SQLDatabase.from_uri(f'mysql+pymysql://web:JAzv3WHQh-y@newsdayinteractive.cr2zrybivkdw.us-east-1.rds.amazonaws.com:3306/daily')