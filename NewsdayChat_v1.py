# Written by Shanni You, 05/29/2025
# This is the intergration of all the components in NewsdayChat.py
# Data Available:
# - SQL table: Company Investigation Cases
# - Article RAG: Newsday articles

###########################################################################
# Here are all the packages that we need to import
###########################################################################
from sqlAgent import *
from RAG import *
from RunnableRouter import *
###########################################################################
# Here are all the constants that includes local llms and ChatGPT models
###########################################################################
set_verbose(True)
comLLM = "gpt-4o"                  # Commercial LLM for accurate and reliable responses
localLLM = "mistral"               # Local LLM for cost-effective responses

model_name = "BAAI/bge-small-en"   # Embedding model for RAG semantic search
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(model_name = model_name, model_kwargs = model_kwargs, encode_kwargs = encode_kwargs)

def test_user_query():
    # need to load queries from json file

    with open('QueryLibs.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data
#user_query = "I want to know what high school students are studing by the intel competition information."    # This is the query we are going to use for testing
#user_query = test_user_query()['user_query1']  # SQL -> more than 25 attempts, need to stop infinite loop
#user_query = test_user_query()['user_query2']  # SQL -> same
#user_query = test_user_query()['user_query3']  # SQL -> same
#user_query = test_user_query()['user_query4']  # SQL -> same
#user_query = test_user_query()['user_query5']   # SQL -> same
#user_query = test_user_query()['user_query6']  # SQL -> works, repeatable
#user_query = test_user_query()['user_query7']  # SQL -> same
#user_query = test_user_query()['user_query8']  # RAG -> works
user_query = test_user_query()['user_query9']  # RAG
#user_query = test_user_query()['user_query10'] # RAG
#user_query = "Could you pull out 10 people in the village of hempstead payroll?"
def main():
    print("The user query that is being tested right now:", user_query)
    # Step 1: The router to decide which model to use
    chain_lists = ['sql_chain', 'article_chain']
    chain_prompt = "Here are the available chains: " + ", ".join(chain_lists) + ".\n" 
    print("Available chains:", chain_prompt)
    llm = CustomHTTPLLM()


    # 
    system_prompt = "You are a helpful assistant that routes queries to the appropriate model.\n" + "Here is the query: {query}\n" + "Please determine which model should handle this query and return only the chain name. Don't say extra words" + chain_prompt
    router_prompt = PromptTemplate.from_template(
        system_prompt)

    router_chain = LLMChain(llm=llm, prompt=router_prompt)

    route = router_chain.invoke({"query":user_query})['text'].replace(" ","")
    print("Route:", route)  # Expected to print the chain name, e.g., "article_chain"

    if 'sql_chain' in route:
        print("Routing to SQL Agent...")
        sql_app = sql_chain(user_query, comLLM)
        response = sql_app.invoke({"messages": [("user", user_query)]})
        print("SQL Agent Response:", response["messages"][-1].content)

    elif 'article_chain' in route:
        print("Routing to Article RAG...")
        article_app = article_chain(user_query, comLLM)
        print(article_app.invoke(user_query)['result'])

    print("All done!")
if __name__ == "__main__":
    main()
