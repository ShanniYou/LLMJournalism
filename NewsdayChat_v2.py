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
user_query = test_user_query()['user_query6']  # SQL -> works, repeatable
#user_query = test_user_query()['user_query7']  # SQL -> same
#user_query = test_user_query()['user_query8']  # RAG -> works
#user_query = test_user_query()['user_query9']  # RAG
#user_query = test_user_query()['user_query10'] # RAG
#user_query = "Could you pull out 10 people in the village of hempstead payroll?"
def main():
    print("The user query that is being tested right now:", user_query)
    # Step 1: The router to decide which model to use
    chain_lists = ['sql_chain', 'article_chain']
    chain_prompt = "Here are the available chains: " + ", ".join(chain_lists) + ".\n" 
    print("Available chains:", chain_prompt)
    llm = CustomHTTPLLM()


    # ReAct-Style Router
    system_prompt = "You are a query decomposition assistant for a hybrid system consisting of: an sql agent that queries a structured investiation data database, " \
                    + "and an article RAG system that retrieves insights from newsday articles. " \
                    + "Given a complex user query, decompose it into exactly three reasoning steps, each aligned with one of our systems or a final synthesis step." \
                    + "Please print out in the following format:\n" \
                    + "Question: {query}\n" \
                    + "Step 1: [sql_chain] - [formulate a sub-question that can be answered using the sql agent on structured data]\n" \
                    + "Step 2: [article_chain] - [formulate a sub-question that searches Newsday articles]\n" \
                    + "Step 3: [final_step] - [using given information and give a comprehensive result to answer: {query}, keep this instruction fixed, don't modify it]\n" \
                    + "Please return only the steps, no extra words, and there will only be 3 steps in total."

    router_prompt = PromptTemplate.from_template(
        system_prompt)

    router_chain = LLMChain(llm=llm, prompt=router_prompt)
    steps = router_chain.invoke({"query": user_query})['text']
    print("###########################################")
    print(steps)  # Expected to print the steps for the models to follow
    print("###########################################")
    # Need to parse the steps and then group adjacent sql query
    parsed_steps = parse_steps(steps)
    

    buffer_messages = []
    for step in parsed_steps:
        print(f"Model: {step[0]}, Instruction: {step[1]}")
        if 'sql_chain' in step[0]:
            print("Routing to SQL Agent...")
            sql_app = sql_chain(step[1], comLLM)
            response = sql_app.invoke({"messages": [("user", step[1])]})
            buffer_messages.append("Source 1, from SQL Agent Response: " + response["messages"][-1].content)
            print("SQL Agent Response:", response["messages"][-1].content)
        elif 'article_chain' in step[0]:
            print("Routing to Article RAG...")
            article_app = article_chain(step[1], comLLM)
            response = article_app.invoke({"query": step[1]})
            buffer_messages.append("Source 2, from Article RAG Response: " + response['result'])
            print("Article RAG Response:", response['result'])
        elif 'final_step' in step[0]:
            print("Final Step: Combining results from SQL and Article RAG...")
            final_response = "\n ".join(buffer_messages)
            #print("Final Response:", final_response)
            system_prompt = "You are a final answer agent, please using the following format:" \
                            + "From SQL table, [information from sql agent]" \
                            + "From Article RAG, [information from article rag]" \
                            + "The final conclusion for the {query} is: [...]" \
                            + "Please return only the final answer, and the information source. no extra words."
            final_prompt = PromptTemplate.from_template(system_prompt)
            final_chian = LLMChain(llm = llm, prompt = final_prompt)
            response = final_chian.invoke({"query": final_response + user_query})
            print("###########################################")
            print(steps)  # Expected to print the steps for the models to follow
            print("###########################################")
            print("Before the final answer, here is the buffer messages: \n", final_response)
            print("###########################################")
            print("Final Answer:\n", response['text'])

    '''
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
    '''
if __name__ == "__main__":
    main()
