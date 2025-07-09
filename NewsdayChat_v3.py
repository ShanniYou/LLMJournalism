# Written by Shanni You, 05/29/2025
# This is the intergration of all the components in NewsdayChat.py
# Data Available:
# - SQL table: Company Investigation Cases
# - Article RAG: Newsday articles
# Add feature: few shot learning for the router to learn how to geneerate sub-questions

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
comLLMHost = ChatOpenAI(model_name=comLLM, temperature=0)
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
#user_query = "I want to know what high school students are studing in the intel competition information."    # This is the query we are going to use for testing
#user_query = test_user_query()['user_query1']  
#user_query = test_user_query()['user_query2'] 
#user_query = test_user_query()['user_query3']  
#user_query = test_user_query()['user_query4']   
#user_query = test_user_query()['user_query5'] 
#user_query = test_user_query()['user_query6']  
#user_query = test_user_query()['user_query7']  
#user_query = test_user_query()['user_query8']   
#user_query = test_user_query()['user_query9']  
#user_query = test_user_query()['user_query10'] 
#user_query = "Could you pull out 10 people in the village of hempstead payroll? and sum up their salary?"  

user_query = "What are new york college students really paying?" 


def main():
    print("The user query that is being tested right now:", user_query)
    # Step 1: The router to decide which model to use
    chain_lists = ['sql_chain', 'article_chain']
    chain_prompt = "Here are the available chains: " + ", ".join(chain_lists) + ".\n" 
    print("Available chains:", chain_prompt)
    llm = CustomHTTPLLM()

    # ReAct-Style Router
    system_prompt = "You are a query decomposition assistant for a hybrid system consisting of: an sql agent that queries a structured investigation data database, " \
                    + "and an article RAG system that retrieves insights from newsday articles. " \
                    + "Given a complex user query, decompose it into exactly three reasoning steps, each aligned with one of our systems or a final synthesis step." \
                    + "Please print out in the following format:\n" \
                    + "Question: {query}\n" \
                    + "Step 1: [sql_chain] - [formulate a sub-question that can be answered using the sql agent on structured data]\n" \
                    + "Step 2: [article_chain] - [formulate a sub-question that searches Newsday articles]\n" \
                    + "Step 3: [final_step] - [using given information and give a comprehensive result to answer: {query}, keep this instruction fixed, don't modify it]\n" \
                    + "Please return only the steps, no extra words, and there will only be 3 steps in total." \
                    + "---\
                    Question: Which people have the same last name in the Village of Hempstead payroll?\
                    Step 1: [sql_chain] - Retrieve employee names from the Village of Hempstead where the last name are the same\
                    Step 2: [article_chain] - What are the last name of people who are mentioned in Newsday Ariticles about the topic of village of Hempstead payroll?\
                    Step 3: [final_step] - Put all the founded last name from the two sources together. \
                    ---\
                    Question: Pull out 10 people in the village of hempstead payroll? and sum up their salary.\
                    Step 1: [sql_chain] - Retrieve the first 10 people in the Village of Hempstead payroll and their salary\
                    Step 2: [article_chain] - Fetch the first 10 people and their salaries in the Village of Hempstead payroll from Newsday article?\
                    Step 3: [final_step] - Sum up the salary of the first 10 people in the Village of Hempstead payroll from the two sources together. \
                    ---\
                    Question: {query}\n" \

    router_prompt = PromptTemplate.from_template(system_prompt)

    router_chain = LLMChain(llm=llm, prompt=router_prompt)
    steps = router_chain.invoke({"query": user_query})['text']
    print("########################################### query decomposition steps ###########################################")
    print(steps)  # Expected to print the steps for the models to follow
    print("#####################################################################################################")
    # Need to parse the steps and then group adjacent sql query
    parsed_steps = parse_steps(steps)
    

    buffer_messages = []
    for step in parsed_steps:
        print(f"Model: {step[0]}, Instruction: {step[1]}")
        if 'sql_chain' in step[0]:
            print("Routing to SQL Agent...")
            sql_app = sql_chain(step[1], comLLM)
            try:
                response = sql_app.invoke({"messages": [("user", step[1])]}, {"recursion_limit": 25})
                response = response['messages'][-1].content
            except Exception as e:
                response = {"messages": "Could not fetch data from SQL Agent"}
                response = response['messages']
            buffer_messages.append("Source (SQL Agent): " + response)
            print("SQL Agent Response:", response)
        elif 'article_chain' in step[0]:
            print("Routing to Article RAG...")
            article_app = article_chain(step[1], comLLM)
            response = article_app.invoke({"query": step[1]})
            buffer_messages.append("Source (Article RAG): " + response['result'])
            print("Article RAG Response:", response['result'])
        elif 'final_step' in step[0]:
            print("Final Step: Combining results from SQL and Article RAG...")
            final_response = "\n ".join(buffer_messages)
            #print("Final Response:", final_response)
            system_prompt = "You are a final answer agent, with the inforamtion provided from two sources (SQL and Article), you try to answer the question: {query}" \
                            + "And please note the source."
            final_prompt = PromptTemplate.from_template(system_prompt)
            final_chian = LLMChain(llm = comLLMHost, prompt = final_prompt)
            response = final_chian.invoke({"query": final_response + user_query})
            print("############################################# User Question ###########################################")
            print(user_query)  # Expected to print the user question
            print("############################################# Query Decomposition Steps ###########################################")
            print(steps)  # Expected to print the steps for the models to follow
            print("############################################# Buffer Messages ###########################################")
            print("Before the final answer, here is the buffer messages: \n", final_response)
            print("############################################# Final Answer ###########################################")
            print("Final Answer:\n", response['text'])

'''
def cal():
    list = 176102 + 149509 + 176102 + 130822 + 157825 + 224496 + 166398 + 150834 + 206536 + 166623
    print("The total number of people in the village of hempstead payroll is: ", list)
'''
if __name__ == "__main__":
    main()
