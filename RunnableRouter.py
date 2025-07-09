# Written by Shanni You, 05/29/2025
# This is a test file for the RunnableRouter implementation with local llm

import requests
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables.router import RouterRunnable
from langchain_core.language_models.llms import LLM
llmmodel = "mistral"  # Specify the model you want to use, e.g., "mistral", "llama3", etc.

def query_to_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": llmmodel,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.5,
            "max_tokens": 1000,
        },
    )
    return response.json()['choices'][0]['message']['content']
#rint("######################### This is Testing for Mistral #########################")
#rint(query_to_ollama("What is the capital of France?"))  # Expected output: Paris


class CustomHTTPLLM(LLM):
    def _call(self, prompt, stop=None):
        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": llmmodel,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5
            },
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    @property
    def _llm_type(self):
        return "custom_http_llm"

# helper function for ReAct-Style Router
def parse_steps(steps):
    lines = steps.strip().split('\n')
    parsed_steps = []
    for line in lines:
        if line.strip():  # Check if the line is not empty
            parts = line.split(' - ')
            if len(parts) == 2:
                model_name, instruction = parts
                parsed_steps.append((model_name.strip(), instruction.strip()))
    return parsed_steps

