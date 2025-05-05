# To figure out 

import ollama
from ollamaTools import *
import inspect
import json
from typing import get_type_hints

import requests
import rich
from rich import print_json


print('hello')
client = ollama.Client(host='http://ollama:11434')

response = client.chat(
    model = 'mistral',
    messages = [
        {
            'role':'user',
            'content':'please help me write a SQL query for searching all the student name'
        }
    ]
)

print(response['message']['content'])



  
def main():

    # Testing
    print(get_weather("Beijing"))
    print(calculate("what is 18 minus 1?"))

    functions_prompt = f"""
    you have access to the following tools:
    {function_to_json(get_weather)}
    {function_to_json(calculate)}

    you must follow these instructions:
    Always select one or more of the above tools base on the user query
    If a tool is found, you must respond in the JSON format matching the following schema:
    {{
        "tools": {{
            "tool": "<name of the selected tool>,
            "tool_input": <parameters for the selected tool, matching the tool's JSON schema
        }}
    }}
    If there are multiple tools required, make sure a list of tools are returned in a JSON array.
    If there is no tool that match the user request, you will respond with empty json.
    Do not add any additional Notes or Explanations

    User Query:
    """
    prompts = [
        "what's the weather in Beaverton, US?",
        "how much is 1+1 equals to?"
    ]

    for prompt in prompts:
        print(f"{prompt}??")
        question = functions_prompt + prompt
        response = generate_full_completion(GPT_MODEL, question)

        try:
            tidy_response = (
                response.get("response", response)
                .strip()
                .replace("\n", "")
                .replace("\\", "")
            )
            print_json(tidy_response)
            rich.print(
                f"[bold]Total duration: {int(response.get('total_duration')) // 1e9} seconds [\bold]"

            )
        except Exception:
            print(f"X Unable to decode JSON. {response}")


if __name__ == '__main__':
    main()