# Shanni You @ 04/22/2025
# This is the testing tool functions for Local Ollama

import inspect
import json
from typing import get_type_hints

import requests
import rich
from rich import print_json
import re


GPT_MODEL = 'Mistral'

class Weather:
    def __init__(self, city, temperature, condition):
        self.city = city
        self.temperature = temperature
        self.condition = condition
    
    def __repr__(self):
        return f"Weather in {self.city}: {self.temperature} 'C, {self.condition} "


def get_weather(city: str) -> Weather:
    """ Get the current weather given a city """
    fake_weather = {
        "New York": ("15", "Cloudy"),
        "Beijing": ("22", "Sunny"),
        "Tokyo": ("18", "Rainy"),
        "Pairs": ("17", "Windy"),
    }

    temp, cond = fake_weather.get(city, ("20", "Unknown"))

    return Weather(city=city, temperature = temp, condition = cond)

class CalculationResult:
    def __init__(self, expression: str, result: float):
        self.expression = expression
        self.result = result
    
    def __repr__(self):
        return f"Calculation: {self.expression} = {self.result}"


def calculate(prompt: str) -> CalculationResult:
    """ parse a prompt and perform basic arithmetic operations.  """
    pattern = r"(\d+\.?\d*)\s*([+\-*/])\s*(\d+\.?\d*)"
    prompt = prompt.lower()
    prompt = prompt.replace("plus", "+")
    prompt = prompt.replace("minus", "-")
    prompt = prompt.replace("times", "*")
    prompt = prompt.replace("multiplied by", "*")
    prompt = prompt.replace("divided by", "/")
    prompt = prompt.replace("over", "/")
    prompt = prompt.replace("x", "*")

    match = re.search(pattern, prompt)
    print(prompt, match)

    if not match:
        raise ValueError("Could not parse a valid calculation from the input.")

    num1, operator, num2 = match.groups()
    num1 = float(num1)
    num2 = float(num2)

    # do calculation
    if operator == "+":
        result = num1 + num2
    elif operator == "-":
        result = num1 - num2
    elif operator == "*":
        result = num1 * num2
    elif operator == "/":
        if num2 == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        result = num1 / num2
    else:
        raise ValueError(f"Unsupported operator: {operator}")

    return CalculationResult(expression=f"{num1} {operator} {num2}", result=result)




def generate_full_completion(model:str, prompt: str, **kwargs) -> dict[str, str]:
    params = {"model": model, "prompt": prompt, "stream": False}

    try: 
        response = requests.post(
            "http://ollama:11434/api/generate",
            headers = {"Content-Type": "application/json"},
            data = json.dumps(params),
            timeout = 60,
        )
        response.raise_for_status()
        return json.loads(response.text)

    except requests.RequestException as err:
        return {"error": f"API call error: {str(err)}"}

def get_type_name(t):
    name = str(t)
    if "list" in name or "dict" in name:
        return name
    else:
        return t.__name__

def function_to_json(func):
    signature = inspect.signature(func)
    type_hints = get_type_hints(func)

    function_info = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {"type": "object", "properties": {}},
        "returns": type_hints.get("return", "void").__name__,
    }
    
    for name, _ in signature.parameters.items():
        param_type = get_type_name(type_hints.get(name, type(None)))
        function_info["parameters"]["properties"][name] = {"type": param_type}

    return json.dumps(function_info, indent = 2)

