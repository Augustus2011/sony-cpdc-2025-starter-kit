import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
import npcdataset.parsers
from function_calls import tool_map, action_map

load_dotenv(".env")


with open("/Users/fs-mk-mb-022/gpu/superai_ss5/cpdc2025/heart-sony-cpdc-2025-starter-kit-submission-c/data/task1_train.json", "r") as fp:
    data = json.load(fp)


sample = data[0]
knowledge = sample['knowledge']
function_list_id = sample['function_list_id']


tool_registry = tool_map[function_list_id]
action_registry = action_map[function_list_id]

def convert_to_qwen_function_format(tool_registry, action_registry):
    formatted_tools = []

    for func_name, func_details in tool_registry['function_registry'].items():
        formatted_tools.append({
            "type": "function",
            "function": {
                "name": func_details['name'],
                "description": func_details['description'],
                "parameters": func_details['parameters']
            }
        })

    for func_name, func_details in action_registry['function_registry'].items():
        formatted_tools.append({
            "type": "function",
            "function": {
                "name": func_details['name'],
                "description": func_details['description'],
                "parameters": func_details['parameters']
            }
        })

    return formatted_tools

formatted_tools = convert_to_qwen_function_format(tool_registry, action_registry)


prompt = f"""You are tasked with generating high-quality game dialogue between a player and an NPC who has a merchant role. You are provided with:

1. A list of available function calls that the NPC can use to respond.
2. Structured knowledge relevant to the NPC’s inventory, abilities, or item lore.

Your responsibilities are:

- Generate a natural and contextually appropriate player dialogue that clearly expresses the player’s intent or question.
- Select a function call from the provided list that appropriately addresses the player’s request.
- Fill in the function’s parameters using only the provided knowledge base. Do not invent new values.

---

#Provided Function(s)
{json.dumps(formatted_tools, indent=2)}

#Knowledge
{json.dumps(knowledge, indent=2)}

---

#Desired Output Format

```json
{{
  "player_dialogue": "<string>",
  "gold_functions": [
    {{
      "name": "<string>",
      "parameters": {{
        "<parameter_name>": "<parameter value>"
      }}
    }}
  ]
}}
#Example Output
{{
  "player_dialogue": "The price is reasonable. Though before deciding, could you tell me more about how other magic users integrate this dagger into their combat style?",
  "gold_functions": [
    {{
      "name": "check_description",
      "parameters": {{
        "item_name": "Man Gauche"
      }}
    }}
  ]
}}
"""


SAFETY_SETTINGS = [
{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
]


def gemini2_5_pro(input_prompt: str = "") -> str:
    client = genai.Client(api_key=os.getenv("GEMINI_API"))


    response = client.models.generate_content(
                model="gemini-2.5-pro-preview-05-06",
                contents=input_prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=20000,
                    #temperature=0.6,
                    top_p=0.95,
                    top_k=2,
                    safety_settings=SAFETY_SETTINGS 

                )
            )
     

 
    if response.candidates and response.candidates[0].finish_reason == 'STOP':
        return response.text
    else:
        return f"[NO OUTPUT] Finish reason: {response.candidates[0].finish_reason}"

result = gemini2_5_pro(prompt)
print(result)