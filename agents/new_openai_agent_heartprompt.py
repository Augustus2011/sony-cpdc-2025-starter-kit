from typing import List, Dict
import copy
import os
from openai import OpenAI
from function_calls import tool_map, action_map, Executor
import npcdataset.parsers
import json




class NewOpenAIAgent(object):
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor): 

        # Step 1: Convert the function information to the OpenAI function calling format, and prepare prompts for function calling. 
        function_messages, all_functions = self._create_messages_for_function(tool_registry, action_registry, dialogue,role)
        
        # Step 2: Call the OpenAI API to generate the necessary function calls. 
        response = self.client.responses.create(
            model="gpt-4o-mini", # We only allow GPT-4o-mini in the API track. Any other models will lead to failure. 
            input=function_messages,
            tools=all_functions,
        )

        # Step 3: Use the `executor` to obtain the function call results. 
        functions_to_call = []
        for return_item in response.output:
            # We only consider the function calls, and ignore text response. 
            if return_item.type != 'function_call':
                continue 
            name = return_item.name
            args = json.loads(return_item.arguments)
            functions_to_call.append({
                'name': name, 
                'parameters': args
            })
        
        function_results = executor.execute(functions_to_call)
 
        # Step 4: With the function call results and the background, prepare prompts for response generation. 
        messages_resp = self._create_messages_for_dialogue(worldview, persona, role, knowledge, state, dialogue, function_results)
        
        # Step 5: Call the OpenAI API to generate the text response. 
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=messages_resp,
        )

        return {
            'final_responses': response.output_text
        }

    def _prepare_openai_functions(self, tool_registry, action_registry):
        openai_tool_functions = []
        for tool_name, tool_function in tool_registry['function_registry'].items():
            tool_function['type'] = 'function'
            openai_tool_functions.append(tool_function)

        openai_action_functions = []
        for action_name, action_function in action_registry['function_registry'].items():
            action_function['type'] = 'function'
            openai_action_functions.append(action_function)
        openai_functions = openai_tool_functions + openai_action_functions
        return openai_functions 

    def _create_messages_for_function(self, tool_registry, action_registry, dialogue, role):
      all_functions = self._prepare_openai_functions(tool_registry, action_registry)
      additional_info = ""
      if len(dialogue[-1]["target_item"]) > 0:
          additional_info = "Target items:\n"
          for info in dialogue[-1]["target_item"]:
              additional_info += f'{info["name"]}\n'

      function_prompt = (
          "# INSTRUCTION:\n"
          "You are an AI assistant playing as an NPC in an RPG game. "
          "Your primary task is to analyze what information or action is needed to respond to the player's message. "
          "Then decide what function needs to be called and specify the function's arguments from the provided context below.\n\n"
          "# Target item:\n"
          "{additional_info}\n\n"
          "# Tools\n\n"
          "You may call one or more functions to assist with the user query.\n\n"
          "You are provided with function definitions within <tools></tools> XML tags:\n"
          "<tools>\n"
          "{functions}\n"
          "</tools>\n\n"
          "# Role: {role_npc}\n\n"
          "# Chat History:\n"
          "{dialogue}"
      )

      formatted_prompt = function_prompt.format(
          additional_info=additional_info,
          functions="\n".join([str(f) for f in all_functions]),
          role_npc=role,
          dialogue=dialogue
      )

      input_messages = [
          {"role": "system", "content": formatted_prompt},
          {"role": "user", "content": dialogue[-1]['text']}
      ]
      print("Function prompt: ", input_messages)
      return input_messages, all_functions
    
    def _create_messages_for_dialogue(self, worldview, persona, role, knowledge, state, dialogue, function_results):
        dialogue_prompt = (
            "# Instruction\n"
            "You are acting as an NPC character in game.\n"
            "Respond naturally and concisely, based only on the provided knowledge.\n"
            "Avoid exaggerated roleplay or guessing. It's okay to say you’re unsure.\n"
            "Speak like a real person in that world — short, simple, and in character.\n"
            "\n"
            "# Npc Character Profile\n"
            "Play this character without over-acting\n"
            "{}\n"
            "\n"
            "# Knowledge\n"
            "1. Function Call Knowledge (recent and specific)\n"
            "{}\n"
            "2. General Knowledge (background/context)\n"
            "{}\n"
            "\n"
            "# Example Dialogue\n"
            "Player: 'I'm gathering information about the legendary sword. Have you heard any of the tales about it?'\n"
            "Npc:'Oh, absolutely. Every warrior dreams of it. Many have ventured into unknown territories in search of it. I've heard stories of people traveling to all sorts of places, from the continent to the seas.'\n"
            "Player: 'Everyone seems to be interested in legendary weapons. I guess they must be that prestigious, huh?'\n"
            "Npc:'Yeah, that's probably true. But I think it's not so much about the weapon itself having honor, but more about whether the person wielding it has the skill and is worthy of it.\n"
            "\n"
           
            )
        worldview = worldview + '\n' + knowledge['general_info']

        character_setting = ""
        for k, v in persona.items():
            character_setting += f'- {k}: {v}\n'
        
        function_knowledge = ""
        for f_result in function_results:
            parameter_info = []
            return_value_info = []
            for arg in f_result["parameters"]:
                parameter_info.append(f'{arg}: {str(f_result["parameters"][arg])}')
            parameter_info = ", ".join(parameter_info)

            for item in f_result["return"]:
                return_value_info.append(str(item))
            return_value_info = ", ".join(return_value_info)
            function_knowledge += f'{f_result["name"]}: {parameter_info} -> {return_value_info}\n'

        
        general_knowledge = ""
        for item in knowledge["knowledge_info"]:
            item_info = []
            for key in item:
                item_info.append(f'{key}: {item[key]}')
            item_info = ", ".join(item_info)
            general_knowledge += f'{item_info}\n'
        
        history_list = []
        for item in dialogue:
            role = "user"
            if item["speaker"] == "npc":
                role = "assistant"
            history_list.append({"role":role, "content":item["text"]})
        

        prompt = dialogue_prompt.format(character_setting, function_knowledge, general_knowledge, worldview)
        
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.extend(history_list)

        return messages

