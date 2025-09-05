from typing import List, Dict
import copy
import os
from openai import OpenAI
from function_calls import tool_map, action_map, Executor
import npcdataset.parsers
import json




class NewOpenAIAgent(object):
    """
    NewOpenAIAgent is a new implementation for API Track of the Sony CPDC 2025 Challenge. 
    
    This agent takes in information from the dialogue, the functions, and the background of the scenario, 
    e.g. worldview, persona, role, etc., and generates corresponding function calls and text responses. 
    The function information has already been converted from langchain.tools to the OpenAI function calling format 
    using the method `convert_to_openai_function`, to simplify the implementation. 
    For details of the function calls, please refer to the `function_calls` directory. 

    Attributes: 
        client: OpenAI client. 
                You can assume that the API keys are automatically configured in the environment variables. 
    """
    def __init__(self):
        """
        Initialize an openai agent. You can assume that the API keys are automatically configured in the environment variables. 
        """
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

    ############################################################
    # The entrypoint of the evaluator.  
    ############################################################
    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor): 
        """
        Generates the responses given the dialogue, the functions, and the background of the video game scenario.

        This method is the entry point called by the evaluator. It is implemented with the following steps: 
        
        1. Prepare prompts for function calling. This has been much simplified by the `convert_to_openai_function` by langchain. 
        2. Call the OpenAI API to generate the necessary function calls. 
        3. Use the `executor` to obtain the function call results. 
        4. With the function call results and the background, prepare prompts for response generation. 
        5. Call the OpenAI API to generate the text response. 

        Args: 
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format). 
            action_registry: A dict mapping action names to action functions (OpenAI function calling format). 
            Implementations can be found in the directory `function_calls`. 
            
            worldview, persona, role, knowledge, state: They are the background information of the video game scenario. 
            dialogue: List[Dict], the full dialogue history. `dialogue[-1]` refers to the current turn. 
            executor: This is implemented in `function_calls/executor.py`. It takes in a list of function calls and return the results. 

        Returns: 
            A dict with the following keys: 
                'final_responses': str, the text responses. 

            You do not have to return the function calls. The `executor` will record all the function calls that are passed to it. 
        """

        # Step 1: Convert the function information to the OpenAI function calling format, and prepare prompts for function calling. 
        function_messages, all_functions = self._create_messages_for_function(tool_registry, action_registry, dialogue,role)
        
        # Step 2: Call the OpenAI API to generate the necessary function calls. 
        response = self.client.responses.create(
            model="gpt-4o-mini", # We only allow GPT-4o-mini in the API track. Any other models will lead to failure. 
            input=function_messages,
            tools=all_functions,
        )
        #print(function_messages)
        #print("-------------------------------------------------------------------")
        #print(response)
        #print("-------------------------------------------------------------------")
   


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
            model="gpt-4o-mini", # Again, we only allow 'gpt-4o-mini' in the API track. Any other models will lead to failure. 
            input=messages_resp,
        )
        return {
            'final_responses': response.output_text
        }

    ############################################################
    # Helper functions. 
    ############################################################

    def _prepare_openai_functions(self, tool_registry, action_registry):
        """
        Prepare the list of functions as inputs to the OpenAI API. 
        The values of `tool_registry` and `action_registry` have already been converted to the OpenAI function calling format. 

        Args: 
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format). 
            action_registry: A dict mapping action names to action function (OpenAI function calling format). 
            Implementations can be found in the directory `function_calls`. 

        Returns: 
            openai_functions: List[Dict], a list of functions in the OpenAI function calling format. 
        """
        openai_tool_functions = []
        for tool_name, tool_function in tool_registry['function_registry'].items():
            tool_function['type'] = 'function'
            # With my openai=1.77.0 and langchain=0.3.25 version, this should be manually added. 
            # Please note that this may not be necessary for all OpenAI and langchain versions. 
            # Please test it before submission. 
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

      # Use curly-brace placeholders for format
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
      # Format the prompt with the additional information, functions, role, and dialogue
      formatted_prompt = function_prompt.format(
          additional_info=additional_info,
          functions="\n".join([str(f) for f in all_functions]),  # you might need to format functions nicely
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
        """
        Based on the background information of the video game and the dialogue history, 
        creates the messages to feed to OpenAI client to generate the text response. 

        Args: 
            worldview, persona, role, knowledge, state: They are the background information of the video game scenario. 
            dialogue: List[Dict], the full dialogue history. `dialogue[-1]` refers to the current turn. 
            function_results: A list of function call results. 
        """
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

        # 'persona' is a dict that specifies properties of the character. 
        character_setting = ""
        for k, v in persona.items():
            character_setting += f'- {k}: {v}\n'
        
        # function_knowledge records the specific knowledge obtained from the function calls. 
        function_knowledge = ""
        for f_result in function_results:
            # record each function call in the following format: 
            # function_name: parameter_name1, parameter_name2, ... -> return_value1, return_value2, ...
            parameter_info = []
            return_value_info = []
            for arg in f_result["parameters"]:
                parameter_info.append(f'{arg}: {str(f_result["parameters"][arg])}')
            parameter_info = ", ".join(parameter_info)

            for item in f_result["return"]:
                return_value_info.append(str(item))
            return_value_info = ", ".join(return_value_info)
            function_knowledge += f'{f_result["name"]}: {parameter_info} -> {return_value_info}\n'

        
        # general_knowledge records the general knowledge of all items involved in the dialogue. 
        general_knowledge = ""
        for item in knowledge["knowledge_info"]:
            item_info = []
            for key in item:
                item_info.append(f'{key}: {item[key]}')
            item_info = ", ".join(item_info)
            general_knowledge += f'{item_info}\n'
        
        # prepare the dialogue history. 
        history_list = []
        for item in dialogue:
            # The dataset uses 'npc' to indicate the characters in the game. 
            role = "user"
            if item["speaker"] == "npc":
                role = "assistant"
            history_list.append({"role":role, "content":item["text"]})
        

        prompt = dialogue_prompt.format(character_setting, function_knowledge, general_knowledge, worldview)
        
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.extend(history_list)

        return messages

