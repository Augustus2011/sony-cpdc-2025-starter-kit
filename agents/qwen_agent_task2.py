from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import AutoTokenizer

import xml.etree.ElementTree as ET
import json
from huggingface_hub import snapshot_download

class QwenAgent(object):
    def __init__(self):
        model_path = 'Qwen/Qwen3-8B'

        self.lora_name1 = 'atsui_umasume'
        self.lora_path1 = snapshot_download(repo_id='augustus2011/atsui_umasume')
        self.lora_name2 = 'samui_umasume'
        self.lora_path2 = snapshot_download(repo_id='augustus2011/samui_umasume')

        self.llm = LLM(
            model=model_path,
            dtype='bfloat16',
            gpu_memory_utilization=0.8,
            enable_lora=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_path)
        self.stop_tokens = [self.tokenizer.eos_token]


    def parse_tool_calls_string(self, tool_calls_string, param_key_name='parameters'):
        parsed_tool_calls = []
        # In case the LLM generates multiple tool_call blocks without a wrapping root element
        wrapped_string = f"<root>{tool_calls_string}</root>"

        try:
            root = ET.fromstring(wrapped_string)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return []

        for tool_call_element in root.findall('tool_call'):
            try:
                json_str = tool_call_element.text.strip()
                tool_data = json.loads(json_str)

                name = tool_data.get('name')
                arguments = tool_data.get('arguments', {})

                if name:
                    cleaned_parameters = {}
                    # Iterate through arguments and ensure all keys are strings
                    for key, value in arguments.items():
                        
                        corrected_key = key
                        # Handle the specific case where a key is a list like `["item_name"]`
                        if isinstance(key, list):
                            if len(key) == 1 and isinstance(key[0], str):
                                corrected_key = key[0]
                                print(f"Warning: Corrected list-based key '{key}' to string '{corrected_key}'.")
                            else:
                                # As a fallback, convert the unexpected list key to a string
                                corrected_key = str(key)
                                print(f"Warning: Converted unusual list-based key '{key}' to string.")

                        # Handle other non-string keys (e.g., integers) by converting them
                        elif not isinstance(key, str):
                            corrected_key = str(key)
                            print(f"Warning: Converted non-string key '{key}' to string '{corrected_key}'.")
                        
                        # Now, use the corrected_key to add the parameter
                        if value != "n/a":
                            cleaned_parameters[corrected_key] = value

                    parsed_tool_calls.append({
                        'name': name,
                        param_key_name: cleaned_parameters
                    })
                else:
                    print(f"Warning: Found a tool_call without a 'name': {json_str}")
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON within <tool_call>: {e} - Content: {tool_call_element.text}")
            except AttributeError as e:
                print(f"Error accessing text within tool_call element: {e}")

        return parsed_tool_calls

    def convert_to_qwen_function_format(self, tool_registry, action_registry):
        formatted_tools = []

        # Process functions from tool_registry
        for func_name, func_details in tool_registry['function_registry'].items():
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": func_details['name'],
                    "description": func_details['description'],
                    "parameters": func_details['parameters']
                }
            })

        # Process functions from action_registry
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

    def generate_functions_and_responses(
        self, tool_registry, action_registry,
        worldview, persona, role, knowledge,
        state, dialogue, executor):

        formatted_tools = self.convert_to_qwen_function_format(tool_registry, action_registry)

        last_prompt = None

        # ===================================================================
        # STAGE 1: Function Selection
        # ===================================================================

        func_messages = self._create_func_messages(
            tool_registry, action_registry, dialogue,
            worldview, persona, role, knowledge, state,
            formatted_tools
        )
        last_prompt = func_messages

        func_prompt_str = self.tokenizer.apply_chat_template(
            func_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )

        func_sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=128, stop=self.stop_tokens)
        outputs = self.llm.generate([func_prompt_str], func_sampling_params, 
                                    lora_request=LoRARequest(self.lora_name1, 1, self.lora_path1),
                                    use_tqdm=False)
        func_text = outputs[0].outputs[0].text.strip()

        # ===================================================================
        # STAGE 2: PARSE FUNCTION
        # ===================================================================

        functions_to_call = self.parse_tool_calls_string(func_text, param_key_name='parameters')

        # ===================================================================
        # STAGE 3: RETRIEVE
        # ===================================================================

        function_results = []
        if functions_to_call:
            try:
                function_results = executor.execute(functions_to_call)
            except:
                print('[ERROR] functions_to_call:', functions_to_call)

        # ===================================================================
        # STAGE 4: GENERATE DIALOGUE
        # ===================================================================

        dialogue_messages = self._create_messages_for_dialogue(
            worldview, persona, role, knowledge, state,
            dialogue, function_results
        )
        last_prompt = dialogue_messages

        dialogue_prompt_str = self.tokenizer.apply_chat_template(
            dialogue_messages, tokenize=False, add_generation_prompt=True, 
            enable_thinking=False
        )
        dialogue_sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=100, stop=self.stop_tokens)
        outputs = self.llm.generate([dialogue_prompt_str], dialogue_sampling_params,
                                    lora_request=LoRARequest(self.lora_name2, 1, self.lora_path2),
                                    use_tqdm=False)
        draft = outputs[0].outputs[0].text.strip()

        # ===================================================================
        # FINISHED !!!
        # ===================================================================

        return {
            'prompts': last_prompt,
            'final_responses': draft
        }

    def _create_func_messages(self,
        tool_functions, action_functions, dialogue,
        worldview, persona, role, knowledge, state, formatted_tools):

        tools = '\n'.join([str(func) for func in formatted_tools])

        history = []
        for j, item in enumerate(dialogue):
            speaker_role = "user" if item["speaker"] == "player" else "assistant"
            if j < len(dialogue) - 1:
                history.append(f'{speaker_role}: {item["text"]}')
            else:
              history.append(f'{speaker_role} (**current turn**): {item["text"]} (**target item: {item["target_item"]}**)')

        history = '\n'.join(history)

        user = (
            "# INSTRUCTION:\n"
            "You are an AI assistant playing as an NPC in an RPG game. "
            "Your primary task is to analyze what information or action needed to response to the player's message. "
            "Then decide what function need to be called and specify function's arguments from provided context below.\n\n"

            "***CRITICAL RULE FOR FUNCTION ARGUMENTS:***\n"
            "**ONLY use argument values that are EXPLICITLY mentioned in the current player message, chat history, or previous function call results.**\n"
            "**NEVER infer, guess values for arguments.**\n\n"
            
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"

            "<tools>\n"
            f"{tools}\n"
            "</tools>\n\n"

            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": "function-name", "arguments": <args-json-object>}\n'
            "</tool_call>\n"

            f"# Role: {role}\n"

            "# Chat History:\n"
            f'{history}\n'
        )

        messages = []
        messages.append({"role": "user", "content": user})

        return messages

    def _create_messages_for_dialogue(self,
        worldview, persona, role, knowledge,
        state, dialogue, function_results):

        history = []
        for j, item in enumerate(dialogue):
            speaker_role = "user" if item["speaker"] == "player" else "assistant"
            if j < len(dialogue) - 1:
                history.append(f'{speaker_role}: {item["text"]}')
            else:
              history.append(f'{speaker_role} (**current turn**): {item["text"]} (**target item: {item["target_item"]}**)')

        history = '\n'.join(history)

        character_setting = ''
        for k, v in persona.items():
            character_setting += f'- {k}: {v}\n'

        state_info = ''
        if state:
            for k, v in state.items():
                state_info += f'- {k}: {v}\n'

        # knowledge_info_str = ''
        # for item in knowledge['knowledge_info']:
        #     item_info = ', '.join([f'{key}: {item[key]}' for key in item])
        #     knowledge_info_str += f'- {item_info}\n'
        knowledge_info = knowledge['knowledge_info']

        general_info = knowledge['general_info']

        function_knowledge = "No new information was gathered."
        if len(function_results) > 0:
            knowledge_lines = []
            for f_result in function_results:
                # Reconstruct the original placeholder for clarity
                params_list = [f'{k}={v}' for k, v in f_result.get("parameters", {}).items()]
                placeholder = f"{f_result['name']}({', '.join(params_list)})"

                return_value = f_result.get("return", "No specific value returned.")

                # Handle list returns cleanly
                if isinstance(return_value, list):
                    return_value_info = ", ".join([json.dumps(item) for item in return_value])
                else:
                    return_value_info = str(return_value)

                knowledge_lines.append(f"- Result for `{placeholder}` is {return_value_info}")
            function_knowledge = '\n'.join(knowledge_lines)

        user = (
            "# INSTRUCTION\n"
            "Imagine you are an NPC character in RPG game.\n"
            "Respond naturally and concisely, based only on the provided knowledge.\n"
            "Avoid exaggerated roleplay or guessing. It's okay to say you’re unsure.\n"
            "Speak like a real person in that world — short, simple, and in character.\n\n"

            "# Character Profile\n"
            f"Role: {role}\n"
            f"{character_setting}\n"

            "# Worldview\n"
            f"{worldview}\n"

            "# State Informations\n"
            f"{state_info}\n"

            "# Knowledges\n"
            "1. Function Call Knowledges (recent and specific)\n"
            f"{function_knowledge}\n"
            "2. General Knowledges (basic knowledge about item/quest/entity)\n"
            f"{knowledge_info}\n"
            "3. General Informations (background/context)\n"
            f"{general_info}\n"

            "# Dialogue History\n"
            f"{history}\n"

            "# RESPONSE\n"
        )

        messages = []
        messages.append({"role": "user", "content": user})

        return messages
