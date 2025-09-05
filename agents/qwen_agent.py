from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from transformers import AutoTokenizer

import xml.etree.ElementTree as ET
import re
import json
from huggingface_hub import snapshot_download

class QwenAgent(object):
    def __init__(self):
        model_path = 'Qwen/Qwen3-14B'

        self.lora_name1 = 'prince_lora'
        self.lora_path1 = snapshot_download(repo_id='Phuree/prince_lora')
        # self.lora_name2 = 'alois_lora'
        # self.lora_path2 = snapshot_download(repo_id='Phuree/alois_lora')

        self.llm = LLM(
            model=model_path,
            dtype='bfloat16',
            gpu_memory_utilization=0.8,
            enable_lora=True,
            max_model_len=4096,
            disable_sliding_window=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.lora_path1)
        self.stop_tokens = [self.tokenizer.eos_token]

        # self.current_role = ''
        # self.memory = set()


    def _fix_malformed_json(self, json_string):
        """
        Attempts to fix common malformed JSON issues, specifically:
        1. Unquoted keys (e.g., name: -> "name":)
        2. Unquoted string values immediately following a colon
           (e.g., : start -> : "start")
        """
        # Step 1: Add quotes to unquoted keys
        # This regex looks for word characters followed by a colon and not preceded by a quote
        # It also ensures it's at the start of an object or after a comma
        fixed_string = re.sub(r'([,{]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_string)

        # Step 2: Add quotes to unquoted string values immediately following a colon
        # This targets words that are values, but not numbers or boolean literals (true, false, null)
        # It also handles nested objects/arrays to avoid quoting them
        fixed_string = re.sub(
            r':\s*([a-zA-Z_][a-zA-Z0-9_]*)(?=[,}])',  # Matches words followed by } or ,
            r': "\1"',
            fixed_string
        )
        fixed_string = re.sub(
            r':\s*([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*[,}\]])', # More general, includes arrays ]
            r': "\1"',
            fixed_string
        )
        # Handles cases like `value}` or `value,`
        fixed_string = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)$', r': "\1"', fixed_string)


        # Add specific handling for the 'start' value in your example if it's not captured
        # This is a more targeted fix if the above regexes are still too broad/narrow
        fixed_string = fixed_string.replace('arguments: {quest_name:', 'arguments: {"quest_name":')
        fixed_string = fixed_string.replace('name: start,', 'name: "start",') # Specific for 'start' value

        return fixed_string

    def parse_tool_calls_string(self, tool_calls_string, param_key_name='parameters'):
        parsed_tool_calls = []
        wrapped_string = f"<root>{tool_calls_string}</root>"

        try:
            root = ET.fromstring(wrapped_string)
        except ET.ParseError as e:
            print(f"Error parsing XML: {e}")
            return []

        for tool_call_element in root.findall('tool_call'):
            original_json_str = tool_call_element.text.strip()
            tool_data = {}
            try:
                tool_data = json.loads(original_json_str)
            except json.JSONDecodeError as e:
                print(f"Initial JSON decoding failed: {e}. Attempting to fix malformed JSON.")
                fixed_json_str = self._fix_malformed_json(original_json_str)
                try:
                    tool_data = json.loads(fixed_json_str)
                    print("Successfully parsed after fixing malformed JSON.")
                except json.JSONDecodeError as e_fixed:
                    print(f"Failed to parse even after fixing: {e_fixed} - Content: {original_json_str}")
                    continue
            except AttributeError as e:
                print(f"Error accessing text within tool_call element: {e}")
                continue

            name = tool_data.get('name')
            arguments = tool_data.get('arguments', {})

            if name:
                cleaned_parameters = {}
                for key, value in arguments.items():
                    corrected_key = key
                    if isinstance(key, list):
                        if len(key) == 1 and isinstance(key[0], str):
                            corrected_key = key[0]
                            print(f"Warning: Corrected list-based key '{key}' to string '{corrected_key}'.")
                        else:
                            corrected_key = str(key)
                            print(f"Warning: Converted unusual list-based key '{key}' to string.")
                    elif not isinstance(key, str):
                        corrected_key = str(key)
                        print(f"Warning: Converted non-string key '{key}' to string '{corrected_key}'.")

                    if str(value) not in ['', '[]', 'n/a']:
                        cleaned_parameters[corrected_key] = value

                parsed_tool_calls.append({
                    'name': name.strip(),
                    param_key_name: cleaned_parameters
                })
            else:
                print(f"Warning: Found a tool_call without a 'name': {original_json_str}")

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

        # if role != self.current_role:
        #     self.current_role = role
        #     self.memory = set()

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

        func_sampling_params = SamplingParams(temperature=0.1, top_p=0.5, top_k=10, min_p=0.0, max_tokens=128, stop=self.stop_tokens)
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

        # tmp_functions_to_call = []
        # for func in functions_to_call:
        #     func_tuple = (func['name'], tuple(sorted(func['parameters'].items())))

        #     if func_tuple in self.memory:
        #         continue
            
        #     self.memory.add(func_tuple)
        #     tmp_functions_to_call.append(func)
        
        # functions_to_call = tmp_functions_to_call

        function_results = []
        if functions_to_call:
            function_results = executor.execute(functions_to_call)

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
        dialogue_sampling_params = SamplingParams(temperature=1.0, top_p=0.95, top_k=30, min_p=0.0, max_tokens=100, stop=self.stop_tokens)
        outputs = self.llm.generate([dialogue_prompt_str], dialogue_sampling_params,
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

        for func in formatted_tools:
            name = func['function']['name'].strip()
            if name in ['search_item', 'search_quest']:
                if 'required' in func['function']['parameters']:
                    func['function']['parameters']['required'] = []

        tools = '\n'.join([str(func) for func in formatted_tools])

        tools = tools.replace('item_names', 'item_name')

        knowledge_info_str = ''
        for item in knowledge['knowledge_info']:
            if 'name' in item and 'type' in item:
                item_info = f'{item["name"]} is a {item["type"]}'
            elif 'name' in item:
                item_info = f'{item["name"]}'
                knowledge_info_str += f'- {item_info}\n'

        history = []
        for j, item in enumerate(dialogue):
            speaker_role = "player" if item["speaker"] == "player" else "npc"
            if j < len(dialogue) - 1:
                history.append(f'{speaker_role}: {item["text"]}')
            else:
                target_item = ''
                if len(item["target_item"]) > 0:
                    items = []
                    for e in item["target_item"]:
                        items.append(e["name"])
                    target_item = ' (Target Items: ' + ', '.join(items) + ')'
                history.append(f'{speaker_role}: {item["text"]}{target_item}')

        history = '\n'.join(history)

        user = (
            "# INSTRUCTION:\n"
            "You are an AI assistant playing as an NPC in an RPG game. "
            "Your primary task is to analyze what information or action needed to response to the player's message. "
            "Then decide what function need to be called and specify function's arguments from provided context below.\n\n"

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

            f"# Item Knowledge: \n{knowledge_info_str}\n"

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
            "# Instruction\n"
            "Imagine you are an NPC character in RPG game.\n"
            "Respond naturally and concisely, based only on the provided knowledge.\n"
            "Avoid exaggerated roleplay or guessing. It's okay to say you’re unsure.\n"
            "Speak like a real person in that world — short, simple, and in character.\n"
            "\n"
            "# Character Profile\n"
            f"Role: {role}\n"
            f"{character_setting}\n"
            "\n"
            "# Knowledge\n"
            "***CRITICAL RULE FOR RESPONSE GENERATION:***\n"
            "**If Function Call Knowledge is provided, please naturally INTEGRATE it in the response**\n"
            "1. Function Call Knowledge (recent and specific)\n"
            f"{function_knowledge}\n"
            "2. General Knowledge (background/context)\n"
            f"{knowledge_info}\n"
            "\n"
            "1# Example Dialogue if you are Weapon seller\n"
            "player: I am looking to buy a weapon for an upcoming battle. What would you recommend?\n"
            "npc: Ah, well that will depend on what type of weapon you need. We carry single and double-handed swords, axes, spears, bows, whips, and a selection of blunt weapons.\n"
            "player: I am interested in purchasing a bow. What about this one here?\n"
            "npc: This one? It is the Hunter's Bow. While it may deal less damage than some of our other weapons, it makes up for it with quick reloading speed allowing you to fire in quick succession. This one here is 100 gold.\n"
            "player: I see, can you show me the one next to it? What are its strength and features? And how much is it?\n"
            "npc: That is the Long Bow, an excellent choice for a strong young man like yourself. While it cannot be reloaded as quickly, it is excellent for long range attacks and can deal more damage than the standard Hunter's Bow. This one costs 300 gold.\n"
            "player: One more thing, could you explain the bow on the bottom right? It looks rather powerful.\n"
            "npc: You are correct! That is the Avis Wind and it has a reputation for speed and strength. It is said that once this bow is used, the target cannot escape. It fires rapidly with powerful strikes, but this reputation comes with a hefty price tag of 7500 gold.\n"
            "player: Money isn't a concern for me at the moment, so the Avis Wind seems like a good choice. I would like to purchase the Avis Wind.\n"
            "npc: Thank you for your patronage! Would you like to equip your item now?\n"
            "player: Yes, I want to equip it immediately.\n"
            "npc: Your weapon has been equipped. Safe travels adventurer, come back anytime!\n"
            "\n"
            "2# Example Dialogue if you are Guild reception\n"
            "player: I'm a member of this guild, but I haven't had a chance to take on a quest yet. Could you explain how it all works?\n"
            "npc: It's great to have you here. So, the way it works is simple. We have a variety of quests here. You can choose one that fits your skill level. Once you complete it, you report back here, and we reward you accordingly.\n"
            "player: Sounds straightforward enough. So are there different levels of quests here?\n"
            "npc: Yes, there are different levels of quests. They range from A, which are the easiest, to C for more experienced adventurers. The highest difficulty is S, for the most challenging task.\n"
            "player: That sounds really interesting. How do other adventurers usually approach these quests? Do they mainly focus on earning rewards or is there more to it?\n"
            "npc: Well, it varies. Some adventurers like to take on quests daily to steadily earn rewards and build up their funds. Others might focus on improving their skills or testing their limits. It's all about what they're aiming for, really!\n"
            "player: I think I'm starting to get the hang of it now. By the way, if you don't mind me asking, how did you end up joining the guild? What's your background?\n"
            "npc: Oh, I've always been fascinated by the world of monsters and adventurers. And I really wanted to do something that could help the guild. Being a receptionist allows me to assist in many ways, and I feel like I'm contributing to the team.\n"
            "player: That's really impressive! I can tell you're passionate about helping the guild. I also want to contribute in my own way. How long have you been working as a receptionist here?\n"
            "npc: I've been here for about a year now. It's been a rewarding experience, and I've learned so much along the way. I enjoy the daily interactions and getting to know the guild members better. How long have you been adventuring?\n"
            "player: I've been exploring for over 15 years and have fought in countless battles against monsters. I've developed a deep understanding of the world and its many dangers. I'm always looking for new and exciting adventures.\n"
            "npc: Wow, 15 years of experience! You must have learned so much. I also enjoy reading about monster biology and behavior. I hope to use that knowledge to contribute to the guild's research and help with the fight.\n"
            "player: That's interesting. Your knowledge of monster biology and behavior will definitely be useful for this job. It's something that can really help with understanding quests and guiding adventurers.\n"
            "npc: I'm really happy to be able to contribute. If you ever have any questions about quests, feel free to ask me anytime!\n"

            "\n# Dialogue History\n"
            f"{history}\n"
        )


        messages = []
        messages.append({"role": "user", "content": user})

        return messages
