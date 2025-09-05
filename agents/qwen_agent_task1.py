from vllm import LLM, SamplingParams
print('L VLLM')
from transformers import AutoTokenizer
print('L TRANSFORMER')
import torch
print('L TORCH')
import pickle
print('L PICKLE')
import numpy as np
print('L NUMPY')
from sentence_transformers import SentenceTransformer
print('L SENTENCE TRANSFORMER')

import xml.etree.ElementTree as ET
print('L XML')
import json
print('L JSON')
from pathlib import Path
print('L PATHLIB')

class QwenAgent(object):
    def __init__(self):
        print('INIT')
        model_path = 'Qwen/Qwen3-30B-A3B-FP8'
        self.llm = LLM(
            model=model_path,
            #dtype='bfloat16',
            gpu_memory_utilization=0.95,
        )
        print('VLLM')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.stop_tokens = [self.tokenizer.eos_token]
        print('TOKENIZER')


        self.qwen3emb = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B").to('cuda')
        print('EMBEDDER')

        current_dir = Path(__file__).parent
        # print('current_dir', current_dir)
        self.playerconvs_embeddings = np.load(current_dir / 'playerconvs_embeddings.npy')
        self.npcconvs_embeddings = np.load(current_dir / 'npcconvs_embeddings.npy')
        self.player_convs = np.load(current_dir / 'player_convs.npy')
        self.npc_convs = np.load(current_dir / 'npc_convs.npy')
        print('NPY')

        with open(current_dir / 'npc_funcs.pkl', 'rb') as f:
            self.npc_funcs = pickle.load(f)
        print('PICKLE')
        self.memory = []
        self.current_role = ''

    def rag(self, query, document_embeddings, player_convs, npc_convs, npc_funcs, top_k=10):
        query_embeddings = self.qwen3emb.encode([query], prompt_name="query")
        similarity = self.qwen3emb.similarity(query_embeddings, document_embeddings)
        top_k_values, top_k_indices = torch.topk(similarity, k=top_k)

        matched_convs = []
        for i, score in zip(top_k_indices[0], top_k_values[0]):
            if float(score) > 0.6:
                matched_convs.append((player_convs[i], npc_convs[i], npc_funcs[i]))

        return matched_convs

    def parse_tool_calls_string(self, tool_calls_string, param_key_name='parameters'):
        parsed_tool_calls = []
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
                    for key, value in arguments.items():
                        if value != "n/a":
                            cleaned_parameters[key] = value

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

        if role != self.current_role:
            self.memory = []
            self.current_role = role
            
        formatted_tools = self.convert_to_qwen_function_format(tool_registry, action_registry)
        last_prompt = None
        rag_res = self.rag(
            query=dialogue[-1]['text'],
            top_k=2,
            document_embeddings=self.playerconvs_embeddings,
            player_convs=self.player_convs,
            npc_convs=self.npc_convs,
            npc_funcs=self.npc_funcs
        )

        # ===================================================================
        # STAGE 1: Function Selection
        # ===================================================================

        # print("\n--- STAGE 1: Function Selection ---")
        func_messages = self._create_func_messages(
            tool_registry, action_registry, dialogue,
            worldview, persona, role, knowledge, state, rag_res, self.memory
        )
        last_prompt = func_messages

        func_prompt_str = self.tokenizer.apply_chat_template(
            func_messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False, tools=formatted_tools
        )
        func_sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=5, min_p=0, max_tokens=50, stop=self.stop_tokens)
        outputs = self.llm.generate([func_prompt_str], func_sampling_params)
        func_text = outputs[0].outputs[0].text.strip()
        # print("GENERATED FUNC:", func_text)

        # ===================================================================
        # STAGE 2: PARSE FUNCTION
        # ===================================================================

        # print("\n--- STAGE 2: PARSE FUNCTION ---")
        functions_to_call = self.parse_tool_calls_string(func_text, param_key_name='parameters')
        # print("Parsed Functions to Call:", functions_to_call)

        # ===================================================================
        # STAGE 3: RETRIEVE
        # ===================================================================

        # print("\n--- STAGE 3: RETRIEVE ---")
        function_results = []
        if functions_to_call:
            function_results = executor.execute(functions_to_call)

        self.memory.append(function_results)

        # ===================================================================
        # STAGE 4: INTEGRATE & DRAFT
        # ===================================================================

        # print("\n--- STAGE 4: INTEGRATE & DRAFT ---")
        dialogue_messages = self._create_messages_for_dialogue(
            worldview, persona, role, knowledge, state,
            dialogue, function_results, rag_res, self.memory
        )
        last_prompt = dialogue_messages

        dialogue_prompt_str = self.tokenizer.apply_chat_template(
            dialogue_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        dialogue_sampling_params = SamplingParams(temperature=0.1, top_p=0.8, top_k=3, min_p=0, max_tokens=50, stop=self.stop_tokens)
        outputs = self.llm.generate([dialogue_prompt_str], dialogue_sampling_params)
        draft = outputs[0].outputs[0].text.strip()
        # print('DRAFT:', draft)

        # ===================================================================
        # STAGE 5: REFINE
        # ===================================================================

        # rag_res = self.rag(
        #     query=draft,
        #     top_k=10,
        #     document_embeddings=self.npcconvs_embeddings,
        #     player_convs=self.player_convs,
        #     npc_convs=self.npc_convs,
        #     npc_funcs=self.npc_funcs
        # )

        # # print("\n--- STAGE 5: REFINE ---")
        # dialogue_messages = self._refine(
        #     worldview, persona, role, knowledge, state,
        #     dialogue, function_results, rag_res, self.memory, draft
        # )
        # last_prompt = dialogue_messages

        # dialogue_prompt_str = self.tokenizer.apply_chat_template(
        #     dialogue_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        # )
        # dialogue_sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=256, stop=self.stop_tokens)
        # outputs = self.llm.generate([dialogue_prompt_str], dialogue_sampling_params)
        # final_response = outputs[0].outputs[0].text.strip()
        # print('REFINED:', final_response)


        return {
            'prompts': last_prompt,
            'final_responses': draft
        }
    
    def _create_func_messages(self,
        tool_functions, action_functions, dialogue,
        worldview, persona, role, knowledge, state, rag_res, memory):

        # new_rag_res = []
        # for player_message, npc_message, npc_functions in rag_res:
        #     if len(npc_functions) == 0:
        #         continue
        #     if len(new_rag_res) >= 5:
        #         break
        #     new_npc_functions = [func.split('-return->')[0].strip() for func in npc_functions]
        #     new_rag_res.append((player_message, npc_message, new_npc_functions))
        # rag_res = new_rag_res

        start_i = 0
        function_call_guidance = ''

        if len(rag_res) > 0:
            function_call_guidance = (
                "# Function Call Gold Standard Example\n"
                "Observe when and how functions are called to gather information or perform actions\n"
                "Based on a highly similar past conversation, here is the ideal way to call functions for the current player message. Follow this format precisely.\n\n"
            )

            for i, (player_message, npc_message, npc_functions) in enumerate(rag_res[start_i:5]):
                example = (f"- Example {i+1}\n"
                        f"**Example Player Message:** {player_message}\n"
                        f"**Correct Function Calls:** {npc_functions}\n"
                        f"**Resulting NPC Response:** {npc_message}\n")
                function_call_guidance += example
            function_call_guidance += '\n'
        else:
            function_call_guidance = ''


        history_list = []
        for item in dialogue:
            speaker_role = "user" if item["speaker"] == "player" else "assistant"
            history_list.append({"role": speaker_role, "content": item["text"]})

        chat_h = [f"{'Player' if h['role']=='user' else 'NPC'}: {h['content']}" for h in history_list[:-1]]
        tmp = []
        cnt = 0
        for i in range(0, len(chat_h), 2):
            tmp.append(chat_h[i])
            tmp.append(f'Function Call: {memory[cnt]}')
            tmp.append(chat_h[i+1])
            cnt += 1
        chat_h = '\n'.join(tmp)

        draft_prompt = (
            "# INSTRUCTION\n"
            "You are an AI assistant playing as an NPC in an RPG game. Your primary task is to analyze what information or action needed to response to the player's message. Then decide what function need to be called and specify function's arguments from provided context and memory.\n"
            "Critical rule for actions: If an action requires the player's explicit confirmation or a choice, Do not call the action function in the same turn you ask for confirmation. The action should only be called in a subsequent turn after the player has confirmed.\n"
            "\n"
            "***CRITICAL RULE FOR FUNCTION ARGUMENTS:***\n"
            "**ONLY use argument values that are EXPLICITLY mentioned in the current player message, chat history, or previous function call results.**\n"
            "**NEVER infer, guess, or use default values for arguments.**\n"
            "**If a required argument's value is NOT found in the provided context, DO NOT include that argument in the function call.**\n"
            "**Do not use 'n/a' or similar placeholders for missing arguments; simply OMIT them.**\n\n"

            f"{function_call_guidance}\n"

            "# Chat History (and Function Call History)\n"
            f"{chat_h}\n"

            "Please carefully analyze the Chat History and any Function Call History to select the most appropriate function to perform in this turn and fill its arguments. Remember the strict rule for argument values."
        )

        # print('FUNCTION CALL GUIDANCE:', function_call_guidance)
        # print('MEMORY:', chat_h)

        if len(dialogue[-1]["target_item"]) > 0:
            history_list[-1] = {"role": "user", "content": dialogue[-1]["text"] + f' (Target Item: {dialogue[-1]["target_item"]})'}

        messages = []
        messages.append({"role": "system", "content": draft_prompt})
        messages.append(history_list[-1])

        return messages
    
    def _create_messages_for_dialogue(self,
        worldview, persona, role, knowledge,
        state, dialogue, function_results, rag_res, memory):

        # Format the function results for the prompt
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
                    return_value_info = ", ".join([str(item) for item in return_value])
                else:
                    return_value_info = str(return_value)

                knowledge_lines.append(f"- Result for `{placeholder}` is {return_value_info}")
            function_knowledge = '\n'.join(knowledge_lines)

        # pre-process rag result
        new_rag_res = []
        for player_message, npc_message, npc_functions in rag_res:
            new_npc_functions = [func.split('-return->')[0].strip() for func in npc_functions]
            new_rag_res.append((player_message, npc_message, new_npc_functions))
        rag_res = new_rag_res

        start_i = 0
        style_guidance = "" #
        if len(rag_res) > 0:
            style_guidance = (
                "# GOLDEN Response Style Guidance\n"
            )
            style_guidance += '\n'.join([f'- Example {i+1}:\n Player: {player_message}\n NPC: {npc_message}\n' for i, (player_message, npc_message, npc_functions) in enumerate(rag_res[start_i:start_i+1])])
            style_guidance += '\n'
        else:
            style_guidance = ""

        state_info = ""
        if state:
            for k, v in state.items():
                state_info += f'- {k}: {v}\n'

        knowledge_info = ""
        if "knowledge_info" in knowledge:
            for item in knowledge["knowledge_info"]:
                item_info = ", ".join([f'{key}: {item[key]}' for key in item])
                knowledge_info += f'- {item_info}\n'

        history_list = []
        for item in dialogue:
            speaker_role = "user" if item["speaker"] == "player" else "assistant"
            history_list.append({"role": speaker_role, "content": item["text"]})

        chat_h = '\n'.join([f"{'Player' if h['role']=='user' else 'NPC'}: {h['content']}" for h in history_list[:-1]])

        character_setting = ""
        for k, v in persona.items():
            character_setting += f'- {k}: {v}\n'

        refine_prompt = (
            "# Instruction\n"
            "You are an AI assistant playing as an NPC in an RPG game. You have already gathered necessary information. "
            "Your primary goal is to generate a single, coherent, in-character message to the player that **directly answers their latest query or confirms their action**, using the `Retrieved Information`.\n"
            "**Prioritize factual accuracy and directness** based on `Retrieved Information`. Avoid unnecessary conversational filler, asking redundant questions, or deflecting if the answer is available.\n"
            "\n"

            "***CRITICAL RULE: STRICT INFORMATION UTILIZATION & DIRECTNESS***\n"
            "1.  **Direct Answer First:** If the `Retrieved Information` provides a direct answer to the player's implicit or explicit query (e.g., item details, lists of types, quest specifics), **your response MUST primarily consist of that direct answer.** Present it clearly, concisely, and in character. Do NOT ask further questions if the answer is already provided.\n"
            "2.  **Exact Phrase/List Replication:** If `Retrieved Information` contains specific phrases, item names, prices, or a list of options (like weapon types), strive to **incorporate those exact terms or enumerate the list directly** in your response, matching the gold standard's style.\n"
            "3.  **Confirm Actions/Outcomes:** If the `Retrieved Information` indicates an action was performed (e.g., item sold, equipped, quest started/accepted), **explicitly state the outcome of that action** to the player. (e.g., 'Your weapon has been equipped.'). Ensure this confirmation aligns with the actual success/failure indicated by `Retrieved Information`.\n"
            "4.  **Handle Missing Data Gracefully:** If `Retrieved Information` indicates no relevant data was found (e.g., 'n/a', empty search results, item not found, 'unavailable'), **clearly and politely communicate this lack of information** to the player. Then, and *only then*, offer alternatives or ask relevant clarifying questions.\n"
            "\n"

            "# Your Character Setting\n"
            f"{character_setting}\n"

            # "# Your Role\n"
            # f"{role}\n"

            # "# Worldview\n"
            # f"{worldview}\n"

            # "# State Information\n"
            # f"{state_info}\n"

            "# General Information\n"
            f"{knowledge_info}\n"

            # "# Retrieved Information (High Priority, Your knowledge for this turn)\n"
            # f"{function_knowledge}\n"

            f"{style_guidance}\n"

            # "# Chat History (High Priority)\n"
            # f"{chat_h}"
        )

        if len(dialogue[-1]["target_item"]) > 0:
            history_list[-1] = {"role": "user", "content": dialogue[-1]["text"] + f' (Target Item: {dialogue[-1]["target_item"]})'}

        messages = []
        messages.append({"role": "system", "content": refine_prompt})
        messages.append(history_list[-1])

        return messages
    
    def _refine(self,
        worldview, persona, role, knowledge,
        state, dialogue, function_results, rag_res, memory, draft):

        # pre-process rag result
        new_rag_res = []
        for player_message, npc_message, npc_functions in rag_res:
            new_npc_functions = [func.split('-return->')[0].strip() for func in npc_functions]
            new_rag_res.append((player_message, npc_message, new_npc_functions))
        rag_res = new_rag_res

        start_i = 0
        style_guidance = ""
        if len(rag_res) > 0:
            style_guidance = (
                "# GOLDEN Response Guidance (best similarity matching from RAG using `draft` as query)\n"
            )
            style_guidance += '\n'.join([f'- Example {i+1}:\n Golden Response: {npc_message}\n' for i, (player_message, npc_message, npc_functions) in enumerate(rag_res[start_i:start_i+1])])
            style_guidance += '\n'
        else:
            style_guidance = ""

        state_info = ""
        if state:
            for k, v in state.items():
                state_info += f'- {k}: {v}\n'

        knowledge_info = ""
        if "knowledge_info" in knowledge:
            for item in knowledge["knowledge_info"]:
                item_info = ", ".join([f'{key}: {item[key]}' for key in item])
                knowledge_info += f'- {item_info}\n'

        history_list = []
        for item in dialogue:
            speaker_role = "user" if item["speaker"] == "player" else "assistant"
            history_list.append({"role": speaker_role, "content": item["text"]})

        if len(dialogue[-1]["target_item"]) > 0:
            history_list[-1] = {"role": "user", "content": dialogue[-1]["text"] + f' (Target Item: {dialogue[-1]["target_item"]})'}

        chat_h = '\n'.join([f"{'Player' if h['role']=='user' else 'NPC'}: {h['content']}" for h in history_list])

        character_setting = ""
        for k, v in persona.items():
            character_setting += f'- {k}: {v}\n'

        # g_temp = [f"```\n{npc_message}\n```" for _, npc_message, _ in rag_res[start_i:5]]
        # if len(g_temp) == 0:
        #     target_length = 150
        #     g_temp = 'no golden template found.'
        # else:
        #     target_length = int(sum([len(s) for s in g_temp]) / len(g_temp))
        #     g_temp = '\n'.join(g_temp)
        g_temp = f"```\n{rag_res[0][1]}\n```"
        target_length = len(rag_res[0][1])

        refine_prompt = (
            "# TASK: REWRITE RESPONSE\n"
            "You are an expert style editor for an RPG. Your task is to rewrite a DRAFT response to perfectly match the style, tone, and length of a GOLDEN TEMPLATE.\n\n"

            "-----------------------------------\n"
            "# GOLDEN TEMPLATE (Your Style Guide)\n"
            f"{g_temp}"
            f"**Target Length:** Approximately {target_length} characters.\n"
            "-----------------------------------\n\n"

            "-----------------------------------\n"
            "# DRAFT (The Content to Rewrite)\n"
            f"```\n{draft}\n```\n"
            "-----------------------------------\n\n"

            "# CRITICAL INSTRUCTIONS\n"
            "1.  **Preserve Key Information:** The final text MUST contain all essential information from the DRAFT (item names, prices, quest details, character names, quantities, etc.).\n"
            "2.  **Adopt the Style:** Rewrite the DRAFT using the sentence structure, vocabulary, and tone of the GOLDEN TEMPLATE.\n"
            "3.  **Match the Length:** The final response's length MUST be very close to the Target Length. Be concise. Do not add any extra words, filler, or explanations.\n"
            "4.  **Final Output Only:** Your response should ONLY be the final, rewritten text. Nothing else.\n\n"

            "# Additional Information\n"
            "## Your Character Setting\n"
            f"{character_setting}\n"

            "## Your Role\n"
            f"{role}\n"

            "## Worldview\n"
            f"{worldview}\n"

            "## State Information\n"
            f"{state_info}\n"

            "## General Information\n"
            f"{knowledge_info}\n"

            "## Chat History\n"
            f"{chat_h}"

            "# REFINED RESPONSE:"
        )

        messages = [
            {"role": "user", "content": refine_prompt}
        ]

        # print(f'[REFINE] DRAFT:\n{draft}')
        # print(f'[REFINE] GOLDEN TEMPLATE:\n{g_temp}')
        # print(f'[REFINE] TARGET LENGTH: {target_length}')

        return messages