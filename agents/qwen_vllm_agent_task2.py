# Cell 1: Imports

import json
from typing import List, Dict, Optional
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import xml.etree.ElementTree as ET


# Cell 2: LoRAInferenceEngine class definition

class LoRAInferenceEngine:
    def __init__(
        self,
        base_model_name: str,
        lora_model_name: Optional[str] = None,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.5,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        enable_lora: bool = True,
        max_loras: int = 1,
        max_lora_rank: int = 64,
    ):
        self.base_model_name = base_model_name
        self.lora_model_name = lora_model_name or base_model_name

        print(f"Initializing vLLM with base model: {base_model_name}")
        if lora_model_name and lora_model_name != base_model_name:
            print(f"LoRA adapter: {lora_model_name}")

        self.llm = LLM(
            model=base_model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enable_lora=enable_lora,
            max_loras=max_loras,
            max_lora_rank=max_lora_rank,
        )

        self.lora_request = None
        if enable_lora and lora_model_name:
            self.lora_request = LoRARequest(
                lora_name="default_lora",
                lora_int_id=1,
                lora_path=lora_model_name,
            )

        print("vLLM engine initialized successfully!")

    def format_prompt(self, instruction: str, input_text: str = "", system_prompt: str = "") -> str:
        if system_prompt:
            if input_text:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### System:
{system_prompt}

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
            else:
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### System:
{system_prompt}

### Instruction:
{instruction}

### Response:
"""
        else:
            if input_text:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""
            else:
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        return prompt

    def generate(
        self,
        prompts: List[str],
        max_tokens: int = 100,
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop=stop or ["### Human:", "### Instruction:", "</s>"],
        )

        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self.lora_request,
        )

        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text.strip()
            responses.append(generated_text)

        return responses

    def single_generate(
        self,
        instruction: str,
        input_text: str = "",
        system_prompt: str = "",
        **kwargs
    ) -> str:
        prompt = self.format_prompt(instruction, input_text, system_prompt)
        responses = self.generate([prompt], **kwargs)
        return responses[0]


class QwenVLLMAgent(object):
    def __init__(self):
        print('INIT QwenVLLMAgent')
        
        # Initialize the LoRA inference engine
        #
        base_model = "augustus2011/goldship_grpo"
        lora_model = "augustus2011/goldship_grpo_lora"
        
        self.engine = LoRAInferenceEngine(
            base_model_name=base_model,
            lora_model_name=lora_model,
            max_model_len=4096,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=1,
            dtype="auto",
            enable_lora=True,
            max_loras=1,
            max_lora_rank=64,
        )
        
        self.memory = []
        self.current_role = ''

    def rag(self, query, document_embeddings, player_convs, npc_convs, npc_funcs, top_k=10):
        # Simplified - return empty results
        return []

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

        print('START turn', len(dialogue) // 2)

        formatted_tools = self.convert_to_qwen_function_format(tool_registry, action_registry)

        # ===================================================================
        # STAGE 1: Function Selection
        # ===================================================================

        print("\n--- STAGE 1: Function Selection ---")
        func_messages = self._create_func_messages(
            tool_registry, action_registry, dialogue,
            worldview, persona, role, knowledge, state
        )

        # Create a simple prompt for function calling
        func_prompt = self._create_function_prompt(func_messages, formatted_tools)
        
        func_text = self.engine.single_generate(
            instruction="Analyze the dialogue and determine what functions to call",
            input_text=func_prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.8,
        )
        print("GENERATED FUNC:", func_text)

        # ===================================================================
        # STAGE 2: PARSE FUNCTION
        # ===================================================================

        print("\n--- STAGE 2: PARSE FUNCTION ---")
        functions_to_call = self.parse_tool_calls_string(func_text, param_key_name='parameters')
        print("Parsed Functions to Call:", functions_to_call)

        # ===================================================================
        # STAGE 3: RETRIEVE
        # ===================================================================

        print("\n--- STAGE 3: RETRIEVE ---")
        function_results = []
        if functions_to_call:
            function_results = executor.execute(functions_to_call)

        self.memory.append(function_results)

        # ===================================================================
        # STAGE 4: INTEGRATE & DRAFT
        # ===================================================================

        print("\n--- STAGE 4: INTEGRATE & DRAFT ---")
        dialogue_messages = self._create_messages_for_dialogue(
            worldview, persona, role, knowledge, state,
            dialogue, function_results
        )

        # Create a simple prompt for response generation
        response_prompt = self._create_response_prompt(dialogue_messages)
        
        draft = self.engine.single_generate(
            instruction="Generate an NPC response based on the dialogue and function results",
            input_text=response_prompt,
            max_tokens=100,
            temperature=0.1,
            top_p=0.8,
        )
        print('DRAFT:', draft)

        return {
            'prompts': dialogue_messages,
            'final_responses': draft
        }

    def _create_function_prompt(self, func_messages, formatted_tools):
        """Create a simple prompt for function calling"""
        prompt = "Available functions:\n"
        for tool in formatted_tools:
            func = tool['function']
            prompt += f"- {func['name']}: {func['description']}\n"
        
        prompt += "\nDialogue:\n"
        for msg in func_messages:
            if msg['role'] == 'user':
                prompt += f"Player: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"NPC: {msg['content']}\n"
        
        prompt += "\nBased on the dialogue, what functions should be called? Return in XML format:\n"
        prompt += "<tool_call>{\"name\": \"function_name\", \"arguments\": {\"param\": \"value\"}}</tool_call>"
        
        return prompt

    def _create_response_prompt(self, dialogue_messages):
        """Create a simple prompt for response generation"""
        prompt = "Context:\n"
        for msg in dialogue_messages:
            if msg['role'] == 'system':
                prompt += f"System: {msg['content']}\n"
            elif msg['role'] == 'user':
                prompt += f"Player: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt += f"NPC: {msg['content']}\n"
        
        prompt += "\nGenerate an appropriate NPC response:"
        return prompt

    def _create_func_messages(self,
        tool_functions, action_functions, dialogue,
        worldview, persona, role, knowledge, state):

        history_list = []
        for item in dialogue:
            speaker_role = "user" if item["speaker"] == "player" else "assistant"
            history_list.append({"role": speaker_role, "content": item["text"]})

        # Simple chat history without complex formatting
        chat_h = '\n'.join([f"{'Player' if h['role']=='user' else 'NPC'}: {h['content']}" for h in history_list[:-1]])

        draft_prompt = (
            "# INSTRUCTION\n"
            "You are an AI assistant playing as an NPC in an RPG game. Your primary task is to analyze what information or action needed to response to the player's message. Then decide what function need to be called and specify function's arguments from provided context.\n"
            "Critical rule for actions: If an action requires the player's explicit confirmation or a choice, Do not call the action function in the same turn you ask for confirmation. The action should only be called in a subsequent turn after the player has confirmed.\n"
            "\n"
            "***CRITICAL RULE FOR FUNCTION ARGUMENTS:***\n"
            "**ONLY use argument values that are EXPLICITLY mentioned in the current player message, chat history, or previous function call results.**\n"
            "**NEVER infer, guess, or use default values for arguments.**\n"
            "**If a required argument's value is NOT found in the provided context, DO NOT include that argument in the function call.**\n"
            "**Do not use 'n/a' or similar placeholders for missing arguments; simply OMIT them.**\n\n"

            "# Chat History\n"
            f"{chat_h}\n"

            "Please carefully analyze the Chat History to select the most appropriate function to perform in this turn and fill its arguments. Remember the strict rule for argument values."
        )

        if len(dialogue[-1]["target_item"]) > 0:
            history_list[-1] = {"role": "user", "content": dialogue[-1]["text"] + f' (Target Item: {dialogue[-1]["target_item"]})'}

        messages = []
        messages.append({"role": "system", "content": draft_prompt})
        messages.append(history_list[-1])

        return messages

    def _create_messages_for_dialogue(self,
        worldview, persona, role, knowledge,
        state, dialogue, function_results):

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

        character_setting = ""
        for k, v in persona.items():
            character_setting += f'- {k}: {v}\n'

        refine_prompt = (
            "# Instruction\n"
            "You are an AI assistant playing as an NPC in an RPG game. You have already gathered necessary information. "
            "Your primary goal is to generate a single, coherent, in-character message to the player that **directly answers their latest query or confirms their action**, using the `Retrieved Information`.\n"
            f"<role>{role}</role>\n"
            f"<persona>{persona}</persona>\n"
            "# General Information\n"
            f"{knowledge_info}\n"
            "# Retrieved Information\n"
            f"{function_knowledge}\n"
        )

        if len(dialogue[-1]["target_item"]) > 0:
            history_list[-1] = {"role": "user", "content": dialogue[-1]["text"] + f' (Target Item: {dialogue[-1]["target_item"]})'}

        messages = []
        messages.append({"role": "system", "content": refine_prompt})
        messages.append(history_list[-1])

        return messages