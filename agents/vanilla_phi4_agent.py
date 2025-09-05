from typing import List, Dict
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch
import copy

class VanillaPhi4Agent(object):

    def __init__(self):
        """Load necessary models and configurations here"""
        #max_seq_length = 2028  # Maximum sequence length
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Phi-4-mini-instruct",
            #max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
        )

        # Enable native 2x faster inference
        FastLanguageModel.for_inference(self.model)

        # Set up chat template
        
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template = "phi-4",
        )

        self.max_seq_len = 2048
        self.max_gen_len = 50

    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor):
        """
        Generates the responses given the dialogue, the functions, and the background of the video game scenario.

        Args: 
            tool_registry: A dict mapping tool names to tool functions (OpenAI function calling format). 
            action_registry: A dict mapping action names to action functions (OpenAI function calling format). 
            worldview, persona, role, knowledge, state: They are the background information of the video game scenario. 
            dialogue: List[Dict], the full dialogue history. `dialogue[-1]` refers to the current turn. 
            executor: This is implemented in `function_calls/executor.py`. It takes in a list of function calls and return the results. 

        Returns: 
            A dict with the following keys: 
                'final_responses': str, the text responses. 
        """
        
        # Step 1: Generate function calls
        function_messages = self._create_messages_for_function(tool_registry, action_registry, dialogue)
        inputs = self.tokenizer.apply_chat_template(
            function_messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids = inputs,
            max_new_tokens = self.max_gen_len,
            use_cache = True,
            temperature = 0.1,
            min_p = 0.1
        )
        
        res = outputs[0][inputs.shape[-1]:]
        items = self.tokenizer.decode(res, skip_special_tokens=True).split("\n")

        # Step 2: Parse function calls
        final_functions = []
        res_item = {}
        for item in items:
            if "function name: " in item:
                if "name" in res_item:
                    final_functions.append(copy.deepcopy(res_item))
                    res_item = {}
                res_item["name"] = item.replace("function name: ", "").split(",")[0]
                res_item["parameters"] = {}
            elif "argument name: " in item:
                arg_name = ""
                arg_val = ""
                if "value: " in item:
                    arg_name = item.split("value: ")[0].replace("argument name: ", "").split(",")[0]
                    arg_val = item.split("value: ")[1]

                if arg_name != "":
                    if not "parameters" in res_item:
                        res_item["parameters"] = {}
                    res_item["parameters"][arg_name] = arg_val
                    
        if "name" in res_item:
            final_functions.append(res_item)

        # Step 3: Execute functions
        function_results = executor.execute(final_functions)

        # Step 4: Generate response
        dialogue_messages = self._create_messages_for_dialogue(worldview, persona, role, knowledge, state, dialogue, function_results)
        inputs = self.tokenizer.apply_chat_template(
            dialogue_messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            input_ids = inputs,
            max_new_tokens = self.max_gen_len,
            use_cache = True,
            temperature = 0.1,
            min_p = 0.1
        )
        
        res = outputs[0][inputs.shape[-1]:]
        res_str = self.tokenizer.decode(res, skip_special_tokens=True).replace("\n", " ")

        return {
            'prompts': dialogue_messages,
            'final_responses': res_str
        }

    def _create_messages_for_function(self, tool_functions, action_functions, dialogue):
        """
        Creates the messages to feed to the model to generate the necessary function calls.
        """
        function_prompt = (
            "# Instruction\n"
            "You are an assistant in estimating function names and arguments given some dialogues in a video game world.\n"
            "You will need the following information to respond to the user's input. \n"
            "Use the following steps to estimate the necessary function names and arguments. \n"
            "\n"
            "1. Read the dialogue and the target item. \n"
            "2. From the given function information, select the functions that can obtain the information you need. \n"
            "3. Fill in the arguments needed by the function as appropriate. \n"
            "The format of the output is:\n"
            "function name: xxx\n"
            "argument name: xxx, value: xxx\n"
            "\n"
            "Note: You may select multiple functions or no functions at all. \n"
            "\n"
            "# Function Information\n"
            "{}\n"
            "# Additional Information\n"
            "{}\n"
        )

        # Prepare function information
        function_information = []
        for tool_name in tool_functions['function_registry'].keys():
            tool_ = tool_functions['function_registry'][tool_name]
            tool_prompt = (
                "# Function Name: {}\n"
                "# Function Docstring: {}\n"
            ).format(tool_['name'], tool_['description'])
            function_information.append(tool_prompt)
        for action_name in action_functions['function_registry'].keys():
            action_ = action_functions['function_registry'][action_name]
            action_prompt = (
                "# Function Name: {}\n"
                "# Function Docstring: {}\n"
            ).format(action_['name'], action_['description'])
            function_information.append(action_prompt)
        function_information_agg = '\n'.join(function_information)

        # Handle target items
        additional_info = ""
        if len(dialogue[-1]["target_item"]) > 0:
            additional_info = "In the dialogue, the user may be referring to the following items: \n"
            for info in dialogue[-1]["target_item"]:
                additional_info += f'parameter name: name, value: {info["name"]}\n'

        input_text = dialogue[-1]["text"]
        prompt = function_prompt.format(function_information_agg, additional_info)
        messages = []
        messages.append({"role":"system", "content":prompt})
        messages.append({"role":"user", "content":input_text})

        return messages

    def _create_messages_for_dialogue(self, worldview, persona, role, knowledge, state, dialogue, function_results):
        """
        Creates messages for dialogue response generation.
        """
        dialogue_prompt = (
            "# Instruction\n"
            "You are acting as an NPC character in game.\n"
            "Respond naturally and concisely, based only on the provided knowledge.\n"
            "Avoid exaggerated roleplay or guessing.\n"
            "Speak like a real person in that world — short, simple, and in character.\n"
            "\n"
            "# Character Profile\n"
            "{}\n"
            "\n"
            "# Knowledge\n"
            "1. Function Call Knowledge (recent and specific)\n"
            "{}\n"
            "2. General Knowledge (background/context)\n"
            "{}\n"
            "\n"
            "# Response Style Guide\n"
            "- Limit to 1–2 short, natural sentences.\n"
            "- Use simple, in-character language.\n"
            "- Only use information in the knowledge.\n"
            "- If unsure, it's okay to express doubt.\n"
        )
        worldview = worldview + '\n' + knowledge['general_info']

        # Format character settings
        character_setting = ""
        for k, v in persona.items():
            character_setting += f'- {k}: {v}\n'
        
        # Format function knowledge
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

        # Format general knowledge
        general_knowledge = ""
        for item in knowledge["knowledge_info"]:
            item_info = []
            for key in item:
                item_info.append(f'{key}: {item[key]}')
            item_info = ", ".join(item_info)
            general_knowledge += f'{item_info}\n'
        
        # Format dialogue history
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


