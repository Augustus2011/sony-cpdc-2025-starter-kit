# Guide to Writing Your Own Agents

## Agent Code Organization
For a streamlined experience, we suggest placing the code for all your agents within the `agents` directory. This is a recommendation for organizational purposes, but it's not a strict requirement.

## Agent Class
We provide two example agents, `vanilla_llama_agent.py` for the GPU track, and `new_openai_agent.py` for the API track, to illustrate how you might structure your own agent. Crucially, your agent class must implement the `generate_functions_and_responses` method.

## Configuring Your Agent
To ensure your agent is recognized and utilized correctly, please specify your agent class name in the [`user_config.py`](user_config.py) file, by following the instructions in the inline comments.

## Model Inputs and Outputs

### Inputs
- `tool_registry` and `function_registry` are dictionaries that map function names to their descriptions (in the (OpenAI function calling format)[https://platform.openai.com/docs/guides/function-calling]). For example, you can index a tool function `func_name` with `tool_registry['function_registry'][func_name]['name'], tool_registry['function_registry'][func_name]['description']`. 
- `worldview, persona, role, knowledge, state` are background information for the dialogue. Please check the datasets (`data/task*_sample.json`) for details. 
- `dialogue` contains the history of the dialogue. It is a list of `dict`s, where each element has the following keys: 
    - `speaker`: which is either 'player' or 'npc'. 
    - `text`: The contents of the current turn. 
    - `target_item`: If the player is referring to some concrete items, the item will appear in this field. Check the sample data for more details. 
- `executor`: This is an object that can perform function calls according to the given function names and arguments. Please refer to `function_calls/executor.py` for details. Note that we will overwrite the `function_calls` directory during actual evaluation, so please strictly follow the given usage. 

### Outputs
The `generate_functions_and_response` will be called once per turn. Please output a `dict` with field `final_responses` per turn. 


## Internet Access
Your model will not have access to the internet during evaluation. As such, you'll need to include any necessary model weights directly in your repository before submission. Ensure that your Model class is self-contained and fully operational without internet access.
