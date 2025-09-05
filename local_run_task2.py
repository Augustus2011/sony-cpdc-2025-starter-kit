from typing import List, Dict, Tuple
import json
import numpy as np
import npcdataset.parsers
from agents.user_config import UserAgent
import argparse 
from tqdm import tqdm
import os
import time
from function_calls import tool_map, action_map, Executor


def load_data(file_path):
    with open(file_path, "r") as fp:
        data = json.load(fp)

    return npcdataset.parsers.parse_conversation_data(data, "test")

def get_responses(agent, cur_conv, cur_turn, tool_registry, action_registry, executor) -> List[Dict[str, str]]: 
    """
        Parameters: 
            agent: UserAgent, the agent that the participants use to generate responses. 
            cur_conv: current conversation. Contains information about worldview, persona, roles, etc. 
            cur_turn: current turn, contains the conversation history up to this turn. 
            tool_registry, agent_registry: functions available for task 2. 
            executor: It can perform adequate function calls and return proper return values. 
        Returns: str, the response of the current turn. 
    """
    dialogue = [
        {
            "speaker": msg.speaker,
            "text": msg.text,
            "target_item": msg.target_items
        }
        for msg in cur_turn.messages
    ]
    all_results,token_function_call,token_response = agent.generate_functions_and_responses(
        tool_registry, 
        action_registry, 
        cur_conv.worldview,
        cur_conv.personas['npc'].to_dict(), 
        cur_conv.roles['npc'], 
        {"general_info": cur_conv.general_knowledge, "knowledge_info": cur_conv.knowledge},
        cur_conv.state, 
        dialogue, 
        executor
    )
    return all_results['final_responses'],token_function_call,token_response

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='results/task2_responses.json')
    args = parser.parse_args()

    start_time = time.time()

    # data_path = 'data/task2_train.json'
    data_path = 'data/task2_sample.json'
    data_set = load_data(data_path)
    agent = UserAgent()

    save_directory = os.path.dirname(args.save_path)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)


    generated_responses = []
    for conv_idx, conversation in tqdm(enumerate(data_set)):
        cur_conv_responses = {}
        tool_registry = tool_map[conversation.function_list_id]
        action_registry = action_map[conversation.function_list_id]
        for turn_idx, turn in enumerate(conversation.turns):
            gold_functions = [
                {
                    "name": function.name,
                    "parameters": function.parameters,
                    'return': function.return_values
                }
                for function in turn.gold_functions
            ]
            cur_turn_exec = Executor(tool_registry, action_registry, gold_functions)
            response,token_func,token_res = get_responses(agent, conversation, turn, tool_registry, action_registry, cur_turn_exec)
            cur_conv_responses[f"turn_{turn_idx}"] = response
            cur_conv_responses[f"token_func_{turn_idx}"]=token_func
            cur_conv_responses[f"token_res_{turn_idx}"]=token_res
            
            
        generated_responses.append(cur_conv_responses)

    with open(args.save_path, 'w') as f:
        json.dump(generated_responses, f, indent=4)
    print("Responses saved to ", args.save_path)
    print("Total time spent: ", time.time() - start_time, ' seconds')
