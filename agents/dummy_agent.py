from typing import Any, List, Dict


# function_prompt = (
#           "# INSTRUCTION:\n"
#           "You are an AI assistant playing as an NPC in an RPG game. "
#           "Your primary task is to analyze what information or action is needed to respond to the player's message. "
#           "Then decide what function needs to be called and specify the function's arguments from the provided context below."
#           "Try use more function even in normal chating dialogue in guild receptionist or weapon shop.\n\n"
#           "# Target item:\n"
#           "{additional_info}\n\n"
#           "# Tools\n\n"
#           "The final answer, you may call one or more functions to assist with the user query.\n\n"
#           "The final answer, you are provided with function definitions within <tools></tools> XML tags:\n"
#           "<tools>{functions}</tools>\n\n"
#           "# Role: {role_npc}\n\n"
#           "# Chat History:\n"
#           "{dialogue}"
#       )
class DummyResponseAgent(object):
    """
    A dummy agent implementation for the Sony CPDC challenge. 
    It returns the same template answer for all questions, and does not make any function calls. 
    """

    def __init__(self):
        """ Load your model(s) or API clients here """
        pass
    
    
    def generate_functions_and_responses(self, tool_registry, action_registry, worldview, persona, role, knowledge, state, dialogue, executor):
        """
        Given the background information, perform adequate function calls, and based on the function call results, generate coherent and reasonable responses. 

        Parameters
        ----------
            tool_registry, action_registry: Dict[str, Dict[str, str]] are function registries for tool and action functions, 
                from which we can index functions with function names. 
                For example, you can index a tool function named 'tool1' via `tool_registry['function_registry']['tool1'][k]`, 
                where k can be 'args', 'description', or 'name'.  
            worldview: str, the worldview of the current dialogue. 
            persona: Dict[str, str], describes the persona of the NPC, e.g. persona['name'], persona['age'], persona['gender'], persona['occupation']. 
                See the sample and training datasets for details. 
            role: str, the role of the NPC. 
            knowledge: Dict[str, Any], contains basic knowledge about the items (e.g. quests, weapons). See the sample and training datasets for details. 
            state: Dict[str, str], the time, location, etc. of the current conversation. 
            dialogue: List[Dict[str, str]]. It records the previous turns of the dialogue. 
                Each dict in the list is of the following format: 
                {
                    "speaker": ..., 
                    "text": ..., 
                    "target_item": ...
                }
            executor: It is a module that can execute function calls you need and record the history of all function calls you make. 


        Returns
        ----------
            Dict[str, str] with the following structure. 
                {
                    "prompts": Optional. The prompt of the current turn. 
                    "final_responses": Your response of the current turn. 
                }
        
        NOTE: You do not need to return the generated function calls. The `executor` will automatically record that. 
        """
        # It does not call any functions, so nothing will be recorded in the executor. 
        # Only return responses. 
        response = {
            "prompts": "",
            "final_responses": "THIS IS A TEST REPLY"
        }
        return response
    
    
    