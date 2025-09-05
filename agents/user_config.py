# try:
#     from agents.qwen_agent_task2 import QwenAgent
# except ImportError:
#     try:
#         from .qwen_agent_task2 import QwenAgent
#     except ImportError:
#         from qwen_agent_task2 import QwenAgent

#from .new_openai_agent_heartprompt import NewOpenAIAgent
from .new_openai_agent_heartprompt import NewOpenAIAgent
#from .new_openai_agent import NewOpenAIAgent
#from .qwen_vllm_agent_task2 import QwenVLLMAgent

#from .new_openai_agent_task_2 import NewOpenAIAgent
UserAgent = NewOpenAIAgent #QwenVLLMAgent#NewOpenAIAgent