# Starter Kit for [Sony Commonsense Persona-Grounded Dialogue Challenge 2025](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025)

![banner image](https://images.aicrowd.com/raw_images/challenges/banner_file/1156/776d3b119fa4ee5d2c43.jpg)

This repository is the [Sony Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025) **Submission template and Starter kit**!   
Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your agents to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your model, etc.
*  **Starter code** for you to get started!

**Note**: The starter kit and all related interfaces are subject to change in future rounds of the challenge, e.g. from Round 1 to Round 2, so please check back frequently for the latest starter kit. 

# Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Dataset](#-dataset)
3. [Tasks](#-tasks)
4. [Evaluation Metrics](#-evaluation-metrics)
5. [Getting Started](#-getting-started)
   - [How to write your own agent?](#️-how-to-write-your-own-agent)
   - [How to start participating?](#-how-to-start-participating)
      - [Setup](#setup)
      - [How to make a submission?](#-how-to-make-a-submission)
      - [What hardware does my code run on?](#-what-hardware-does-my-code-run-on-)
6. [Frequently Asked Questions](#-frequently-asked-questions)
6. [Important Links](#-important-links)

# Competition Overview

You’re playing your favourite video game, navigating a bustling medieval city on your quest. When you meet a blacksmith, he greets you and mentions last night’s storm that damaged his roof. You ask about a new weapon, and he recalls your last visit, suggests an upgrade, and even offers a discount because you helped him in a previous quest.

NPCs that are context-aware respond naturally and adapt to the world around them to enable dynamic in-game interactions.

But most NPCs today have repetitive, disconnected, and robotic dialogue, struggling to balance small talk with task-driven exchanges—the very elements that make games exciting and immersive.

🎮 Enter the Commonsense Persona-grounded Dialogue Challenge (CPDC 2025)! 🎮

How can we make NPCs feel real? This challenge pushes the boundaries of AI-driven dialogue—creating characters that think, remember, and interact naturally for richer, more immersive game worlds.

This year, the challenge consists of three tasks:

- **Task 1**: Task-Oriented Dialogue Response Generation
- **Task 2**: Commonsense Dialogue Response Generation
- **Task 3**: A hybrid of Task 1 and Task 2, evaluating whether both objectives can be achieved simultaneously with a single agent

# Dataset
We provide two dataset splits: 
- `data/task*_sample.json`: They are minimal data splits mainly for debugging. 
- `data/task*_train.json`: They serve as training data for the challenge. 

Each `.json` file contains several multi-turn conversations between a player and an NPC in a game environment. Each conversation has its unique worldviews, settings, player and NPC persona, etc. 

`npcdataset/` provides an interface for participants to parse the raw data. We will use the `npcdataset` class to parse, load, and feed the data during our online evaluation, so do not tamper with this class, otherwise you risk data loading failures. 

Under the challenge rules, participants can feel free to use any training data to build their solutions. 

# Tasks
The Sony CPDC challenge will be split into three tasks. 
- **Task 1: Task-Oriented Dialogue Response Generation**: The data for task 1 will include persona and worldview information as common information, along with available function definitions and role-specific knowledge. Participants will use this information to call functions when necessary and may use the results of these function calls to generate responses.
- **Task 2: Commonsense Dialogue Response Generation**: The data for task 2 will include persona and worldview information as common information, along with available function definitions and role-specific knowledge. Based on this information, participants will generate natural and character-appropriate responses.
- **Task 3: A hybrid of Task 1 and Task 2**, evaluating whether both objectives can be achieved simultaneously with a single agent. Submitting to Task 3 will automatically result in evaluation under both Task 1 and Task 2. Therefore, participants should prepare an agent that meets the requirements of both tasks.

Sony CPDC 2025 features two tracks, GPU track and API track. In the GPU track, we provide participants with a GPU server to locally run LLMs to generate answers. In the API track, we provide participants with access to OpenAI APIs. Participants are required to build their solutions with fixed state-of-the-art LLMs. 

Please find more information about how to submit to a specific track/task in [docs/submission.md#submitting-to-different-tasks](docs/submission.md#submitting-to-different-tasks).

# Evaluation Metrics 
Systems for task 1 will be evaluated on both function calling and response generation. Systems for task 2 will only be evaluated on response generation. 

To avoid overfitting the metrics, we will not disclose the exact metrics used to evaluate the systems. Also, the leaderboard will only show relative scores instead of absolute scores. 

Please refer to [local_run_task1.py](local_run_task1.py) and [local_run_task2.py](local_run_task2.py) for details on how we will run your system to get responses. 

# 🏁 Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025).
2. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/sony-commonsense-persona-grounded-dialogue-challenge-2025/starter-kit-sony/-/forks/new) to create a fork.
3. **Clone** your forked repo and start developing your agent.
4. **Develop** your agent(s) following the template in [how to write your own agent](#how-to-write-your-own-agent) section.
5. [**Submit**](#-how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#-how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the private datasets and report the metrics on the leaderboard of the competition.

# ✍️ How to write your own agent?

In `agents/vanilla_llama_agent.py` we implement a simple baseline that directly calls [LLaMA-3.1-8B-Instruct model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) to generate the function calls and the responses. You can run it as follows: 

```
pip install -r requirements.txt
python3 local_run_task1.py
python3 local_run_task2.py

# By default, the output will be saved in 'results/task*_responses.json'. 
```

If you want to try your own agent, follow these steps: 
- Put everything you need (including model weights---you won't have access to the Internet during evaluation) in `agents/`. 
- Implement a class `MyAgent` in `agents/my_agent.py` that has the method `generate_functions_and_responses()`, which will be called for both Task 1 and Task 2. 
- In `agents/user_config.py`, set `UserAgent = MyAgent(...)`. 

Please follow the instructions in [agents/README.md](agents/README.md) for instructions and examples on how to write your own agents for this competition.

# 🚴 How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/-/user_settings/ssh_keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/user/ssh.html).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/sony-commonsense-persona-grounded-dialogue-challenge-2025/sony-cpdc-2025-starter-kit/-/forks/new) to create a fork.

3.  **Clone the repository**

    ```bash
    git clone git@gitlab.aicrowd.com:<YOUR-AICROWD-USER-NAME>/sony-cpdc-2025-starter-kit.git
    cd sony-cpdc-2025-starter-kit
    ```

4. **Install** competition specific dependencies!
    ```bash
    pip install -r requirements.txt
    ```

5. Write your own agent as described in [How to write your own agent](#how-to-write-your-own-agent) section.

6. Test your agent locally using `python local_run_task*.py`.

To test for any software runtime errors, you can also build the docker image and run everything locally by: 
```
./docker_run.sh --task1
./docker_run.sh --task2
./docker_run.sh --task3
```

7. Accept the Challenge Rules on the main [challenge page](https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025) by clicking on the **Participate** button. Also accept the Challenge Rules on the Task specific page (link on the challenge page) that you want to submit to.

8. Make a submission as described in [How to make a submission](#-how-to-make-a-submission) section.

## 📮 How to make a submission?

Please follow the instructions in [docs/submission.md](docs/submission.md) to make your first submission. 
This also includes instructions on [specifying your software runtime](docs/submission.md#specifying-software-runtime-and-dependencies), [code structure](docs/submission.md#code-structure-guidelines), [submitting to different tasks](docs/submission.md#submitting-to-different-tasks).

**Note**: **Remember to accept the Challenge Rules** on the challenge page, **and** the task page before making your first submission.


## 💻 What hardware does my code run on ?
You can find more details about the hardware and system configuration in [docs/hardware-and-system-config.md](docs/hardware-and-system-config.md).


# ❓ Frequently Asked Questions 
## Which task is this starter kit for ?
This starter kit can be used to submit to any of the tasks. You can find more information in [docs/submission.md#submitting-to-different-tasks](docs/submission.md#submitting-to-different-tasks).

**Best of Luck** :tada: :tada:

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025
- 🗣 Discussion Forum: https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/commonsense-persona-grounded-dialogue-challenge-2025/leaderboards


