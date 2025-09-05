# Guide to Making Your First Submission

This document is designed to assist you in making your initial submission smoothly. Below, you'll find step-by-step instructions on specifying your software runtime and dependencies, structuring your code, and finally, submitting your project. Follow these guidelines to ensure a smooth submission process.

# Table of Contents

1. [Specifying Software Runtime and Dependencies](#specifying-software-runtime-and-dependencies)
2. [Code Structure Guidelines](#code-structure-guidelines)
3. [Submitting to Different Tasks](#submitting-to-different-tasks)
4. [Submission Entry Point](#submission-entry-point)
5. [Setting Up SSH Keys](#setting-up-ssh-keys)
6. [How to Submit Your Code](#how-to-submit-your-code)


## Specifying Software Runtime and Dependencies

Our platform supports custom runtime environments. This means you have the flexibility to choose any libraries or frameworks necessary for your project. Here’s how you can specify your runtime and dependencies:

- **`requirements.txt`**: List any PyPI packages your project needs. **Do specify versions, as we observe significant difference in inference time between different `transformer` versions.**
- **`apt.txt`**: Include any apt packages required.
- **`Dockerfile`**: The one located at the root will be used by default to build your submission. **You can specify the python version here if you need specific ones**. 

For detailed setup instructions regarding runtime dependencies, refer to the documentation in the `docs/runtime.md` file.

## Code Structure Guidelines

Your project should follow the structure outlined in the starter kit. Here’s a brief overview of what each component represents:

```
.
├── .dockerignore                   # Please specify the paths to your model checkpoints so that the large files won't be built into the docker image. 
├── README.md                       # Project documentation and setup instructions
├── aicrowd.json                    # Submission meta information - like your username, task name
├── agents
│   ├── README.md                   # Documentation specific to the implementation of model interfaces
│   ├── dummy_agent.py              # A simple or placeholder model for demonstration or testing.
│   ├── test_agent.py               # We also implement a simple LLaMA3.1-8B-instruct baseline here. 
│   └── user_config.py              # IMPORTANT: Configuration file to specify your model 
├── data
│   ├── task*_sample.json           # Minimal dataset for debugging. 
│   ├── task*_train.json            # Training datasets (approximately 40 conversations). 
├── docs
│   └── runtime.md                  # Documentation on the runtime environment setup, dependency configs
├── Dockerfile                      # The Dockerfile that will be used to build your submission and all dependencies. The default one will work fine, but you can write your own. 
├── docker_run.sh                   # This script builds your submission locally and calls `local_evaluation.py`. It can be used to debug (if your submission fails to build). 
├── function_calls/           # The interface for the agent to call tool and action functions (for task 1). 
│                                   # IMPORTANT: This directory will be overwritten during evaluation. DO NOT TAMPER WITH IT. 
│   ├──`executor.py`                # This is the interface to implement the function calls made by the agent. For the starter kit, 
│                                   # the `Executor` will only return valid values if the agent's function calls match the `gold_functions`. 
│                                   # Otherwise, it will not return anything. However, during the actual evaluation, 
│                                   # `Executor` will return adequate values as long as it is a valid function call. 
├── local_run_task1.py              # Use this to check your model runs locally
├── local_run_task2.py              # Use this to check your model runs locally
├── npcdataset/                     # A class to read and parse the dataset. You can feel free to implement your own. 
├── requirements.txt                # Python packages to be installed for model development
└── utilities
    └── _Dockerfile                 # Example Dockerfile for specifying runtime via Docker
```

Remember, 
- **your submission metadata JSON (`aicrowd.json`)** is crucial for mapping your submission to the challenge. Ensure it contains the correct `challenge_id`, `authors`, and other necessary information. 
- To submit to the GPU track, set the `"gpu": true` flag in your `aicrowd.json`. Otherwise, if you set `"gpu": false`, the submission will automatically go to the API track. 
- The entire `function_calls` directory will be overwritten during actual evaluation. All local modifications will be lost. **DO NOT TAMPER WITH.**

## Submitting to Different Tasks

Specify the task by setting the appropriate `challenge_id` in your [aicrowd.json](aicrowd.json). Here are the challenge IDs for the three tasks in this challenge:

| Task          |      `challenge_id`|
|---------------|-----------------|
| Task 1: Task-Oriented Dialogue | `"task-oriented-dialogue-task-1"`| 
| Task 2: Context-Aware Dialogue | `"context-aware-dialogue-task-2"`|
| Task 3: Integrating Contextual Dialogue and Task Execution | `"integrating-contextual-dialogue-and-task-execution-hybrid"`|

In addition, if you set `"gpu": true`, you automatically submit to the GPU track. If you set `"gpu": false`, the submission automatically goes to the API track. 

## Using models on HuggingFace

If you want to use a model available HuggingFace, please include itsa reference to its model spec in `aicrowd.json` as: 
```
    "hf_models": [
      {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "revision": "main"
      },
      {
        "repo_id": "meta-llama/Llama-3.1-8B",
        "revision": "main",
        "ignore_patterns": "*.md",
      }
        ...
    ]
```

The evaluators will ensure that before the evaluation begins (in a container without network access), these models are available in the local huggingface cache of the evaluation container.

The keys for the `model_spec` dictionary can include any parameter supported by the [`huggingface_hub.snapshot_download`](https://huggingface.co/docs/huggingface_hub/v0.30.2/en/package_reference/file_download#huggingface_hub.snapshot_download) function.

**Important:**
- Models specified must be publicly available, or the [aicrowd Hugging Face account](https://huggingface.co/aicrowd) must be explicitly granted access.
- If your model repository is private, you must grant access to the [`aicrowd` user](https://huggingface.co/aicrowd). Otherwise, your submission will fail.

**Granting access to private repositories:**
To provide access to a private repository, create an organization on Hugging Face specifically for your participation in this competition. Create your private repository within this organization and add the `aicrowd` user as a member to ensure seamless access.


## Submission Entry Point

The evaluation process will instantiate an agent from `agent/user_config.py` for evaluation. Ensure this configuration is set correctly.

## Setting Up SSH Keys

You will have to add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/-/user_settings/ssh_keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/ssh/README.html#generating-a-new-ssh-key-pair).


## How to Submit Your Code

To submit your code, push a tag beginning with "submission-" to your repository on [GitLab](https://gitlab.aicrowd.com/). Follow these steps to make a submission:

Assuming, you have cloned the repo already by following the instructions [here](../README.md#setup) and made your changes.

1. Commit your changes with `git commit -am "Your commit message"`.
2. Tag your submission (e.g., `git tag -am "submission-v0.1" submission-v0.1`).
3. Push your changes and tags to the AIcrowd repository (e.g. `git push origin submission-v0.1`)

After pushing your tag, you can view your submission details at `https://gitlab.aicrowd.com/<YOUR-AICROWD-USER-NAME>/<YOUR-REPO>/issues`. It may take about **30 minutes** for each submission to build and begin evaluation, so please be patient. 

Ensure your `aicrowd.json` is correctly filled with the necessary metadata, and you've replaced `<YOUR-AICROWD-USER-NAME>` with your GitLab username in the provided URL.
