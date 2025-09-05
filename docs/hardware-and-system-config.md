## Hardware and System Configuration

### GPU Track 
Each submission will be run on an AWS g6e.2xlarge node. The node is equipped with 8 vCPUs, 64 GM RAM, and one NVIDIA L40s GPU with 48 GB GPU memory. 

The timeout for each turn is set at 7s. 

### API Track
Each submission will be run on an AWS m5.large node. This node is equipped with 2 vCPUs and 8GB RAM. 

In addition, we enforce the following constraints on API usage. 

- A maximum of 2 API calls per turn. 

- The input length is limited to 2000 tokens per turn. The output length is limited to 200 tokens per turn. 

- We only allow `gpt-4o-mini` models. All other models will lead to immediate failure. 

- We do not allow fine-tuned model APIs. 

- Network access is blocked (except for the API call). 

- The timeout for each turn is set at 7s. 