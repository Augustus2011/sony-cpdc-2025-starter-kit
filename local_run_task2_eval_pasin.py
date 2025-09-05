import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score_fn
from sklearn.metrics import f1_score
import json
import os
import argparse
from tqdm import tqdm
import time
import getpass
from function_calls import tool_map, action_map, Executor
from agents.user_config import UserAgent
import npcdataset.parsers


def load_data(file_path):
    with open(file_path, "r") as fp:
        data = json.load(fp)

    return npcdataset.parsers.parse_conversation_data(data, "test")

def get_responses(agent, cur_conv, cur_turn, tool_registry, action_registry, executor):
    dialogue = [
        {
            "speaker": msg.speaker,
            "text": msg.text,
            "target_item": msg.target_items
        }
        for msg in cur_turn.messages
    ]
    all_results = agent.generate_functions_and_responses(
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
    return all_results['final_responses']

def evaluate_responses(response_path, label_path):
    with open(response_path, 'r') as f:
        responses = json.load(f)
    with open(label_path, 'r') as f:
        labels = json.load(f)

    llm_response = []
    gold_response = []

    for i in responses:
        llm_turn = []
        for key in i:
            if key.startswith('turn_'):
                llm_turn.append(i[key])
        llm_response.append(llm_turn)

    for i in labels:
        gold_turn = []
        for key in i:
            if key.startswith('turn_'):
                gold_turn.append(i[key]['gold_response'])
        gold_response.append(gold_turn)

    max_turns = max(max(len(d) for d in llm_response), max(len(d) for d in gold_response))
    rows = []

    for idx, (llm, gold) in enumerate(zip(llm_response, gold_response)):
        for turn_idx in range(max_turns):
            llm_val = llm[turn_idx] if turn_idx < len(llm) else None
            gold_val = gold[turn_idx] if turn_idx < len(gold) else None
            rows.append({
                "dialogue_id": idx,
                "turn": turn_idx,
                "llm_response": llm_val,
                "gold_response": gold_val
            })

    df = pd.DataFrame(rows)

    nltk.download('punkt')
    smooth = SmoothingFunction().method1

    df['bleu4'] = None
    df['word_f1'] = None
    df['bert_score'] = None

    for idx, row in df.iterrows():
        pred = row['llm_response']
        ref = row['gold_response']

        if isinstance(pred, str) and isinstance(ref, str) and pred.strip() and ref.strip():
            pred_tokens = nltk.word_tokenize(pred.lower())
            ref_tokens = nltk.word_tokenize(ref.lower())

            all_tokens = list(set(pred_tokens + ref_tokens))
            pred_vector = [1 if token in pred_tokens else 0 for token in all_tokens]
            ref_vector = [1 if token in ref_tokens else 0 for token in all_tokens]

            bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
            f1 = f1_score(ref_vector, pred_vector, zero_division=0)
            _, _, F1 = bert_score_fn([pred], [ref], lang='en', verbose=False, model_type='bert-base-uncased')
            bert = F1[0].item()

            df.at[idx, 'bleu4'] = bleu
            df.at[idx, 'word_f1'] = f1
            df.at[idx, 'bert_score'] = bert
        else:
            df.at[idx, 'bleu4'] = None
            df.at[idx, 'word_f1'] = None
            df.at[idx, 'bert_score'] = None

    bleu4_avg = df['bleu4'].mean()
    word_f1_avg = df['word_f1'].mean()
    bert_score_avg = df['bert_score'].mean()

    print(f"Average BLEU-4: {bleu4_avg}")
    print(f"Average Word-level F1: {word_f1_avg}")
    print(f"Average BERTScore: {bert_score_avg}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='./result_task2_responses.json')
    args = parser.parse_args()

    start_time = time.time()
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    data_path = 'data/task2_train.json'
    data_set = load_data(data_path)
    agent = UserAgent()

    # save_directory = os.path.dirname(args.save_path)
    # if not os.path.exists(save_directory):
    #     os.makedirs(save_directory)

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
            response = get_responses(agent, conversation, turn, tool_registry, action_registry, cur_turn_exec)
            cur_conv_responses[f"turn_{turn_idx}"] = response
        generated_responses.append(cur_conv_responses)

    with open("result_task2_train.json", 'w') as f:
        json.dump(generated_responses, f, indent=4)
    print("Responses saved to ", args.save_path)
    print("Total time spent: ", time.time() - start_time, ' seconds')

    evaluate_responses("result_task2_train.json", data_path)
