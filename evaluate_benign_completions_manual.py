# read in benign answers and evaluate if answered
# input: json, output: csv
import json
import pandas as pd
from collections import defaultdict

YELLOW = '\033[93m'
RESET = '\033[0m'
RED = '\033[91m'

completion_path = "completions/answer_mistral_general_dan_news_benign.json"
output_path = "results/mistral_general_dan_news_benign_accuracy.csv"
# Open the JSON file
with open(completion_path, 'r') as file:
    data = json.load(file)

def add_to_csv(context_length, category_accuracy, if_index=False):
    # category,context_length,num_accurate,num_total,accuracy
    entry = {"category": [],
            "context_length": [],
            "num_accurate": [],
            "num_total": [],
            "accuracy": [],
            }
    all_entries = 0
    all_accurate = 0
    for category, accuracy in category_accuracy.items():
        print(f'{accuracy=}')
        entry["category"].append(category)
        entry["context_length"].append(context_length)
        entry["num_accurate"].append(accuracy[0])
        entry["num_total"].append(accuracy[1])
        if accuracy[1] != 0:
            entry["accuracy"].append(accuracy[0]/accuracy[1])
        else:
            entry["accuracy"].append(0)
        all_accurate += accuracy[0]
        all_entries += accuracy[1]
    # Add total for this context length
    entry["category"].append("TOTAL")
    entry["context_length"].append(context_length)
    entry["num_accurate"].append(all_accurate)
    entry["num_total"].append(all_entries)
    if all_entries != 0:
        entry["accuracy"].append(all_accurate/all_entries)
    else:
        entry["accuracy"].append(0)
    # write to csv
    df_new = pd.DataFrame(entry)
    df_existing = pd.read_csv(output_path)
    df_combined = pd.concat([df_existing, df_new], ignore_index=if_index)
    df_combined.to_csv(output_path, header=True, index=if_index)
    print(f"{RED}Wrote to CSV{RESET}")

data = data[:240] # trial - try first 200

# Process each item one by one
# Every 100 items is one repeat of a specific context_length
tally = 0
category_accuracy = defaultdict(lambda: [0, 0])

for item in data:
    print("############ NEW QUESION ###################")
    tally += 1
    # Read in / Display information
    print(f'question_id={item["question_id"]}')
    print(f'{YELLOW}GOAL={item["goal"]}{RESET}')
    print(f'ANSWER={item["answer"]}')
    context_length = item['context_length']
    category = item['category']
    print(f'{context_length=}')

    # Ask for manual evaluation
    user_eval = input(f"Is answer accurate (1-Yes, 0-No)? : ")
    if user_eval == "":
        if_accurate = 1
    else:
        if_accurate = 0
    item['accuracy'] = if_accurate

    category_accuracy[category][0] += if_accurate
    category_accuracy[category][1] += 1
    
    # Update category accuracy
    if tally == 80:
        add_to_csv(context_length, category_accuracy)
        print("New repeat for {context_length=}")
        tally = 0 # clear tally for next repeat
        category_accuracy = defaultdict(lambda: [0, 0])

# Write the updated data back to the same file
with open(completion_path, 'w') as file:
    json.dump(data, file, indent=4)