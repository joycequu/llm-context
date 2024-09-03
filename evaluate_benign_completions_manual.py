# read in benign answers and evaluate if answered
# input: json, output: csv
import json

completion_path = "answer_mistral_general_dan_news_benign.json"
output_path = "results/mistral_general_dan_news_benign_accuracy.csv"
# Open the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Process each item one by one
for item in data:
    # Do something with the item
    print(item)
