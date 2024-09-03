# read in benign answers and evaluate if answered
# input: json, output: csv
import json

# Open the JSON file
with open('data.json', 'r') as file:
    data = json.load(file)

# Process each item one by one
for item in data:
    # Do something with the item
    print(item)
