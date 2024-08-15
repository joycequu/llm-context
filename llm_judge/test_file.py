import json

# Open the .jsonl file
with open('/Users/joycequ/Documents/UROP/Context/llm-context/llm_judge/vicuna_bench_questions.jsonl', 'r') as file:
    # Iterate over each line in the file
    for line in file:
        # Parse the JSON object from the line
        json_object = json.loads(line)
        # Process the JSON object
        print(json_object)

print(json_object.get('question_id'))