import jailbreakbench as jbb
import json

artifact_path = "/Users/joycequ/Documents/UROP/artifacts/attack-artifacts/GCG/white_box/vicuna-13b-v1.5.json"

# artifact = jbb.read_artifact(
#     method="GCG",
#     model_name="vicuna-13b-v1.5",
# )

with open(artifact_path, 'r') as file:
    artifact = json.load(file)

# print(artifact)

# print(len(artifact))
jailbreak_info = artifact.get('jailbreaks', [])
# print(jailbreak_info[75])

print(jailbreak_info[75]['prompt'])

# models = ['vicuna-13b-v1.5', 'llama-2-7b-chat-hf', 'gpt-3.5-turbo-1106', 'gpt-4-0125-preview']