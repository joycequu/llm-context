import jailbreakbench as jbb

artifact = jbb.read_artifact(
    method="GCG",
    model_name="vicuna-13b-v1.5"
)

# print(artifact)

print(len(artifact))
# print(artifact.jailbreaks[75])

# models = ['vicuna-13b-v1.5', 'llama-2-7b-chat-hf', 'gpt-3.5-turbo-1106', 'gpt-4-0125-preview']