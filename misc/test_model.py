from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = '/home/gridsan/ywang5/hf/models/HarmBench-Llama-2-13b-cls'

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Perform a simple inference
input_text = "Hello, world!"
inputs = tokenizer(input_text, return_tensors='pt')
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))