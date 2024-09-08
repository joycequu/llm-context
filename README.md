# LLM-Context

## Description
- Program (Python) out-of-context defenses and test such defenses against different forms of in-context jailbreaking attacks on various LLMs, facilitated by established evaluation frameworks (HarmBench).
- Devise and investigate methods that enhance LLM’s ability to solve the given task using the provided in-context exemplars instead of relying on its prior knowledge.


## Evaluation Frame (References)
- When evaluating harmful prompts, evaluate_completions.py (implemented with FastChat + Llama instead of HarmBench) and eval_utils.py are modified from the same files in https://github.com/centerforaisafety/HarmBench.git.
- vicuna_bench_questions.jsonl is reformated from https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge/data/vicuna_bench.


## Models
- meta-llama/Meta-Llama-3.1-8B (for evaluate_completions.py)
- mistralai/Mistral-7B-Instruct-v0.2 (for genearte_completions.py)

## Steps:
1. Generate completions for harmful prompts.
2. Evaluate if jailbreaking is successful.
3. Evaluate if benign prompts still work.
4. Experiment with if the model will remember context/info prior to the DAN prompt.
