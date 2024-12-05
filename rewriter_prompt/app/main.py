import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

input_query = "What are the names of employees who earn more than $50,000 per year?"

prompt = f"""
You are an advanced AI assistant with expertise in natural language processing. Your primary role is to act as a rewriter that refines natural language queries for better understanding and alignment with SQL operations. Your task is to rewrite the given user prompt by fixing typos, improving clarity, and making it more precise and unambiguous. You are NOT required to generate SQL, only to rewrite the user prompt.

Your output must meet the following criteria:
1. **Clarity**: Ensure the rewritten prompt is unambiguous, concise, and easy to interpret.
2. **Preservation of Intent**: Retain the full meaning and intent of the original user prompt without omitting any critical information.
3. **Formal Language**: Use formal and precise language to enhance readability.
4. **Fixing Errors**: Correct any typos, grammatical errors, or ambiguous phrasing.
5. **Human-Readability**: Optimize the output so that it is understandable for both technical and non-technical users.
"""

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": input_query},
]

outputs = pipe(
    messages,
    max_new_tokens=256,
)

print(outputs[0]["generated_text"][-1])
