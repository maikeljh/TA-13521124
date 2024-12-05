import torch
from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
)

input_query = "What are the names of employees who earn more than $50,000 per year?"

prompt = f"""
You are an expert assistant specialized in transforming natural language queries into a format suitable for SQL generation. Your task is to rewrite the input text query to make it clearer and closer to how SQL operations are described. Ensure the rewritten query is precise, unambiguous, and aligned with SQL terminology.

### Input Query:
"{input_query}"

### Rewritten Query:
"""

outputs = pipe(
    prompt,
)

rewritten_query = outputs[0]["generated_text"]
print("Rewritten Query:\n", rewritten_query)
