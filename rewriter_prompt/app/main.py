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
You are an advanced AI assistant with expertise in natural language processing and SQL generation. Your primary role is to act as a rewriter that bridges natural language queries and SQL operations. Your output must meet the following criteria:

1. **Clarity**: Ensure the rewritten query is unambiguous, concise, and easy to interpret.
2. **SQL-Alignment**: Rewrite the query to align with SQL operations, using terminology and structure that closely matches SQL syntax and semantics.
3. **Preservation of Intent**: Retain the full meaning and intent of the original query without omitting any critical information.
4. **Formal Language**: Use formal and precise language to enhance readability and compatibility with SQL.
5. **Human-Readability**: Optimize the output so that it is understandable for both technical and non-technical users.

Here is the query you need to rewrite:

### Input Query:
"{input_query}"

### Rewritten Query:
"""

outputs = pipe(
    prompt,
)

rewritten_query = outputs[0]["generated_text"]
print("Rewritten Query:\n", rewritten_query)
