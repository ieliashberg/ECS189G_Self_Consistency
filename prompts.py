def build_prompt(question, cot=True):
    if cot:
        return f"""Solve the following problem step by step.
Explain your reasoning clearly.

Question:
{question}

Final Answer:
"""
    else:
        return f"""Question:
{question}

Answer:
"""

