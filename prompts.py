
from enum import Enum

class BenchmarkType(Enum):
    SVAMP = "svamp"
    AQUA = "aqua"
    GSM8K = "gsm8k"
    STRATEGY_QA = "strategy_qa"

SYSTEM_CONTEXTS = {
    BenchmarkType.SVAMP: "You are solving arithmetic word problems. Your final answer must be a number.",
    BenchmarkType.AQUA: "You are solving math problems with multiple choice answers. Your final answer must be one of the provided option letters (A, B, C, D, or E).",
    BenchmarkType.GSM8K: "You are solving grade school math problems. Your final answer must be a number.",
    BenchmarkType.STRATEGY_QA: "You are answering yes/no reasoning questions. Your final answer must be either 'True' or 'False'.",
}

COT_INSTRUCTIONS = {
    BenchmarkType.SVAMP: "Identify the quantities and operation needed, then compute step by step.",
    BenchmarkType.AQUA: "Evaluate each option methodically. Show your working before selecting.",
    BenchmarkType.GSM8K: "Break the problem into steps. Track all intermediate values.",
    BenchmarkType.STRATEGY_QA: "Think about what facts are needed to answer this, then reason to a yes/no conclusion.",
}


ANSWER_FORMATS = {
    BenchmarkType.SVAMP: "Final Answer: <number>",
    BenchmarkType.AQUA: "Final Answer: <letter>",
    BenchmarkType.GSM8K: "Final Answer: <number>",
    BenchmarkType.STRATEGY_QA: "Final Answer: <True or False>",
}

def build_prompt(question: str, benchmark: BenchmarkType, cot: bool = True) -> str:
    parts = [SYSTEM_CONTEXTS[benchmark], f"Question:\n{question}"]

    if cot:
        parts.append(COT_INSTRUCTIONS[benchmark])
        parts.append(f"Show your reasoning, then end with:\n{ANSWER_FORMATS[benchmark]}")
    else:
        parts.append(f"Respond only with:\n{ANSWER_FORMATS[benchmark]}")

    return "\n\n".join(parts)