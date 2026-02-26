
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


EXAMPLES = {
    BenchmarkType.SVAMP: """Q: Each pack of dvds costs 76 dollars. If there is a discount of 25 dollars on each pack. How much do you have to pay to buy each pack?
A: The original price is 76 dollars. The discount is 25 dollars. So you have to pay 76 - 25 = 51 dollars. The answer is 51.""",

    BenchmarkType.STRATEGY_QA: """Q: Was the ship that recovered Apollo 13 named after a World War II battle?
A: Apollo 13 was recovered by the USS Iwo Jima. The Battle of Iwo Jima was fought in World War II. So the answer is yes.""",

    BenchmarkType.AQUA: """Q: A car is being driven towards the base of a vertical tower. It takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will the car reach the base?
Answer Choices: A)5(√3 + 1) B)6(√3 + √2) C)7(√3 – 1) D)8(√3 – 2) E)None of these
A: Let the height of the tower be h. At 45°, distance = h. At 60°, distance = h/√3. In 10 minutes it travelled h - h/√3. Time to travel h/√3 = 10*(1/√3)/(1-1/√3) = 5(√3+1). The answer is (a).""",

    BenchmarkType.GSM8K: """Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the remainder for $2 per egg. How much does she make every day?
A: Janet sells 16 - 3 - 4 = 9 eggs per day. She makes 9 * 2 = 18 dollars. The answer is 18.""",
}




def build_prompt(question: str, benchmark: BenchmarkType, cot: bool = True) -> str:
    parts = [SYSTEM_CONTEXTS[benchmark]]

    # add few shot examples
    if cot and benchmark in EXAMPLES:
        parts.append(EXAMPLES[benchmark])

    parts.append(f"Q: {question}")

    if cot:
        parts.append(COT_INSTRUCTIONS[benchmark])
        parts.append(f"Show your reasoning, then end with:\n{ANSWER_FORMATS[benchmark]}")
    else:
        parts.append(f"Respond only with:\n{ANSWER_FORMATS[benchmark]}")

    return "\n\n".join(parts)