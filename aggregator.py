import re
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI


client = OpenAI()

def model_fn(prompts, model="gpt-3.5-turbo", temperature=0):
    responses = []
    for prompt in prompts:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        responses.append(r.choices[0].message.content)
    return responses


@dataclass
class BenchmarkResults:
    name: str
    correct: int = 0
    total: int = 0
    failures: list = field(default_factory=list)

    @property
    def accuracy(self):
        return self.correct / self.total if self.total > 0 else 0.0

    def __repr__(self):
        return f"{self.name}: {self.correct}/{self.total} ({self.accuracy:.1%})"


def extract_answer(response: str, benchmark: BenchmarkType) -> Optional[str]:
    patterns = {
        BenchmarkType.SVAMP:       r"[Tt]he answer is ([\d,\.]+)",
        BenchmarkType.GSM8K:       r"[Tt]he answer is ([\d,\.]+)",
        BenchmarkType.AQUA:        r"[Tt]he answer is \(?([a-eA-E])\)?",
        BenchmarkType.STRATEGY_QA: r"[Tt]he answer is (yes|no|True|False)",
    }
    match = re.search(patterns[benchmark], response)
    return match.group(1).replace(",", "").strip() if match else None


def grade_answer(predicted: Optional[str], gold: str, benchmark: BenchmarkType) -> bool:
    if predicted is None:
        return False
    
    if benchmark in (BenchmarkType.SVAMP, BenchmarkType.GSM8K):
        try:
            return float(predicted) == float(gold.replace(",", ""))
        except ValueError:
            return False
    
    if benchmark == BenchmarkType.AQUA:
        return predicted.strip().lower() == gold.strip().lower()
    
    if benchmark == BenchmarkType.STRATEGY_QA:
        yes_no_to_bool = {"yes": "true", "no": "false"}
        normalized = yes_no_to_bool.get(predicted.strip().lower(), predicted.strip().lower())
        return normalized == gold.strip().lower()

    return predicted.strip().lower() == gold.strip().lower()


def aggregate(model_fn, dataloader, benchmark: BenchmarkType, cot: bool = True) -> BenchmarkResults:
    results = BenchmarkResults(name=benchmark.value)

    for batch in dataloader:
        questions = batch['question']
        gold_answers = batch['final_answer']
        prompts = [build_prompt(q, benchmark, cot=cot) for q in questions]
        responses = model_fn(prompts)

        for q, response, gold in zip(questions, responses, gold_answers):
            predicted = extract_answer(response, benchmark)
            correct = grade_answer(predicted, gold, benchmark)
            results.total += 1
            if correct:
                results.correct += 1
            else:
                results.failures.append({
                    "question": q,
                    "predicted": predicted,
                    "gold": gold,
                    "raw_response": response,
                })

    return results


DATALOADER_BENCHMARK_MAP = {
    "svamp":       (svamp_dataloader,       BenchmarkType.SVAMP),
    "aqua":        (aqua_dataloader,        BenchmarkType.AQUA),
    "gsm8k":       (gsm8k_dataloader,       BenchmarkType.GSM8K),
    "strategy_qa": (strategy_qa_dataloader, BenchmarkType.STRATEGY_QA),
}

all_results = []
for name, (dataloader, benchmark) in DATALOADER_BENCHMARK_MAP.items():
    print(f"Evaluating {name}...")
    results = aggregate(model_fn, dataloader, benchmark, cot=True)
    all_results.append(results)
    print(results)

print("\n=== Summary ===")
for r in all_results:
    print(r)

overall_correct = sum(r.correct for r in all_results)
overall_total = sum(r.total for r in all_results)
print(f"\nOverall: {overall_correct}/{overall_total} ({overall_correct/overall_total:.1%})")