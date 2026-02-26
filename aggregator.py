import re
from dataclasses import dataclass, field
from typing import Optional

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
    """Pull the final answer out of a model response."""
    patterns = {
        BenchmarkType.SVAMP:       r"Final Answer:\s*([\d,\.]+)",
        BenchmarkType.GSM8K:       r"Final Answer:\s*([\d,\.]+)",
        BenchmarkType.AQUA:        r"Final Answer:\s*([A-E])",
        BenchmarkType.STRATEGY_QA: r"Final Answer:\s*(True|False)",
    }
    match = re.search(patterns[benchmark], response, re.IGNORECASE)
    return match.group(1).replace(",", "").strip() if match else None


def grade_answer(predicted: Optional[str], gold: str, benchmark: BenchmarkType) -> bool:
    """Compare predicted vs gold, with benchmark-aware normalization."""
    if predicted is None:
        return False
    
    if benchmark in (BenchmarkType.SVAMP, BenchmarkType.GSM8K):
        try:
            return float(predicted) == float(gold.replace(",", ""))
        except ValueError:
            return False
    

    return predicted.strip().lower() == gold.strip().lower()


def aggregate(
    model_fn,         
    dataloader,
    benchmark: BenchmarkType,
    cot: bool = True,
) -> BenchmarkResults:
    
    name = benchmark.value
    results = BenchmarkResults(name=name)

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