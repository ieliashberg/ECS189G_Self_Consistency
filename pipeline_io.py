from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List

from pipeline_types import EvalStats


def load_openai_key_from_envfile(env_path: str = ".env") -> None:
    if os.getenv("OPENAI_API_KEY"):
        return

    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == "OPENAI_API_KEY":
            os.environ["OPENAI_API_KEY"] = value.strip()
            return


def parse_csv_list(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def parse_k_values(raw: str) -> List[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if any(v <= 0 for v in values):
        raise ValueError("All k values must be positive integers.")
    return values


def write_results(rows: List[EvalStats], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp_utc", "model", "dataset", "method", "k", "correct", "total", "parsed", "accuracy"]
        )
        timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        for row in rows:
            writer.writerow(
                [
                    timestamp,
                    row.model,
                    row.dataset,
                    row.method,
                    row.k,
                    row.correct,
                    row.total,
                    row.parsed,
                    f"{row.accuracy:.6f}",
                ]
            )


def print_results_summary(rows: List[EvalStats], output_csv: str) -> None:
    print("\n=== Results Summary ===")
    for row in rows:
        print(
            f"{row.dataset:12s} | {row.method:16s} | k={row.k:<3d} | "
            f"acc={row.accuracy:.2%} ({row.correct}/{row.total}) | parsed={row.parsed}/{row.total}"
        )
    print(f"\nSaved: {output_csv}")
