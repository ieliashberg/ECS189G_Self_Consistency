from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

import matplotlib.pyplot as plt


def load_results(path: str) -> pd.DataFrame:
    file_path = Path(path)

    df = pd.read_csv(file_path)

    df = df.copy()
    df["k"] = df["k"].astype(int)
    return df[["model", "dataset", "method", "k", "accuracy"]]


def plot_self_consistency_curves(
    df: pd.DataFrame,
    output_dir: str,
    k_values: Iterable[int] | None = None,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if k_values is not None:
        df = df[df["k"].isin(list(k_values))]

    for dataset_name in sorted(df["dataset"].unique()):
        dataset_df = df[df["dataset"] == dataset_name].copy()

        plt.figure(figsize=(9, 5))
        for (model_name, method_name), subset in dataset_df.groupby(["model", "method"]):
            ordered = subset.sort_values("k")
            plt.plot(ordered["k"], ordered["accuracy"], marker="o", label=f"{model_name} ({method_name})")

        plt.title(f"{dataset_name}: accuracy vs k")
        plt.xlabel("k")
        plt.ylabel("accuracy")
        plt.grid(True, alpha=0.2)
        plt.legend(fontsize=8)
        plt.tight_layout()

        png_path = out_dir / f"{dataset_name}_accuracy_vs_k.png"
        plt.savefig(png_path, dpi=200)
        plt.close()


if __name__ == "__main__":
    data = load_results("results.csv")
    plot_self_consistency_curves(
        df=data,
        output_dir="plots",
        k_values=[1, 5, 10, 20, 40],
    )
