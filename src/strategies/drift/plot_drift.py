import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

sns.set_style("darkgrid")


def main(
    input_path: Path,
    output_path: Path,
    benchmark: Optional[str] = typer.Option(None, "-b", "--benchmark"),
    model_name: Optional[str] = typer.Option(None, "-m", "--model_name"),
    replay: bool = typer.Option(False, "-r", "--replay"),
    memory_size: Optional[int] = typer.Option(None, "-s", "--memory_size"),
):
    with input_path.open("r") as f:
        stats = json.load(f)

    save_drift_plot(
        stats,
        output_path,
        benchmark=benchmark,
        model_name=model_name,
        replay=replay,
        memory_size=memory_size,
    )


def save_drift_plot(
    drift_stats: dict,
    output_path: Path,
    benchmark: Optional[str] = None,
    model_name: Optional[str] = None,
    replay: bool = False,
    ewc_lambda: float = 0.0,
    lwf_alpha: float = 0.0,
    lwf_temperature: float = 0.0,
    memory_size: Optional[int] = None,
) -> None:
    df = []
    for exp_idx, layers_info in drift_stats.items():
        for layer_idx, drift_stats in layers_info.items():
            for epoch, drift in enumerate(drift_stats):
                df.append(
                    {
                        "epoch": int(epoch + 1),
                        "cosine_dist": float(drift),
                        "exp_idx": int(exp_idx),
                        "layer_idx": int(layer_idx),
                    }
                )

    df = pd.DataFrame(df)
    exp_ids = df["exp_idx"].unique().tolist()
    palette = {
        0: "Greys",
        1: "Reds",
        2: "Blues",
        3: "Greens",
        4: "Oranges",
        5: "Purples",
        6: "PuRd",
        7: "RdPu",
        8: "BuGn",
        9: "YlGn",
    }

    max_layer = int(df["layer_idx"].max())

    for exp_idx in exp_ids:
        data = df[df["exp_idx"] == exp_idx]
        plot = sns.lineplot(
            data=data,
            x="epoch",
            y="cosine_dist",
            hue="layer_idx",
            hue_norm=(0, max_layer),
            palette=palette[exp_idx],
        )
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles[:max_layer], title="layer_idx")
    plt.ylim(0, 1.0)
    title = ""
    if benchmark is not None:
        title += benchmark
    if model_name is not None:
        if title:
            title += " "
        title += model_name
    if replay is True:
        assert memory_size is not None
        if title:
            title += ", "
        title += f"replay, mem size: {memory_size}"
    if ewc_lambda > 0:
        if title:
            title += ", "
        title += f"EWC lambda: {ewc_lambda}"
    if lwf_alpha > 0:
        if title:
            title += ", "
        title += f"LWF alpha: {lwf_alpha}, LWF temp: {lwf_temperature}"
    plot.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path)


if __name__ == "__main__":
    typer.run(main)
