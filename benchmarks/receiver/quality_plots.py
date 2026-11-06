"""Plot one saved quality report; no Torch or checkpoint loading required."""

import matplotlib.pyplot as plt

LABELS = {
    "bleu": "Corpus BLEU (0–100)",
    "chrf": "Corpus chrF++ (0–100)",
    "semantic_similarity": "Semantic similarity (cosine)",
    "rouge_l": "ROUGE-L F1 (0–1)",
    "token_accuracy": "BPE token accuracy (0–1)",
}
MODELS = ("M0", "M1", "M2", "M3")
COLORS = ("#4477AA", "#228833", "#AA3377", "#CC6600")
NAMES = ("M0 · Teacher", "M1 · CE only", "M2 · Combined KD", "M3 · Liu adaptation")


def plot_quality(summary, manifest):
    """Plot means and sample SD across channel trials for a single training seed."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, LABELS.items()):
        for model, color, name in zip(MODELS, COLORS, NAMES):
            rows = summary.loc[summary["model"].eq(model)].sort_values("snr_db")
            if rows.empty:
                raise ValueError(f"Quality summary is missing {model}")
            x = rows["snr_db"].to_numpy()
            mean = rows[f"{metric}_mean"].to_numpy()
            sd = rows[f"{metric}_sd"].to_numpy()
            ax.plot(x, mean, marker="o", linewidth=1.5, color=color, label=name)
            if rows["trials"].min() > 1:
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.11)
        ax.set(xlabel="Nominal SNR (dB)", ylabel=label)
        ax.grid(alpha=0.25)
    legend = axes.flat[-1]
    legend.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    legend.legend(handles, labels, loc="upper left", frameon=False)
    protocol = manifest["protocol"]
    split = "Validation" if protocol["split"] == "val" else "Held-out test"
    if protocol["partial"]:
        split += " subset"
    description = (
        f"{split} · {protocol['channel']} · {protocol['examples']:,} examples\n"
        f"Training seed {protocol['training_seed']} · {len(protocol['trial_seeds'])} channel trials\n"
        "Shading: ±1 SD across trials"
    )
    if manifest.get("protocol_warnings"):
        description += "\nExploratory: training controls differ"
    legend.text(0.04, 0.30, description, transform=legend.transAxes, fontsize=9, va="top")
    fig.suptitle("Receiver reconstruction quality versus SNR", fontsize=15)
    return fig
