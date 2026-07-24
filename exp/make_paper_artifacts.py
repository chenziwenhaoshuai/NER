from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DATASETS = ["SMD", "MSL", "SMAP", "PSM", "SWaT"]
MODELS = ["AnomalyTransformer", "Transformer", "Autoformer", "TimesNet", "KANAD"]
FINAL = "Counterfactual-temperature MoE"
VARIANT_LABELS = {
    "Baseline": "Baseline",
    "+ Temporal calibration": "+ Temporal calibration",
    "Prior + NMS": "+ Event prior",
    "SelfNet only": "SelfNet only",
    "Geometry ConvAE only": "Geometry ConvAE only",
    "Augmented ConvAE only": "Score-augmented ConvAE only",
    "Neural-only mixed": "Neural-only mixed",
    FINAL: r"\textbf{NER}",
}


def fmt(value: float, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}"


def signed(value: float, digits: int = 2) -> str:
    return f"{float(value):+.{digits}f}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def require(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. {hint}")


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def tex_table(
    caption: str,
    label: str,
    columns: str,
    header: str,
    rows: list[str],
    size: str = "small",
    tabcolsep: float = 4.0,
    star: bool = False,
) -> str:
    environment = "table*" if star else "table"
    return (
        f"\\begin{{{environment}}}[t]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\n"
        f"\\begin{{{size}}}\n"
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep:.1f}pt}}\n"
        f"\\begin{{tabular}}{{{columns}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\end{{{size}}}\n"
        f"\\end{{{environment}}}\n"
    )


def load_main(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_dir = results_dir / "main"
    require(main_dir / "pair_metrics.csv", "Run `python reproduce.py` first.")
    pair = pd.read_csv(main_dir / "pair_metrics.csv")
    overall = pd.read_csv(main_dir / "overall_metrics.csv")
    dataset = pd.read_csv(main_dir / "dataset_metrics_fraction.csv")
    backbone = pd.read_csv(main_dir / "backbone_metrics_fraction.csv")
    return pair, dataset, backbone


def write_main_tables(pair: pd.DataFrame, dataset: pd.DataFrame, backbone: pd.DataFrame, table_dir: Path) -> None:
    metrics = ["pa_f1", "event_f1", "range_f1"]
    data = dataset[dataset["variant"].isin(["Baseline", FINAL])].copy()
    for metric in metrics:
        data[metric] *= 100.0
    baseline = data[data["variant"] == "Baseline"].set_index("dataset")
    final = data[data["variant"] == FINAL].set_index("dataset")

    rows = []
    for dataset_name in DATASETS:
        b = baseline.loc[dataset_name]
        m = final.loc[dataset_name]
        rows.append(
            f"{dataset_name} & {fmt(b.pa_f1)} & {fmt(b.event_f1)} & {fmt(b.range_f1)} "
            f"& {fmt(m.pa_f1)} & {fmt(m.event_f1)} & {fmt(m.range_f1)} "
            f"& {signed(m.pa_f1 - b.pa_f1)} & {signed(m.event_f1 - b.event_f1)} & {signed(m.range_f1 - b.range_f1)} \\\\"
        )
    b_avg = baseline[metrics].mean()
    m_avg = final[metrics].mean()
    rows.append(
        "\\midrule\n"
        f"\\textbf{{Average}} & {fmt(b_avg.pa_f1)} & {fmt(b_avg.event_f1)} & {fmt(b_avg.range_f1)} "
        f"& \\textbf{{{fmt(m_avg.pa_f1)}}} & \\textbf{{{fmt(m_avg.event_f1)}}} & \\textbf{{{fmt(m_avg.range_f1)}}} "
        f"& \\textbf{{{signed(m_avg.pa_f1 - b_avg.pa_f1)}}} & \\textbf{{{signed(m_avg.event_f1 - b_avg.event_f1)}}} & \\textbf{{{signed(m_avg.range_f1 - b_avg.range_f1)}}} \\\\"
    )
    write(
        table_dir / "table_main_dataset_event_range.tex",
        tex_table(
            "Main evidence across datasets. Each number is averaged over five backbones.",
            "tab:main_results",
            "lccc|ccc|ccc",
            (
                "Dataset & \\multicolumn{3}{c|}{Baseline} & \\multicolumn{3}{c|}{NER} "
                "& \\multicolumn{3}{c}{Gain} \\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
                "& PA-F1 & Event F1 & Range F1 & PA-F1 & Event F1 & Range F1 "
                "& $\\Delta$PA & $\\Delta$Event & $\\Delta$Range \\\\"
            ),
            rows,
            star=True,
        ),
    )

    backbone_data = backbone[backbone["variant"].isin(["Baseline", FINAL])].copy()
    backbone_data["pa_f1"] *= 100.0
    b_backbone = backbone_data[backbone_data["variant"] == "Baseline"].set_index("model")
    m_backbone = backbone_data[backbone_data["variant"] == FINAL].set_index("model")
    rows = []
    for model in MODELS:
        b = float(b_backbone.loc[model, "pa_f1"])
        m = float(m_backbone.loc[model, "pa_f1"])
        rows.append(f"{model} & {fmt(b)} & {fmt(m)} & {signed(m - b)} \\\\")
    write(
        table_dir / "table_backbone_average_pa_f1.tex",
        tex_table(
            "Backbone-average PA-F1. Each row averages over five datasets.",
            "tab:backbone_mean",
            "lccc",
            "Backbone & Baseline & NER & $\\Delta$ \\\\",
            rows,
        ),
    )

    pair_data = pair[pair["variant"].isin(["Baseline", FINAL])].copy()
    pair_data["pa_f1"] *= 100.0
    b_pair = pair_data[pair_data["variant"] == "Baseline"].set_index(["dataset", "model"])
    m_pair = pair_data[pair_data["variant"] == FINAL].set_index(["dataset", "model"])
    rows = []
    for dataset_name in DATASETS:
        for index, model in enumerate(MODELS):
            b = float(b_pair.loc[(dataset_name, model), "pa_f1"])
            m = float(m_pair.loc[(dataset_name, model), "pa_f1"])
            rows.append(f"{dataset_name if index == 0 else ''} & {model} & {fmt(b)} & {fmt(m)} & {signed(m - b)} \\\\")
        if dataset_name != DATASETS[-1]:
            rows.append("\\midrule")
    write(
        table_dir / "table_full_pair_transfer_pa_f1.tex",
        tex_table(
            "Complete pair-level transfer results. Values are PA-F1.",
            "tab:full_pair_transfer",
            "llrrr",
            "Dataset & Backbone & Baseline & NER & $\\Delta$ \\\\",
            rows,
        ),
    )


def write_component_and_guardrail_tables(pair: pd.DataFrame, table_dir: Path) -> None:
    order = [
        "Baseline",
        "+ Temporal calibration",
        "Prior + NMS",
        "SelfNet only",
        "Geometry ConvAE only",
        "Augmented ConvAE only",
        "Neural-only mixed",
        FINAL,
    ]
    grouped = pair.groupby("variant", as_index=False).mean(numeric_only=True)
    for column in [
        "pa_precision",
        "pa_recall",
        "pa_f1",
        "point_f1",
        "event_f1",
        "range_precision",
        "range_recall",
        "range_f1",
        "strict_event_precision",
        "strict_event_recall",
        "strict_event_f1",
        "fpr",
    ]:
        grouped[column] *= 100.0
    indexed = grouped.set_index("variant")
    rows = []
    for variant in order:
        row = indexed.loc[variant]
        rows.append(
            f"{VARIANT_LABELS[variant]} & {fmt(row.pa_precision)} & {fmt(row.pa_recall)} "
            f"& {fmt(row.pa_f1)} & {fmt(row.event_f1)} & {fmt(row.range_f1)} \\\\"
        )
    write(
        table_dir / "table_component_ablation.tex",
        tex_table(
            "Component analysis averaged over 25 backbone--dataset pairs.",
            "tab:component_ablation",
            "lrrrrr",
            "Method & P & R & PA-F1 & Event F1 & Range F1 \\\\",
            rows,
            tabcolsep=3.2,
        ),
    )

    rows = []
    for variant in ["Baseline", "+ Temporal calibration", "Prior + NMS", "Neural-only mixed", FINAL]:
        row = indexed.loc[variant]
        rows.append(
            f"{VARIANT_LABELS[variant]} & {fmt(row.point_f1)} & {fmt(row.event_f1)} "
            f"& {fmt(row.range_precision)} & {fmt(row.range_recall)} & {fmt(row.range_f1)} & {fmt(row.fpr)} \\\\"
        )
    write(
        table_dir / "table_event_range_metrics.tex",
        tex_table(
            "Complementary pointwise, event, and range-overlap metrics.",
            "tab:event_range_metrics",
            "lrrrrrr",
            "Method & Point F1 & Event F1 & Range P & Range R & Range F1 & FPR \\\\",
            rows,
            size="scriptsize",
            tabcolsep=2.8,
        ),
    )

    rows = []
    for variant in ["Baseline", "+ Temporal calibration", "Prior + NMS", "Neural-only mixed", FINAL]:
        row = indexed.loc[variant]
        rows.append(
            f"{VARIANT_LABELS[variant]} & {fmt(row.strict_event_precision)} & {fmt(row.strict_event_recall)} "
            f"& {fmt(row.strict_event_f1)} & {fmt(row.false_events_per_100k)} \\\\"
        )
    write(
        table_dir / "table_strict_event_metrics.tex",
        tex_table(
            "Strict one-to-one event metrics. Each ground-truth event can be matched by at most one predicted segment.",
            "tab:strict_event_metrics",
            "lrrrr",
            "Method & Strict Event P & Strict Event R & Strict Event F1 & False Events / 100K \\\\",
            rows,
            size="scriptsize",
        ),
    )


def write_random_table(results_dir: Path, table_dir: Path) -> None:
    path = results_dir / "random_guardrail" / "random_insertion_pair_seed_metrics.csv"
    require(path, "Run `python exp/random_insertion_guardrail.py` first.")
    raw = pd.read_csv(path)
    columns = [
        "pa_f1",
        "point_f1",
        "event_f1",
        "range_f1",
        "strict_event_recall",
        "fpr",
        "false_events_per_100k",
    ]
    grouped = raw.groupby("method", as_index=False)[columns].mean(numeric_only=True).set_index("method")
    for column in columns:
        if column != "false_events_per_100k":
            grouped[column] *= 100.0
    labels = {
        "Baseline": "Baseline",
        "+ Temporal calibration": "+ Temporal calibration",
        "Baseline + Random": "Baseline + Random NMS",
        "Temporal + Random": "Temporal + Random NMS",
        "Prior + NMS": "Prior + NMS",
        "Neural Event Rescue": "NER",
    }
    rows = []
    for method in labels:
        row = grouped.loc[method]
        rows.append(
            f"{labels[method]} & {fmt(row.pa_f1)} & {fmt(row.point_f1)} & {fmt(row.event_f1)} "
            f"& {fmt(row.strict_event_recall)} & {fmt(row.range_f1)} & {fmt(row.fpr)} "
            f"& {fmt(row.false_events_per_100k)} \\\\"
        )
    write(
        table_dir / "table_random_insertion_guardrail.tex",
        tex_table(
            "Random-insertion guardrail averaged over 25 backbone--dataset pairs.",
            "tab:random_guardrail",
            "lrrrrrrr",
            "Method & PA-F1 & Point F1 & Event F1 & Strict Event R & Range F1 & FPR & False Events/100K \\\\",
            rows,
            size="scriptsize",
            tabcolsep=3.0,
        ),
    )


def write_center_table(results_dir: Path, table_dir: Path) -> None:
    path = results_dir / "center_to_interval" / "center_to_interval_summary.csv"
    require(path, "Run `python exp/center_to_interval.py` first.")
    summary = pd.read_csv(path)
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"{int(row.radius)} & {fmt(100 * row.pa_f1)} & {fmt(100 * row.event_f1)} "
            f"& {fmt(100 * row.range_precision)} & {fmt(100 * row.range_recall)} "
            f"& {fmt(100 * row.range_f1)} & {fmt(row.false_events_per_100k)} \\\\"
        )
    write(
        table_dir / "table_center_to_interval_conversion.tex",
        tex_table(
            "Converting rescued event centers into finite alarm intervals. Radius zero is the center-only decision used by the main method.",
            "tab:center_to_interval",
            "lrrrrrr",
            "Radius & PA-F1 & Event F1 & Range P & Range R & Range F1 & False Events/100K \\\\",
            rows,
        ),
    )


def write_moe_tables(artifact_dir: Path, table_dir: Path) -> None:
    temperature_path = artifact_dir / "reference" / "temperature_selection.csv"
    require(temperature_path, "Use the v35 release artifacts.")
    temperature = pd.read_csv(temperature_path)
    rows = []
    for _, row in temperature.iterrows():
        rows.append(
            f"{row.dataset} & {fmt(row.weight_self, 3)} & {fmt(row.weight_geometry, 3)} "
            f"& {fmt(row.weight_augmented, 3)} & {fmt(row.selected_gamma, 1)} & {fmt(row.best_ap, 3)} \\\\"
        )
    write(
        table_dir / "table_moe_temperature_selection.tex",
        tex_table(
            "Label-free MoE calibration profile.",
            "tab:moe_temperature",
            "lrrrrr",
            "Dataset & Self & Geometry & Score-aug. & $\\gamma$ & Probe AP \\\\",
            rows,
            tabcolsep=4.3,
        ),
    )


def plot_operating_profile(results_dir: Path, figure_dir: Path) -> None:
    path = results_dir / "sensitivity" / "operating_sensitivity_summary.csv"
    require(path, "Run `python exp/operating_sensitivity.py` first.")
    summary = pd.read_csv(path)
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.55))
    budget = summary[summary["experiment"] == "budget"].sort_values("multiplier")
    spacing = summary[summary["experiment"] == "spacing"].sort_values("multiplier")
    colors = {"PA-F1": "#2F6DAE", "Event F1": "#D55E00", "Range F1": "#6A51A3"}
    for metric, label in [("pa_f1", "PA-F1"), ("event_f1", "Event F1"), ("range_f1", "Range F1")]:
        axes[0].plot(
            budget["multiplier"].to_numpy(float),
            (100.0 * budget[metric]).to_numpy(float),
            marker="o",
            lw=1.4,
            ms=3.5,
            color=colors[label],
            label=label,
        )
        axes[1].plot(
            spacing["multiplier"].to_numpy(float),
            (100.0 * spacing[metric]).to_numpy(float),
            marker="o",
            lw=1.4,
            ms=3.5,
            color=colors[label],
            label=label,
        )
    false_axis = axes[0].twinx()
    false_axis.plot(
        budget["multiplier"].to_numpy(float),
        budget["false_events_per_100k"].to_numpy(float),
        marker="s",
        lw=1.1,
        ms=3.2,
        color="#666666",
        label="False events / 100K",
    )
    false_axis.set_ylabel("False events / 100K", color="#666666")
    false_axis.tick_params(axis="y", colors="#666666", width=0.6)
    false_axis.spines["top"].set_visible(False)
    false_axis.spines["right"].set_color("#888888")
    axes[0].set_xscale("symlog", linthresh=0.25)
    axes[0].set_xticks([0, 0.25, 0.5, 1, 2, 4, 8, 16])
    axes[0].set_xticklabels(["0", "0.25", "0.5", "1", "2", "4", "8", "16"])
    axes[0].set_title("(a) Alert budget")
    axes[1].set_title("(b) Refractory spacing")
    axes[0].set_xlabel("Budget multiplier")
    axes[1].set_xlabel("Spacing multiplier")
    axes[0].set_ylabel("Metric (%)")
    axes[1].set_ylabel("Metric (%)")
    axes[1].set_xticks(spacing["multiplier"].to_numpy(float))
    for axis in axes:
        axis.grid(axis="y", color="#DDDDDD", lw=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(width=0.6, length=3)
    handles, labels = axes[0].get_legend_handles_labels()
    handles2, labels2 = false_axis.get_legend_handles_labels()
    fig.legend(
        handles + handles2,
        labels + labels2,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "fig_operating_profile_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_operating_profile_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_stress(results_dir: Path, figure_dir: Path) -> None:
    path = results_dir / "evidence_stress" / "evidence_stress_pair_metrics.csv"
    require(path, "Run `python exp/evidence_stress.py` first.")
    raw = pd.read_csv(path)
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    for axis, family in zip(axes, ["noise", "drift", "dropout"]):
        current = raw[raw["family"] == family].groupby(
            ["severity", "method"], as_index=False
        )["event_f1"].mean()
        for method, color, marker in [
            ("Prior + NMS", "#777777", "s"),
            ("Neural Event Rescue", "#0072B2", "o"),
        ]:
            line = current[current["method"] == method].sort_values("severity")
            axis.plot(
                line["severity"].to_numpy(float),
                (100.0 * line["event_f1"]).to_numpy(float),
                color=color,
                marker=marker,
                lw=1.35,
                ms=3.5,
                label=method,
            )
        axis.set_title(family.capitalize())
        axis.set_xlabel("Evidence perturbation")
        axis.grid(axis="y", color="#DDDDDD", lw=0.5)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Event F1 (%)")
    axes[0].legend(frameon=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_candidate_evidence_stress.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_candidate_evidence_stress.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_center_to_interval(results_dir: Path, figure_dir: Path) -> None:
    path = results_dir / "center_to_interval" / "center_to_interval_summary.csv"
    require(path, "Run `python exp/center_to_interval.py` first.")
    summary = pd.read_csv(path)
    configure_plot_style()
    fig, axis = plt.subplots(figsize=(4.2, 2.7))
    for metric, label, color, marker in [
        ("pa_f1", "PA-F1", "#0072B2", "o"),
        ("event_f1", "Event F1", "#D55E00", "s"),
        ("range_f1", "Range F1", "#7B61A8", "^"),
    ]:
        axis.plot(
            summary["radius"].to_numpy(float),
            (100.0 * summary[metric]).to_numpy(float),
            marker=marker,
            linewidth=1.4,
            markersize=4,
            label=label,
            color=color,
        )
    axis.set_xlabel("Interval expansion radius")
    axis.set_ylabel("Metric (%)")
    axis.grid(axis="y", color="#dddddd", linewidth=0.5)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False)
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_center_to_interval_conversion.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_center_to_interval_conversion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready LaTeX tables and quantitative figures from reproduced NER results."
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts" / "v35")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "paper_artifacts")
    args = parser.parse_args()

    pair, dataset, backbone = load_main(args.results_dir)
    table_dir = args.output_dir / "tables"
    figure_dir = args.output_dir / "figures"
    write_main_tables(pair, dataset, backbone, table_dir)
    write_component_and_guardrail_tables(pair, table_dir)
    write_random_table(args.results_dir, table_dir)
    write_center_table(args.results_dir, table_dir)
    write_moe_tables(args.artifact_dir, table_dir)
    plot_operating_profile(args.results_dir, figure_dir)
    plot_evidence_stress(args.results_dir, figure_dir)
    plot_center_to_interval(args.results_dir, figure_dir)
    print(f"Paper artifacts written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
