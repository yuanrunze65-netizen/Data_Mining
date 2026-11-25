"""Draw the report's vector figures from the saved FinanceQA experiment."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import PercentFormatter

INK, MUTED, GRID = "#263747", "#556879", "#E2E9EF"
TEAL, RUST, GREY, BLUE = "#197A74", "#BD573C", "#6F7D8B", "#416E9D"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": GRID,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.bbox": "tight", "savefig.pad_inches": .05, "pdf.fonttype": 42,
    })


def save(fig, out, filename):
    fig.savefig(out / filename)
    plt.close(fig)


def outcomes(rows, out, filename, xlabel, ticks):
    keys = ["correct_info", "incorrect_info", "insufficient_info"]
    categories = [("Supported", TEAL, ""), ("Contradictory", RUST, "//"), ("Insufficient", GREY, "..")]
    for _, row in rows:
        if not row["evaluation_complete"] or sum(row[key] for key in keys) != row["expected_questions"]:
            raise ValueError("Chart requires complete scoring for every question")
    fig, ax = plt.subplots(figsize=(4.15, 1.7), layout="constrained")
    y = list(range(len(rows)))
    total = max(row["expected_questions"] for _, row in rows)
    ax.set(yticks=y, yticklabels=[name for name, _ in rows], xticks=ticks,
           xlim=(0, total * 1.26), ylim=(len(rows) - .48, -.55), xlabel=xlabel)
    ax.tick_params(axis="both", length=0, pad=3)
    ax.xaxis.grid(True, color=GRID, linewidth=.65)
    ax.set_axisbelow(True)
    legend = []
    for key, (label, color, hatch) in zip(keys, categories):
        if any(row[key] for _, row in rows):
            legend.append(Patch(facecolor=color, edgecolor="white", hatch=hatch, label=label))
    for band, (_, row) in enumerate(rows):
        left = 0
        for key, (_, color, hatch) in zip(keys, categories):
            n = row[key]
            if n:
                ax.barh(band, n, left=left, height=.63, color=color, edgecolor="white", hatch=hatch, zorder=3)
                # Narrow cells use a leader to keep their counts legible.
                if n / total < .04:
                    ax.annotate(str(n), (left + n / 2, band + .30), (left + n / 2, band + .47),
                                ha="center", va="top", fontsize=7,
                                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": .6})
                else:
                    ax.text(left + n / 2, band, str(n), ha="center", va="center", color="white",
                            fontweight="bold", fontsize=8, bbox={"facecolor": color, "edgecolor": "none", "pad": .45})
            left += n
        ax.text(total * 1.035, band, f"{row['support_percentage']:.1f}%", ha="left", va="center", fontweight="bold")
    ax.legend(handles=legend, ncol=len(legend), loc="lower left", bbox_to_anchor=(0, 1),
              frameon=False, fontsize=7, handlelength=1.2, columnspacing=1.2, borderaxespad=0)
    save(fig, out, filename)


def revenue(root, out):
    trace_path = root / "train-hybrid-v2/trace.jsonl"
    with trace_path.open(encoding="utf-8") as file:
        trace = next(row for line in file if (row := json.loads(line))["question_id"] == "FQA-1640147-05")
    facts = [row for call in trace["calls"] if call.get("tool") == "annual"
             for row in call["result"]["rows"]
             if row["fact_name"] == "RevenueFromContractWithCustomerExcludingAssessedTax"
             and row["unit"] == "USD" and row["period_basis"] == "documented annual duration"]
    by_year = {int(row["end_date"][:4]): row["fact_value"] / 1e9 for row in facts}
    years = sorted(by_year)
    if years != list(range(2021, 2026)):
        raise ValueError("Unexpected periods in the Snowflake trace")
    amounts = [by_year[year] for year in years]
    growth = [100 * (current / previous - 1) for previous, current in zip(amounts, amounts[1:])]
    fig, (upper, lower) = plt.subplots(2, 1, figsize=(4.15, 2.48), sharex=True, layout="constrained")
    for ax, xs, ys, ylabel, color, limit, ticks, label in [
        (upper, years, amounts, "Revenue (USD bn)", BLUE, 4.5, [0, 2, 4], lambda x: f"{x:.2f}"),
        (lower, years[1:], growth, "YoY growth (%)", RUST, 140, [0, 60, 120], lambda x: f"{x:.1f}%"),
    ]:
        ax.plot(xs, ys, color=color, marker="o", markersize=3.6, markerfacecolor="white", lw=1.55)
        for x, value in zip(xs, ys):
            ax.annotate(label(value), (x, value), (0, 5), textcoords="offset points", ha="center", color=color, fontsize=7)
        ax.set(ylabel=ylabel, ylim=(0, limit), yticks=ticks, xlim=(2020.62, 2025.38))
        ax.grid(axis="y", color=GRID, lw=.65)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
    lower.set(xticks=years, xticklabels=[f"FY{year}" for year in years])
    lower.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0, symbol=""))
    save(fig, out, "snowflake-revenue.pdf")


def workflow(out):
    fig, ax = plt.subplots(figsize=(7.25, 1.7), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 2.75))
    ax.axis("off")

    def box(x, y, width, height, title, subtitle, edge, fill):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.05,rounding_size=.12",
                                    facecolor=fill, edgecolor=edge, linewidth=1.1))
        ax.text(x + width / 2, y + height * .64, title, ha="center", va="center", fontsize=9, fontweight="medium")
        ax.text(x + width / 2, y + height * .29, subtitle, ha="center", va="center", fontsize=7, color=MUTED)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1, color=MUTED))

    box(.2, 1.73, 2.35, .72, "Financial question", "Company ID + question", BLUE, "#EFF4F8")
    box(4.43, 1.73, 3.15, .72, "Flash chooses next action", "Use the question and evidence so far", TEAL, "#EAF4F3")
    box(9.45, 1.73, 2.35, .72, "Answer", "Conclusion + source IDs", BLUE, "#EFF4F8")
    box(3.35, .15, 5.3, .72, "Run the requested tool", "Look up financial facts  /  Search filings  /  Calculate", TEAL, "#EAF4F3")
    arrow((2.65, 2.09), (4.32, 2.09))
    arrow((7.68, 2.09), (9.34, 2.09))
    ax.text(8.49, 2.25, "Ready to answer", ha="center", fontsize=7, color=MUTED)
    arrow((5.57, 1.65), (5.57, .96))
    arrow((6.44, .96), (6.44, 1.65))
    ax.text(5.42, 1.3, "Tool request", ha="right", va="center", fontsize=7, color=MUTED)
    ax.text(6.59, 1.3, "Tool result", ha="left", va="center", fontsize=7, color=MUTED)
    save(fig, out, "financeqa-workflow.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/financeqa"))
    parser.add_argument("--output", type=Path, default=Path("report/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    d = args.results
    outcomes([("Baseline", read(d / "train-basic-v2/evaluation-recovered.json")),
              ("Enhanced", read(d / "train-hybrid-v2/evaluation.json"))],
             args.output, "financeqa-development.pdf", "Development questions", [0, 10, 20, 30, 37])
    test = read(d / "test-v2/evaluation.json")["per_company"]
    outcomes([("Palantir", test["1321655"]), ("Abbott", test["1800"])],
             args.output, "financeqa-test.pdf", "Test questions", [0, 2, 4, 6, 8, 10])
    revenue(d, args.output)
    workflow(args.output)


if __name__ == "__main__":
    main()
