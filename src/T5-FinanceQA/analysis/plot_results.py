"""Draw the report's vector figures from the saved FinanceQA experiment."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Patch, Polygon
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
    """Draw the decision loop and its three optional tool types."""
    font = FontProperties(fname=Path(__file__).resolve().parents[3] / "report/fonts/SimHei.ttf")
    fig, ax = plt.subplots(figsize=(7.25, 2.52))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 12), ylim=(0, 4.17), aspect="equal")
    ax.axis("off")
    azure, jade, amber = "#346EA3", "#137969", "#9D6124"

    def label(x, y, s, size=10, color=INK, align="center"):
        ax.text(x, y, s, fontsize=size, fontproperties=font, color=color,
                ha=align, va="center")

    def rounded(x, y, w, h, fill, edge="none", radius=.10, lw=1.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle=f"round,pad=0,rounding_size={radius}",
                                    facecolor=fill, edgecolor=edge, linewidth=lw))

    def arrow(start, end, color, thick=1.6):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                     linewidth=thick, shrinkA=0, shrinkB=0, color=color))

    def icon(kind, x, y, color, pale, scale=1):
        # Draw each pictogram in local coordinates so it scales as a unit.
        def point(u, v):
            return x + u * scale, y + v * scale

        def line(*pairs):
            px, py = zip(*(point(*p) for p in pairs))
            ax.plot(px, py, color=color, lw=1.3, solid_capstyle="round",
                    solid_joinstyle="round", zorder=3)

        def shape(pairs, fill="white"):
            ax.add_patch(Polygon([point(*p) for p in pairs], closed=True, facecolor=fill,
                                 edgecolor=color, linewidth=1.35, joinstyle="round", zorder=2))

        def rect(u, v, w, h, fill="white", r=.05):
            rounded(*point(u, v), w * scale, h * scale, fill, color, r * scale, 1.35)

        def ellipse(u, v, w, h, fill):
            ax.add_patch(Ellipse(point(u, v), w * scale, h * scale,
                                 edgecolor=color, facecolor=fill, linewidth=1.3, zorder=2))

        rounded(*point(-.49, -.46), .98 * scale, .92 * scale, pale, radius=.19 * scale)
        if kind == "question":
            shape([(-.33, -.17), (-.33, .27), (.33, .27), (.33, -.17),
                   (.01, -.17), (-.16, -.31), (-.16, -.17)])
            ax.text(x, y + .052 * scale, "?", color=color, fontsize=18 * scale,
                    fontweight="bold", ha="center", va="center")
        elif kind == "agent":
            line((0, .23), (0, .36))
            ellipse(0, .37, .078, .078, "white")
            rect(-.32, -.24, .64, .48)
            line((-.39, -.13), (-.39, .11))
            line((.39, -.13), (.39, .11))
            for u in (-.125, .125):
                ellipse(u, .06, .075, .075, color)
            line((-.09, -.095), (.09, -.095))
        elif kind in ("report", "answer"):
            shape([(-.25, -.33), (-.25, .33), (.10, .33), (.26, .17), (.26, -.33)])
            line((.10, .33), (.10, .17), (.26, .17))
            line((-.14, .07), (.13, .07))
            line((-.14, -.04), (.06, -.04))
            if kind == "answer":
                ellipse(.23, -.20, .32, .32, pale)
                line((.155, -.20), (.21, -.245), (.302, -.155))
            else:
                line((-.14, -.15), (.13, -.15))
        elif kind == "database":
            shape([(-.305, -.22), (-.305, .22), (.305, .22), (.305, -.22)],
                  fill="white")
            for v in (-.28, -.11, .055, .22):
                ellipse(0, v, .61, .20, pale if v == .22 else "white")
            line((-.305, -.28), (-.305, .22))
            line((.305, -.28), (.305, .22))
        elif kind == "calculator":
            rect(-.265, -.34, .53, .68)
            rect(-.17, .10, .34, .13, pale, .02)
            for v in (-.08, -.22):
                for u in (-.15, 0, .15):
                    ellipse(u, v, .04, .04, color)

    for x, kind, title, subtitle, color, pale in [
        (1.40, "question", "财务问题", "公司标识与英文问题", azure, "#E7F2FC"),
        (6.00, "agent", "模型决策", "结合问题与已有证据", jade, "#E2F3EC"),
        (10.60, "answer", "输出答案", "英文答案与来源编号", amber, "#FCF0DB"),
    ]:
        icon(kind, x, 3.56, color, pale, scale=1.16)
        label(x, 2.84, title, 12)
        label(x, 2.54, subtitle, 9, MUTED)

    arrow((2.18, 3.55), (5.19, 3.55), "#6694BD")
    arrow((6.82, 3.55), (9.81, 3.55), "#C69255")
    label(8.32, 3.80, "证据充分时", 9, amber)

    rounded(1.47, .07, 9.06, 1.66, "#F4FAF7", "#B9D9CC", radius=.18)
    label(1.76, 1.46, "按需调用工具", 9, jade, align="left")
    arrow((5.61, 2.32), (5.61, 1.74), jade, 1.35)
    arrow((6.39, 1.74), (6.39, 2.32), jade, 1.35)
    label(5.38, 2.04, "发出请求", 9, jade, align="right")
    label(6.62, 2.04, "返回证据", 9, jade, align="left")

    for x, kind, title, subtitle, color, pale in [
        (3.32, "database", "查财务数据", "按指定期间提取财务数值", azure, "#E7F2FC"),
        (6.00, "report", "检索年报", "从年报中定位作答依据", jade, "#E2F3EC"),
        (8.68, "calculator", "计算指标", "用原始数值算出增幅或占比", amber, "#FCF0DB"),
    ]:
        icon(kind, x, 1.02, color, pale, scale=.82)
        label(x, .48, title, 10)
        label(x, .24, subtitle, 8, MUTED)

    save(fig, out, "financeqa-workflow.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/financeqa"))
    parser.add_argument("--output", type=Path, default=Path("report/figures"))
    parser.add_argument("--workflow-only", action="store_true", help="Only regenerate the Chinese workflow diagram")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    style()
    if args.workflow_only:
        workflow(args.output)
        return
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
