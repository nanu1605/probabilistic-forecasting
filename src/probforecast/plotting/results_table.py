"""Render the full model-comparison results table as markdown + LaTeX."""

from __future__ import annotations

from pathlib import Path

COLUMNS = ["Model", "MAE", "RMSE", "CRPS", "Cov@50", "Cov@80", "Cov@90", "ECE"]


def _fmt(v) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "—"
    return f"{v:.3f}" if abs(v) < 10 else f"{v:.2f}"


def render_markdown(rows: list[dict]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c)) for c in COLUMNS) + " |")
    return "\n".join(lines)


def render_latex(rows: list[dict]) -> str:
    spec = "l" + "r" * (len(COLUMNS) - 1)
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    out.append(" & ".join(COLUMNS) + " \\\\")
    out.append("\\midrule")
    for r in rows:
        out.append(" & ".join(_fmt(r.get(c)) for c in COLUMNS) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def write_results_table(rows: list[dict], md_path: str | Path) -> Path:
    md_path = Path(md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md = render_markdown(rows)
    latex = render_latex(rows)
    md_path.write_text("# Full results\n\n" + md + "\n\n## LaTeX\n\n```latex\n" + latex + "\n```\n")
    return md_path


__all__ = ["render_markdown", "render_latex", "write_results_table", "COLUMNS"]
