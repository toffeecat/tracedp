"""Rendering-only views for the controlled PhasePool mechanism sweep."""

from __future__ import annotations

from collections.abc import Mapping
from statistics import mean
from typing import Any

_METHODS = {
    "coarse": "g2po_tn_loto",
    "hard": "budget_refined_tn_loto_hard_split",
    "phasepool": "phasepool_tn_loto",
}


def render_factorial_svg(receipt: Mapping[str, Any]) -> str:
    """Render the main mechanism map and every registered marginal factor."""

    grouped: dict[tuple[float, float], dict[str, list[float]]] = {}
    for cell in receipt["cells"]:
        key = (float(cell["aliasing_probability"]), float(cell["effect_size_delta"]))
        bucket = grouped.setdefault(key, {name: [] for name in _METHODS})
        for name, estimator in _METHODS.items():
            bucket[name].append(float(cell["estimator_ladder"][estimator]["success_brier"]["mean"]))
    aliases = sorted({key[0] for key in grouped})
    effects = sorted({key[1] for key in grouped})
    width, height = 1120, 700
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="28" y="34" font-family="sans-serif" font-size="20" font-weight="700">PhasePool mechanism map: gains and honest regret</text>',
        '<text x="28" y="58" font-family="sans-serif" font-size="12" fill="#475569">Blue is better. Red is worse. Each cell macros over group size, balance, horizon, revisit, and repetitions.</text>',
        '<text x="28" y="92" font-family="sans-serif" font-size="15" font-weight="600">PhasePool − coarse success Brier</text>',
        '<text x="510" y="92" font-family="sans-serif" font-size="15" font-weight="600">PhasePool regret to min(coarse, hard)</text>',
    ]
    for panel_index, metric in enumerate(("gain", "regret")):
        panel_left = 28 + panel_index * 482
        cell_width = 410 / max(1, len(effects))
        cell_height = 270 / max(1, len(aliases))
        for row_index, aliasing in enumerate(reversed(aliases)):
            for column_index, effect in enumerate(effects):
                bucket = grouped[(aliasing, effect)]
                coarse = mean(bucket["coarse"])
                hard = mean(bucket["hard"])
                phasepool = mean(bucket["phasepool"])
                value = phasepool - coarse if metric == "gain" else phasepool - min(coarse, hard)
                x = panel_left + 58 + column_index * cell_width
                y = 112 + row_index * cell_height
                parts.extend(
                    [
                        f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width - 4:.1f}" height="{cell_height - 4:.1f}" rx="5" fill="{_delta_color(value)}" stroke="#cbd5e1"/>',
                        f'<text x="{x + (cell_width - 4) / 2:.1f}" y="{y + (cell_height - 4) / 2 + 4:.1f}" text-anchor="middle" font-family="monospace" font-size="13">{value:+.3f}</text>',
                    ]
                )
            parts.append(
                f'<text x="{panel_left + 50}" y="{112 + row_index * cell_height + cell_height / 2:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">C={aliasing:g}</text>'
            )
        for column_index, effect in enumerate(effects):
            x = panel_left + 58 + column_index * cell_width + (cell_width - 4) / 2
            parts.append(
                f'<text x="{x:.1f}" y="404" text-anchor="middle" font-family="sans-serif" font-size="11">Δp={effect:g}</text>'
            )
    parts.extend(
        [
            '<circle cx="30" cy="443" r="5" fill="#bfdbfe"/><text x="42" y="447" font-family="sans-serif" font-size="11">PhasePool improves</text>',
            '<circle cx="175" cy="443" r="5" fill="#fecaca"/><text x="187" y="447" font-family="sans-serif" font-size="11">PhasePool degrades</text>',
            '<text x="510" y="443" font-family="sans-serif" font-size="11" fill="#475569">Positive right-panel values mean PhasePool does not dominate the coarse/hard lower envelope.</text>',
            '<text x="28" y="490" font-family="sans-serif" font-size="15" font-weight="600">Marginal factor ablation</text>',
            '<text x="250" y="490" font-family="sans-serif" font-size="11" fill="#475569">Each value averages all remaining factors; Δc = PhasePool − coarse, R = regret to best endpoint.</text>',
        ]
    )
    factor_specs = (
        ("group_size", "Group size"),
        ("phase_balance", "Phase balance"),
        ("horizon", "Horizon"),
        ("revisit_probability", "Revisit"),
    )
    for factor_index, (field, label) in enumerate(factor_specs):
        left = 28 + factor_index * 272
        parts.append(
            f'<text x="{left}" y="525" font-family="sans-serif" font-size="13" font-weight="600">{label}</text>'
        )
        levels = sorted({float(cell[field]) for cell in receipt["cells"]})
        for level_index, level in enumerate(levels):
            cells = [cell for cell in receipt["cells"] if float(cell[field]) == level]
            coarse, hard, phasepool = _endpoint_means(cells)
            delta = phasepool - coarse
            regret = phasepool - min(coarse, hard)
            y = 552 + level_index * 54
            parts.extend(
                [
                    f'<rect x="{left}" y="{y - 18}" width="250" height="40" rx="5" fill="{_delta_color(delta)}" stroke="#cbd5e1"/>',
                    f'<text x="{left + 10}" y="{y + 5}" font-family="monospace" font-size="12">{level:g}: Δc={delta:+.3f}  R={regret:+.3f}</text>',
                ]
            )
    parts.extend(
        [
            '<text x="28" y="680" font-family="sans-serif" font-size="11" fill="#475569">Synthetic estimator mechanism only; no model, environment, GPU, gradient, or policy update.</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def _endpoint_means(cells: list[Mapping[str, Any]]) -> tuple[float, float, float]:
    return tuple(
        mean(float(cell["estimator_ladder"][estimator]["success_brier"]["mean"]) for cell in cells)
        for estimator in (
            _METHODS["coarse"],
            _METHODS["hard"],
            _METHODS["phasepool"],
        )
    )


def _delta_color(value: float) -> str:
    magnitude = min(1.0, abs(value) / 0.08)
    if value < 0:
        red = int(238 - 70 * magnitude)
        green = int(248 - 40 * magnitude)
        blue = 255
    else:
        red = 255
        green = int(247 - 90 * magnitude)
        blue = int(237 - 80 * magnitude)
    return f"rgb({red},{green},{blue})"


__all__ = ["render_factorial_svg"]
