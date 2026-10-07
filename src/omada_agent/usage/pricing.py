"""Model prices (US$ per million tokens) used to turn token counts into money."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from omada_agent.config import Settings

# Reference list prices (input, output) per 1M tokens. They change over time: check OpenAI's pricing page
# and override with PRICE_INPUT_PER_MTOK / PRICE_OUTPUT_PER_MTOK in the .env when they differ, or when you
# use a model that is not listed here.
REFERENCE_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.5": (5.00, 30.00),
}


def reference_price(model: str) -> tuple[float, float] | None:
    """Exact match, else the longest listed name that prefixes the model (dated snapshots)."""
    if model in REFERENCE_PRICES:
        return REFERENCE_PRICES[model]
    matches = [name for name in REFERENCE_PRICES if model.startswith(name)]
    return REFERENCE_PRICES[max(matches, key=len)] if matches else None


def price_for(settings: "Settings", model: str | None = None) -> tuple[float, float] | None:
    """(input, output) price per 1M tokens: .env override first, then the reference list."""
    for low, high in (
        (settings.price_input_per_mtok, settings.price_output_per_mtok),
        (settings.eval_price_input_per_mtok, settings.eval_price_output_per_mtok),
    ):
        if low is not None and high is not None:
            return (low, high)
    return reference_price(model or settings.openai_model)


def cost_of(settings: "Settings", input_tokens: int, output_tokens: int, model: str | None = None) -> float | None:
    """Estimated US$ cost of one request, or ``None`` when the model's price is unknown."""
    price = price_for(settings, model)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
