"""Per-unit prices are never stated from the index (plan §8: prices redesigned).

A pump price or a crude price changes often, so one read from a stored page can be
stale: Lime Green once quoted a sample's price for an unrelated product (its X16).
NNPC's totals (revenue, profit, remittances, a loan) are figures of a period, so they
stay answerable with their date. A per-unit price, a currency amount per litre,
barrel, MMBtu, copy and so on, is answered only from the live tier with the time it
was fetched; a claim stating one from indexed text is removed (`verify`).
"""

import re

# A currency before an amount, as NNPC writes it: ₦, N or NGN for naira; $, US$ or
# USD for dollars (measured on index version 1).
CURRENCY = r"(?:₦|NGN|US\$|USD|\$|£|GBP|€|EUR|\bN)"
AMOUNT = r"[0-9][0-9,]*(?:\.[0-9]+)?"
# An amount per a unit that is not a period of time: "₦945 per litre", "$75/bbl",
# "N617 a litre". "₦2 billion per year" or "$500,000 a day" is a total over a period.
PERIOD = r"(?:second|minute|hour|day|week|month|quarter|year|annum)s?\b"
PER_UNIT_PRICE = re.compile(
    rf"{CURRENCY}\s?{AMOUNT}(?:\s?[a-z]+)?\s?(?:per\s+|a\s+|/\s?)(?!{PERIOD})[a-z]",
    re.IGNORECASE,
)


def states_price(text: str) -> bool:
    """Whether the text states a currency amount per unit (a price)."""
    return PER_UNIT_PRICE.search(text) is not None
