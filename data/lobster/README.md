# LOBSTER sample data

The stylized-fact comparison measures the simulated tape against a real limit
order book tape from [LOBSTER](https://lobsterdata.com/), which reconstructs
order books from NASDAQ ITCH feeds.

Populate this directory with `scripts/fetch_lobster_sample.py`. It pulls
LOBSTER's free academic sample (AAPL, 2012-06-21, top of book only) from a
Hugging Face mirror, since lobsterdata.com has no static download endpoint to
script against, and writes two files:

- `message.csv` — event-level messages, no header: time, event type, order id,
  size, price ($0.0001 units), direction.
- `orderbook.csv` — best bid and ask price and size, no header, one row per
  message and row-aligned 1:1 with `message.csv`.

`src/lob_simulator/research/lobster.py` documents the exact column semantics,
including the direction sign convention.

Both files are gitignored. LOBSTER's sample has its own redistribution terms,
so it is fetched on demand rather than committed.
