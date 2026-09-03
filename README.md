# Kalshi-arbitrage
[README_arbitrage.md](https://github.com/user-attachments/files/31801622/README_arbitrage.md)
# Prediction Market Mispricing & Arbitrage Engine

A Python engine that detects riskless arbitrage in Kalshi-style binary event prediction markets, using simulated live order book data.

## Overview

Prediction markets price events as contracts worth $0 or $1 depending on the outcome. Basic no-arbitrage logic means certain price relationships must always hold — when they don't, there's a way to lock in a guaranteed profit regardless of the outcome. This engine simulates a live order book and continuously checks for two such mispricings:

1. **Complement arbitrage (single market)** — a YES/NO pair on the same event must always sum to $1, since exactly one side pays out. If `YES_ask + NO_ask < $1`, buying both sides locks in a guaranteed profit.

2. **Group arbitrage (mutually exclusive outcomes)** — if a set of contracts covers every possible outcome of one event (e.g. "who wins: A, B, or C?"), their prices must sum to $1. If the sum of asks is below $1, buying one contract in every outcome guarantees a $1 payout for less than $1 spent.

**Note:** this uses simulated order book data with injected mispricing shocks — there is no live Kalshi feed connected. The detection logic itself is identical to what would run against a real market data feed.

## Example output

```
============================================================
PREDICTION MARKET ARBITRAGE ENGINE — SIMULATION RESULTS
============================================================
Ticks simulated: 300
------------------------------------------------------------
COMPLEMENT ARBITRAGE (single market YES+NO)
  Opportunities found : 30 / 300 ticks
  Avg profit when found: $0.0297
  Best opportunity     : $0.0598 at tick 262
------------------------------------------------------------
GROUP ARBITRAGE (mutually exclusive outcomes A/B/C)
  Opportunities found : 95 / 300 ticks
  Avg profit when found: $0.0413
  Best opportunity     : $0.1628 at tick 229
============================================================

Example: at tick 229, outcome ask prices were:
  Candidate A    : $0.387
  Candidate B    : $0.247
  Candidate C    : $0.204
  Sum of asks: $0.837
  -> Buying one contract in each outcome costs $0.837
  -> Guaranteed payout at settlement: $1.00 (exactly one outcome wins)
  -> Riskless profit: $0.163 per unit
```

## How it works

1. **Order book simulation** — outcome prices are generated as a mean-reverting random walk around a set of "true" underlying probabilities, with occasional injected mispricing shocks to create realistic arbitrage windows.
2. **Complement check** — computes `YES_ask + NO_ask` on every tick and flags ticks where the sum drops below $1.
3. **Group check** — sums the ask prices across all mutually exclusive outcomes on every tick and flags ticks where the sum drops below $1.
4. **Profit calculation** — for every flagged tick, computes the riskless profit per contract as `$1 - sum of asks`.

## Requirements

```
numpy
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python arbitrage_engine.py
```

This prints a summary of detected arbitrage opportunities to the console and saves `kalshi_arbitrage.png` with four diagnostic plots.

## Parameters

Key parameters are set near the top of `arbitrage_engine.py`:

| Parameter | Description | Default |
|---|---|---|
| `outcomes` | Names of the mutually exclusive outcomes | 3 candidates |
| `n_ticks` | Number of simulated price updates | 300 |
| `true_probs` | "True" underlying probability of each outcome | [0.45, 0.35, 0.20] |
| `spread` | Bid-ask spread on each contract | 0.02 |
| `mispricing_rate` | Probability of an injected mispricing shock per tick | 0.08 |

## Author

Mikhail Peganov — BSc Artificial Intelligence, King's College London
