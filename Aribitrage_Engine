"""
Prediction Market Mispricing & Arbitrage Engine
------------------------------------------------
Detects riskless arbitrage in Kalshi-style binary event markets.

Two arbitrage rules are checked continuously as prices update:

1. Complement arbitrage (single market):
   A YES/NO pair on the same event must sum to $1, since exactly
   one side must pay out. If YES_ask + NO_ask < $1, buying both
   sides locks in a guaranteed profit at settlement.

2. Group arbitrage (mutually exclusive outcomes):
   If a set of markets covers every possible outcome of one event
   (e.g. "which of A/B/C wins"), their YES prices must sum to $1.
   If the sum of asks < $1, buying one contract in every outcome
   guarantees a $1 payout for less than $1 spent.

NOTE: This uses simulated order book data (no live Kalshi feed is
connected). The detection logic is identical to what you'd run
against a real market data feed.

Author: Mikhail Peganov
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

# ----------------------------
# 1. Simulate a market: 3 mutually exclusive outcomes of one event
#    e.g. "Who wins the election: Candidate A / B / C?"
# ----------------------------
outcomes = ["Candidate A", "Candidate B", "Candidate C"]
n_ticks = 300          # number of price updates ("time steps") to simulate
true_probs = np.array([0.45, 0.35, 0.20])   # the "real" underlying probabilities

spread = 0.02          # typical bid-ask spread on each contract


def simulate_order_book(true_probs, n_ticks, spread, noise=0.03, mispricing_rate=0.08):
    """
    Simulates YES bid/ask prices for each outcome over time.
    Occasionally injects a mispricing event where the market
    temporarily drifts away from a consistent probability sum.
    """
    n_outcomes = len(true_probs)
    mid_prices = np.zeros((n_ticks, n_outcomes))

    current = true_probs.copy()
    for t in range(n_ticks):
        # random walk around the true probability, with mean reversion
        current += np.random.normal(0, noise, n_outcomes) * 0.3
        current += (true_probs - current) * 0.1   # pull back toward true value
        current = np.clip(current, 0.01, 0.99)

        # occasionally inject a mispricing shock on a random outcome
        if np.random.rand() < mispricing_rate:
            shock_idx = np.random.randint(n_outcomes)
            current[shock_idx] += np.random.choice([-1, 1]) * np.random.uniform(0.04, 0.09)
            current = np.clip(current, 0.01, 0.99)

        mid_prices[t] = current

    bids = np.clip(mid_prices - spread / 2, 0.01, 0.99)
    asks = np.clip(mid_prices + spread / 2, 0.01, 0.99)
    return bids, asks, mid_prices


bids, asks, mids = simulate_order_book(true_probs, n_ticks, spread)


# ----------------------------
# 2. Complement arbitrage check (single-market YES/NO)
#    Simulated separately: NO_ask = 1 - YES_bid normally, with
#    occasional slippage that breaks the $1 rule.
# ----------------------------
def simulate_complement_pair(n_ticks, base_yes=0.55, noise=0.03, mispricing_rate=0.08):
    yes_mid = np.clip(base_yes + np.cumsum(np.random.normal(0, noise, n_ticks) * 0.05), 0.05, 0.95)
    no_mid = 1 - yes_mid

    # add independent noise to NO side to simulate a separate order book
    no_mid = np.clip(no_mid + np.random.normal(0, 0.01, n_ticks), 0.05, 0.95)

    # inject occasional real mispricings
    mispriced = np.random.rand(n_ticks) < mispricing_rate
    no_mid[mispriced] -= np.random.uniform(0.03, 0.08, mispriced.sum())
    no_mid = np.clip(no_mid, 0.02, 0.98)

    yes_ask = np.clip(yes_mid + spread / 2, 0.02, 0.98)
    no_ask = np.clip(no_mid + spread / 2, 0.02, 0.98)
    return yes_ask, no_ask


yes_ask, no_ask = simulate_complement_pair(n_ticks)
complement_sum = yes_ask + no_ask
complement_arb = complement_sum < 1.0
complement_profit = np.clip(1.0 - complement_sum, 0, None)


# ----------------------------
# 3. Group arbitrage check (mutually exclusive outcomes)
# ----------------------------
group_ask_sum = asks.sum(axis=1)
group_arb = group_ask_sum < 1.0
group_profit = np.clip(1.0 - group_ask_sum, 0, None)


# ----------------------------
# 4. Results summary
# ----------------------------
print("=" * 60)
print("PREDICTION MARKET ARBITRAGE ENGINE — SIMULATION RESULTS")
print("=" * 60)
print(f"Ticks simulated: {n_ticks}")
print("-" * 60)
print("COMPLEMENT ARBITRAGE (single market YES+NO)")
print(f"  Opportunities found : {complement_arb.sum()} / {n_ticks} ticks")
print(f"  Avg profit when found: ${complement_profit[complement_arb].mean():.4f}" if complement_arb.any() else "  None found")
print(f"  Best opportunity     : ${complement_profit.max():.4f} at tick {complement_profit.argmax()}")
print("-" * 60)
print("GROUP ARBITRAGE (mutually exclusive outcomes A/B/C)")
print(f"  Opportunities found : {group_arb.sum()} / {n_ticks} ticks")
print(f"  Avg profit when found: ${group_profit[group_arb].mean():.4f}" if group_arb.any() else "  None found")
print(f"  Best opportunity     : ${group_profit.max():.4f} at tick {group_profit.argmax()}")
print("=" * 60)

best_tick = group_profit.argmax()
print(f"\nExample: at tick {best_tick}, outcome ask prices were:")
for name, price in zip(outcomes, asks[best_tick]):
    print(f"  {name:15}: ${price:.3f}")
print(f"  Sum of asks: ${asks[best_tick].sum():.3f}")
print(f"  -> Buying one contract in each outcome costs ${asks[best_tick].sum():.3f}")
print(f"  -> Guaranteed payout at settlement: $1.00 (exactly one outcome wins)")
print(f"  -> Riskless profit: ${1 - asks[best_tick].sum():.3f} per unit")


# ----------------------------
# 5. Visualisation
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# (a) Individual outcome prices over time
for i, name in enumerate(outcomes):
    axes[0, 0].plot(mids[:, i], label=name, linewidth=1)
axes[0, 0].set_title("Simulated Outcome Prices Over Time (mid price)")
axes[0, 0].set_xlabel("Tick")
axes[0, 0].set_ylabel("Implied probability ($)")
axes[0, 0].legend()

# (b) Sum of group ask prices vs the $1 no-arbitrage line
axes[0, 1].plot(group_ask_sum, color="steelblue", label="Sum of outcome ask prices")
axes[0, 1].axhline(1.0, color="black", linestyle="--", label="No-arbitrage line ($1)")
axes[0, 1].fill_between(range(n_ticks), group_ask_sum, 1.0,
                         where=group_arb, color="green", alpha=0.4, label="Arbitrage window")
axes[0, 1].set_title("Group Arbitrage Detection (A+B+C ask prices)")
axes[0, 1].set_xlabel("Tick")
axes[0, 1].set_ylabel("Price sum ($)")
axes[0, 1].legend()

# (c) Complement pair sum vs $1 line
axes[1, 0].plot(complement_sum, color="darkorange", label="YES ask + NO ask")
axes[1, 0].axhline(1.0, color="black", linestyle="--", label="No-arbitrage line ($1)")
axes[1, 0].fill_between(range(n_ticks), complement_sum, 1.0,
                         where=complement_arb, color="green", alpha=0.4, label="Arbitrage window")
axes[1, 0].set_title("Complement Arbitrage Detection (single market)")
axes[1, 0].set_xlabel("Tick")
axes[1, 0].set_ylabel("Price sum ($)")
axes[1, 0].legend()

# (d) Profit per unit when arbitrage exists, over time
axes[1, 1].plot(group_profit, label="Group arb profit/unit", color="green")
axes[1, 1].plot(complement_profit, label="Complement arb profit/unit", color="darkorange")
axes[1, 1].set_title("Riskless Profit per Contract When Arbitrage Exists")
axes[1, 1].set_xlabel("Tick")
axes[1, 1].set_ylabel("Profit ($ per unit)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("kalshi_arbitrage.png", dpi=130)
print("\nChart saved to kalshi_arbitrage.png")
