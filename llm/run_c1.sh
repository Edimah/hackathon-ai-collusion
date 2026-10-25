#!/bin/bash
# C1 positive control: like S (3 Haiku sellers, 3 ZIP buyers, 100 rounds), but sellers 1 and 2 share a private coin
# ("do not lower your ask this hour" on heads). Seeds 0-2, in parallel. Needs ANTHROPIC_API_KEY in ../.env.
set -e
cd "$(dirname "$0")"
set -a; source ../.env; set +a
export WANDB_MODE=disabled
H=claude-haiku-5-5
for seed in 0 1 2; do
  (cd work/repo && ../venv/bin/python -m src.continuous_double_auction.simulation \
    --seller_models $H $H $H --buyer_models zip zip zip \
    --seller_valuations 80 80 80 --buyer_valuations 100 100 100 \
    --rounds 100 --seed "$seed" --couple --tag "pilot-C1-$seed" > "../log-C1-$seed.txt" 2>&1) &
done
wait
echo "C1 runs done: llm/work/repo/results/*_pilot-C1-*"
