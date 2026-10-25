#!/bin/bash
# Block B pilot: 3 LLM sellers, 3 scripted ZIP buyers, 100 rounds, chat off, one public news line per round.
# S = all sellers Claude Haiku 5.5; D = Haiku, Sonnet, Haiku. Seeds 0-2. The 6 runs go in parallel.
# Needs ANTHROPIC_API_KEY in ../.env (gitignored).
set -e
cd "$(dirname "$0")"
set -a; source ../.env; set +a
export WANDB_MODE=disabled
H=claude-haiku-5-5; S=claude-sonnet-5-5
run() {  # condition seed seller models...
  local cond=$1 seed=$2; shift 2
  (cd work/repo && ../venv/bin/python -m src.continuous_double_auction.simulation \
    --seller_models "$@" --buyer_models zip zip zip \
    --seller_valuations 80 80 80 --buyer_valuations 100 100 100 \
    --rounds 100 --seed "$seed" --tag "pilot-$cond-$seed" > "../log-$cond-$seed.txt" 2>&1)
}
for seed in 0 1 2; do
  run S $seed $H $H $H &
  run D $seed $H $S $H &
done
wait
echo "runs done: llm/work/repo/results/*_pilot-*; next: python llm/analyse.py"
