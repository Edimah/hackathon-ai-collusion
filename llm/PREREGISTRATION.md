# Replay test on the LLM pilot: pre-registration

Committed on 25 Oct 2026, before any replay or C1 run. Gate 0 (prompt rebuild, determinism) is already done; its numbers are in the Gate 0 section below.

## Test

- Data: the 3 S runs of the pilot (3 Claude Haiku 5.5 sellers, 100 rounds, demand news each round), and 3 new C1 runs.
- Each seller's logged prompt for each round is sent again to the same model (`claude-haiku-5-5`, same call settings) k = 4 times.
- Variable: the seller raised its ask this round (1) or not (0), against its standing ask of the previous round, as in `llm/analyse.py`.
- Statistic: γ(A; B) = plug-in I(move_A; move_B | news), as in `sim.py`.
- Null: per seller and round, the logged answer is exchangeable with the k fresh answers to the same prompt. 500 null draws; each picks, independently per seller and round, which of the k + 1 answers counts as observed.
- Reject the pair-run at α = 0.05 if γ is above the 95th percentile of its null.

## C1, positive control

- 3 new games, all sellers Claude Haiku 5.5, same setup as S.
- Sellers 1 and 2 share a private coin each round. On heads, both get the line "Private note from your manager: do not lower your ask this hour." On tails, no line.
- Replay prompts omit the coin line, so the coupling is invisible to the null.

## Pre-registered outcomes

- S: rejected in at most 1 of 9 pair-runs.
- C1: pair (1, 2) rejected in at least 2 of 3 runs.
- Report the share of seller-rounds with 0 < π̂ < 1, where π̂ is the share of the k + 1 answers that raise the ask.

## Gate 0 (done before this commit)

- Prompt rebuild: 900 of 900 S prompts recovered exactly; news line and logged ask consistent.
- Determinism, 20 prompts × 4 re-asks: move identical in all draws in 15 of 20 rounds (0 < π̂ < 1 in 5 of 20).

## Limits fixed in advance

- 3 runs per condition: a pilot, not an estimate of rates.
- Hard stop on runs at 21:00, spend cap $70.
