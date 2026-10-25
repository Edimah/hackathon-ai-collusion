"""Qualitative check of the LLM pilot: share of seller plans that mention demand or news, and the raise rate by news value.
Keyword match, not a reading of meaning. Run from the repo root: python llm/qualitative.py"""
import collections, glob, re, sys
import numpy as np
sys.path.insert(0, "llm")
import analyse

KW = re.compile(r"\b(demand|analyst|news)\w*", re.I)
PLAN = re.compile(r'"plan_for_this_hour":\s*"((?:[^"\\]|\\.)*)"')

for c in "SD":
    plans, t = [], np.zeros((3, 2))
    for d in sorted(glob.glob(f"llm/work/repo/results/*_pilot-{c}-*")):
        for f in glob.glob(f"{d}/Seller *.md"):
            plans += PLAN.findall(open(f).read())
        news, asks = analyse.load(d)
        keep, P = analyse.binarise(asks)
        for z, row in zip(news[keep], P):
            for x in row: t[z, x] += 1
    template = collections.Counter(plans).most_common(1)[0][0]  # the prompt's output template, repeated in every prompt
    plans = [p for p in plans if p != template]
    m = sum(bool(KW.search(p)) for p in plans)
    print(f"{c}: {m}/{len(plans)} plans mention demand or news ({m / len(plans):.0%}); "
          f"P(raise | falling, steady, rising) = {(t[:, 1] / t.sum(1)).round(2)}")
