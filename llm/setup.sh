#!/bin/bash
# Clone the Agrawal et al. double auction at a fixed commit, apply our pilot patch, build its venv in llm/work/.
# The upstream repo has no licence, so its code is not copied here; only our patch is.
set -e
cd "$(dirname "$0")"
mkdir -p work
[ -d work/repo ] || git clone -q https://github.com/andyjliu/llm-agent-collusion work/repo
git -C work/repo checkout -q 2bb2a7e
git -C work/repo apply ../../pilot.patch 2>/dev/null || git -C work/repo apply --check -R ../../pilot.patch  # already applied is fine
python3 -m venv work/venv
work/venv/bin/pip install -q -r requirements.txt anthropic openai
echo "setup done"
