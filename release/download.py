"""Fetch the four public checkpoints the bridge needs into the HF cache. ~7 GB, ungated.

Run once, then the demo works offline.
"""
import bridge
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

for repo in (bridge.RECEIVER, bridge.SHARER, bridge.FUSER_REPO):
    rev = bridge.REVISIONS[repo]
    print(f"fetching {repo}@{rev} ...", flush=True)
    snapshot_download(repo_id=repo, revision=rev, token=False)

AutoTokenizer.from_pretrained(bridge.RECEIVER)
p = bridge.pair_paths()
print("\nready:")
for k, v in p.items():
    print(f"  {k:8} {v}")
