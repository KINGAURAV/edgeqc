#!/usr/bin/env python3
"""Token map for pruning Llama 3.2's output layer to the vocabulary the advisor can use (ExecuTorch
`model.output_prune_map`, docs/EXECUTORCH_LLAMA.md).

The output layer (2048 x 128,256) is ~21% of the weights read per decoded token. The advisor only ever writes short
English advice, so we keep: the first N BPE ranks (tiktoken merges are learned in frequency order, so low ids are
the common English pieces), every token of the knowledge base / prompts / rule-based replies, and all special
tokens. The pruned layer's logits are scattered back to the full vocabulary inside the model, so the runner and
tokenizer are unchanged; tokens outside the map simply can never be generated.

    .venv-et/bin/python scripts/build_vocab_map.py --keep-top 24000 --out models/llm/token_map_24k.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def corpus() -> list[str]:
    from edgeqc.advisor import prompts

    kb = yaml.safe_load((ROOT / "configs" / "defect_kb.yaml").read_text())
    texts = [prompts.SYSTEM_PROMPT, prompts.EXAMPLE_USER, prompts.EXAMPLE_ASSISTANT, yaml.safe_dump(kb)]
    for name, e in (kb.items() if isinstance(kb, dict) else []):
        texts.append(str(name).replace("_", " "))
        for v in (e or {}).values() if isinstance(e, dict) else []:
            texts.extend(v if isinstance(v, list) else [str(v)])
    # numbers, zones and verdict words the replies use
    texts += [" ".join(str(i) for i in range(101)), "top bottom middle left right center REJECT REVIEW PASS "
              "MINOR MAJOR CRITICAL OK"]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(ROOT / "models" / "llm" / "tokenizer.model"))
    ap.add_argument("--keep-top", type=int, default=24000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from pytorch_tokenizers import get_tokenizer

    tok = get_tokenizer(args.tokenizer)
    keep = set(range(args.keep_top)) | set(range(128000, 128256))       # + all special tokens
    extra = set()
    for t in corpus():
        for variant in (t, " " + t, t.lower(), t.capitalize()):
            extra.update(tok.encode(variant, bos=False, eos=False))
    keep |= extra
    ids = sorted(keep)
    Path(args.out).write_text(json.dumps({str(i): old for i, old in enumerate(ids)}))
    print(f"kept {len(ids)} of 128256 output tokens ({len(extra - set(range(args.keep_top)))} beyond the top "
          f"{args.keep_top} come from the KB/prompts) -> {args.out}")


if __name__ == "__main__":
    main()
