# Llama 3.2 on ExecuTorch (advisory pipeline)

The advisory component uses **Llama 3.2 1B Instruct**, exported to an ExecuTorch `.pte` with the
XNNPACK delegate and 8-bit-dynamic-activation / 4-bit-weight (`8da4w`) quantisation. It runs on the
**Arm CPU** through ExecuTorch's Llama runner: our resident `llama_server` (preferred) or the stock
`llama_main`. This is true on the Mac (Apple Silicon, arm64) and on the target device, a Raspberry Pi 4
Model B (Cortex-A72, aarch64).

> Pinned and tested version: **ExecuTorch v1.5.1**. `scripts/setup_executorch.sh` runs every step below.
> It installs into a separate `.venv-et`, because ExecuTorch pins its own torch build.

## 1. Get the weights (access is gated; approval received 4 Nov)
```bash
hf auth login
hf download meta-llama/Llama-3.2-1B-Instruct --include "original/*" --local-dir models/llm/Llama-3.2-1B-Instruct
```
This gives you `consolidated.00.pth`, `params.json` and `tokenizer.model`.

## 2. Install ExecuTorch v1.5.1 from source
```bash
python3 -m venv .venv-et && source .venv-et/bin/activate
git clone -b v1.5.1 --depth 1 https://github.com/pytorch/executorch.git third_party/executorch
cd third_party/executorch && git submodule update --init --recursive --depth 1
./install_executorch.sh
./examples/models/llama/install_requirements.sh
```

## 3. Export to .pte (XNNPACK, 8da4w, KV cache, 1024-token context)
```bash
CK=../../models/llm/Llama-3.2-1B-Instruct/original
python -m extension.llm.export.export_llm \
  --config examples/models/llama/config/llama_xnnpack.yaml \
  ++base.model_class=llama3_2 ++base.checkpoint=$CK/consolidated.00.pth ++base.params=$CK/params.json \
  ++export.max_seq_length=1024 ++export.max_context_length=1024 \
  ++export.output_name=../../models/llm/llama3_2_1b_instruct_8da4w.pte
cp $CK/tokenizer.model ../../models/llm/tokenizer.model
```
`llama_xnnpack.yaml` sets 8da4w with group size 128, 4-bit embeddings (`4,32`), the SDPA KV cache, XNNPACK and the
Llama 3 BOS/EOS metadata. The default context is only **128 tokens**. Our prompt (findings, KB notes and one worked
example) is several hundred tokens, so both length overrides are required.

## 4. Build the runners (`llama_main` and `llama_server`)
On the machine that will run it. The `.pte` is portable across Arm64; the runner binaries are not, so build them
again on the Pi (`SKIP_EXPORT=1 scripts/setup_executorch.sh`, about 1.5 h there).
```bash
cmake --workflow llm-release
(cd examples/models/llama && cmake --workflow --preset llama-release)
```
This gives `third_party/executorch/cmake-out/examples/models/llama/llama_main` (`advisor.runner_path`).
`setup_executorch.sh` then adds two hooks to the `TextLLMRunner` header, copies
`native/llama_server/llama_server.cpp` into the same CMake project and builds `llama_server`
(`advisor.server_path`). After editing `llama_server.cpp`, rebuild only that target (about 20 s on the Pi):
```bash
cp native/llama_server/llama_server.cpp third_party/executorch/examples/models/llama/edgeqc_llama_server.cpp
(cd third_party/executorch && cmake --build cmake-out/examples/models/llama --target llama_server -j3)
```

Runner flags we use: `--max_new_tokens` (`--seq_len` is deprecated), `--temperature`, `--cpu_threads`, and
**`--num_bos=1`**. In v1.5.1 the runner does not prepend `<|begin_of_text|>` unless asked.

## 5. Verify
```bash
python scripts/llm_smoke_test.py      # prints the advisor engine, the replies, latencies and token counts
```
The engine should be `llama-executorch-resident` (`llama_server`) or `llama-executorch` (`llama_main` only). If it
prints `rule`, the `.pte`, the tokenizer or both runner binaries are missing (`advisor.model_path / tokenizer_path /
server_path / runner_path`). Add `--device configs/devices/pi4.yaml` on the Pi to use its 3-thread core split.

## Measured on the dev Mac (Apple Silicon, 8 GB, 4 CPU threads, 4 Nov 2026)
| Item | Value |
|---|---|
| `.pte` size (8da4w, 1024 context) | 803 MB (765 MiB) |
| Export time / peak RAM | about 5 min / 4.3 GB |
| Model load | 1.9 s per call (subprocess) |
| Prefill | about 169 tok/s |
| Decode | about 20.6 tok/s |
| Advice end to end (`llm_smoke_test.py`), one-shot `llama_main` | 2.8-6.2 s |
| Advice end to end, resident `llama_server` | 1.9-2.5 s |
| Runner peak RAM | about 0.85 GB |

All three smoke-test replies passed `is_grounded`.

## Measured on the Raspberry Pi 4 Model B (4 GB, Cortex-A72, 4 threads, 12 Nov 2026)
Built natively on the Pi with `SKIP_EXPORT=1 scripts/setup_executorch.sh` (about 1.5 h; the `.pte` is copied from the Mac).

| Item | Value |
|---|---|
| Prefill | about 9-11 tok/s (the A72 has no int8 dot-product instructions) |
| Decode | about 3.5 tok/s |
| Advice, resident server, full prompt (~460 tokens) | 57-82 s (prefill 41-54 s) |
| Advice, resident server **+ prefix cache** (12 Nov) | 26-38 s (time to first token 11-21 s) |
| + `Log:` line written by code, slimmer prompt (13 Nov; 3 threads on cores 1-3) | **23-30 s** (first token 11-19 s) |
| + class-notes cache, same defect class as the previous request | **18.7 s** (first token 6.5 s) |
| Same, while the vision loop runs on core 0 (real app, 16 Nov, 6 replies) | 42-68 s |
| A defect signature that already has advice (cached, also across restarts) | no LLM call |
| One-off at app start: model load + prefix prefill (warm-up thread) | 35 s |

The Pi throttles at 80-82 C during long runs without a heatsink/fan; numbers above include that.

## Prefix cache (`llama_server`)
About 320 of the ~460 prompt tokens (system prompt + worked example, `llama3_static_prefix()`) are identical in
every request. The Python client registers that text with `PREFIX`; the server prefills it once (during the
warm-up at app start), remembers the KV-cache position, and for every prompt starting with it rewinds the runner to
that position and prefills only the findings. Needs two inline hooks on `TextLLMRunner` (`edgeqc_pos()`,
`edgeqc_rewind()`), which `setup_executorch.sh` adds to the ExecuTorch header.

Verified: a reused prefix gives byte-identical replies to a freshly prefilled one (greedy, 3 cases). Replies differ
slightly from a single-batch prefill of the whole prompt (461 tokens in one call vs 320 + 141): the XNNPACK kernels
round differently for different batch shapes and greedy decoding amplifies that. Both are grounded.

## Class-notes cache (second KV-cache level, 13 Nov)
The user turn puts the defect class's reference notes before the per-frame findings
(`llama3_prompt_parts()`), and the client tells the server where that reusable head ends (`GEN ... <segment_bytes>`).
`llama_server` keeps the KV position after prefix + notes; when the next request has the same head it rewinds there
and prefills only the findings (~67 tokens). Pi 4: time to first token 11.2 s → 6.5 s, reply 23.6 s → 18.7 s for a
same-class repeat. A different class falls back to the static prefix and caches its notes instead.

## Advice cache and pre-generation (16 Nov, `advisor/service.py`)
The reply depends on the defect *signature* (product, verdict, each defect class's worst severity); the per-frame
`Log:` line is written by code. So the service calls the LLM once per signature and reuses the text, stores it in
`<data_dir>/advice_cache.json` (invalidated when the model file, prompt, knowledge base or decoding settings
change), and, when no advice has been requested for `advisor.pregenerate_idle_s` (20 s), generates the advice for
every (class, severity) in the background. On the Pi 4 that is ~30 s per signature, once. While the LLM is idle the
vision loop uses its cores (`vision.boost_threads`). Limitation: `llama_server` cannot abort a reply, so a new
defect type that appears during a pre-generation waits for it to finish.

## Pruned output vocabulary (optional)
`scripts/build_vocab_map.py` → `models/llm/token_map_24k.json` (top-24k BPE ranks + every KB/prompt token + specials),
then export with `++model.output_prune_map=<abs path>/token_map_24k.json` (absolute paths: `third_party/executorch`
is a symlink). Result `llama3_2_1b_instruct_8da4w_v24k.pte`: 693 MB (vs 803), decode 12% faster on the Pi 4,
end-to-end within noise. Use with `--set advisor.model_path=models/llm/llama3_2_1b_instruct_8da4w_v24k.pte`.

## Troubleshooting (macOS)
- `'iostream' file not found`: an incomplete `/Library/Developer/CommandLineTools/usr/include/c++` shadows the SDK
  headers. `setup_executorch.sh` sets `CPLUS_INCLUDE_PATH` to the SDK copy. The permanent fix is to reinstall the Command Line Tools.
- `no such file or directory: 'Challenge/...libregex_lookahead.a'`: the pytorch_tokenizers build breaks when the
  project path contains a space. The script clones ExecuTorch to `~/edgeqc-deps/executorch` and symlinks it into `third_party/`.
- Build killed for low memory (8 GB machines): set `CMAKE_BUILD_PARALLEL_LEVEL=2`.

## Notes for the report
- Record the `.pte` size, time-to-first-token and tokens/s printed by `llama_main` (`PyTorchObserver` line), and peak RAM.
- BOS handling: `build_llama3_prompt` omits `<|begin_of_text|>`, and we pass `--num_bos=1` instead.
- The LLM is **advisory only**. Pass/fail comes from `vision/severity.py`. The prompt includes knowledge-base notes
  (`configs/defect_kb.yaml`) to ground the reply.
- Done: the model stays resident (`llama_server`), the static prompt prefix and the class notes are KV-cached,
  replies are streamed to the dashboard, and advice is cached per defect signature (see above).
- The `Log:` line is written by code, not by the model (PLAN.md §6 explains the deviation from the slide wording).
