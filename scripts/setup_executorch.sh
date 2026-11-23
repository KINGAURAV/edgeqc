#!/usr/bin/env bash
# Build ExecuTorch + llama_main and export Llama 3.2 1B Instruct to .pte (XNNPACK, 8da4w, KV cache).
# Usage: scripts/setup_executorch.sh [executorch-release-tag]   (default: v1.5.1, the version we tested)
# Prereq: weights in models/llm/Llama-3.2-1B-Instruct/original (see docs/EXECUTORCH_LLAMA.md)
#
# On the Raspberry Pi you only need the BUILD step (copy the .pte + tokenizer from the Mac):
#   SKIP_EXPORT=1 scripts/setup_executorch.sh
set -euo pipefail

TAG="${1:-v1.5.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ET="$ROOT/third_party/executorch"
CK="$ROOT/models/llm/Llama-3.2-1B-Instruct/original"
OUT="$ROOT/models/llm"
VENV="$ROOT/.venv-et"            # separate env: ExecuTorch pins its own torch

if [[ ! -d "$VENV" ]]; then python3 -m venv "$VENV"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# macOS: a partial Command Line Tools update can leave an incomplete
# CommandLineTools/usr/include/c++/v1 that shadows the SDK's libc++ headers
# ("'iostream' file not found"). Point clang at the SDK copy if so.
if [[ "$(uname)" == "Darwin" ]] && ! [[ -e /Library/Developer/CommandLineTools/usr/include/c++/v1/iostream ]]; then
  export CPLUS_INCLUDE_PATH="$(xcrun --show-sdk-path)/usr/include/c++/v1${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
fi

if [[ ! -e "$ET" ]]; then
  if [[ "$ROOT" == *" "* ]]; then
    # The pytorch_tokenizers build doesn't quote its link paths, so a space in the project path
    # breaks it. Clone to a space-free folder and symlink it into third_party/.
    REAL="${ET_HOME:-$HOME/edgeqc-deps}/executorch"
    mkdir -p "$(dirname "$REAL")" "$ROOT/third_party"
    [[ -d "$REAL" ]] || git clone -b "$TAG" --depth 1 https://github.com/pytorch/executorch.git "$REAL"
    ln -s "$REAL" "$ET"
  else
    git clone -b "$TAG" --depth 1 https://github.com/pytorch/executorch.git "$ET"
  fi
fi
cd "$(cd "$ET" && pwd -P)"     # build from the real (space-free) path
git submodule update --init --recursive --depth 1
if [[ "${SKIP_EXPORT:-0}" == "1" ]]; then
  # Runner-only build (Raspberry Pi): skip the ExecuTorch Python package (hours of compiling on a Pi 4).
  # The CMake kernel codegen only needs torchgen (ships inside torch; CPU wheel, same version as the
  # export env) and the YAML parsers.
  python -c "import torch, yaml, ruamel.yaml" 2>/dev/null || {
    python -m pip install -q --upgrade pip
    python -m pip install -q "torch==${ET_TORCH:-2.14.0}" --index-url https://download.pytorch.org/whl/cpu
    python -m pip install -q pyyaml ruamel.yaml
  }
  export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-3}"
else
  python -c "import executorch" 2>/dev/null || ./install_executorch.sh
  ./examples/models/llama/install_requirements.sh
fi

if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  [[ -f "$CK/consolidated.00.pth" ]] || { echo "Missing $CK/consolidated.00.pth: download the weights first"; exit 1; }
  echo ">> Exporting .pte"
  python -m extension.llm.export.export_llm \
    --config examples/models/llama/config/llama_xnnpack.yaml \
    ++base.model_class="llama3_2" \
    ++base.checkpoint="$CK/consolidated.00.pth" \
    ++base.params="$CK/params.json" \
    ++export.max_seq_length=1024 \
    ++export.max_context_length=1024 \
    ++export.output_name="$OUT/llama3_2_1b_instruct_8da4w.pte"
  cp "$CK/tokenizer.model" "$OUT/tokenizer.model"
fi

echo ">> Building ExecuTorch (llm-release) and llama_main"
cmake --workflow llm-release
( cd examples/models/llama && cmake --workflow --preset llama-release )

echo ">> Building llama_server (resident model, native/llama_server/llama_server.cpp)"
# Two public hooks on TextLLMRunner so llama_server can reuse the KV cache of the static prompt prefix
# (read the KV position; rewind to it). Inline in the header: the runner library itself is unchanged.
RH=extension/llm/runner/text_llm_runner.h
if ! grep -q edgeqc_rewind "$RH"; then
  python3 - "$RH" <<'PY'
import sys
p = sys.argv[1]; t = open(p).read()
anchor = "  void reset() override;\n"
assert anchor in t, "TextLLMRunner::reset() declaration not found; update the EdgeQC hook patch"
hook = ("  // --- EdgeQC hooks (prompt-prefix KV-cache reuse in llama_server; added by setup_executorch.sh) ---\n"
        "  int64_t edgeqc_pos() const { return pos_; }\n"
        "  void edgeqc_rewind(int64_t pos) { stats_->reset(); pos_ = pos; prefill_next_token_.reset(); }\n")
open(p, "w").write(t.replace(anchor, anchor + hook, 1))
PY
fi
cp "$ROOT/native/llama_server/llama_server.cpp" examples/models/llama/edgeqc_llama_server.cpp
if ! grep -q edgeqc_llama_server examples/models/llama/CMakeLists.txt; then
  cat >> examples/models/llama/CMakeLists.txt <<'CMK'

# --- EdgeQC: resident Llama server (added by EdgeQC scripts/setup_executorch.sh) ---
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/edgeqc_llama_server.cpp)
  add_executable(llama_server edgeqc_llama_server.cpp)
  target_include_directories(llama_server PUBLIC ${_common_include_directories})
  target_link_libraries(llama_server PUBLIC llama_runner ${link_libraries})
  target_compile_options(llama_server PUBLIC ${_common_compile_options})
  if(APPLE)
    target_link_options(llama_server PRIVATE -Wl,-rpath,@loader_path)
  else()
    set_target_properties(llama_server PROPERTIES LINK_FLAGS "-Wl,-rpath='$ORIGIN'")
  endif()
endif()
CMK
fi
( cd examples/models/llama && cmake --preset llama-release >/dev/null )
cmake --build cmake-out/examples/models/llama --target llama_server

ls -lh "$OUT"/*.pte cmake-out/examples/models/llama/llama_main cmake-out/examples/models/llama/llama_server 2>/dev/null || true
echo ">> Done. Verify with: python scripts/llm_smoke_test.py"
