"""Llama 3.2 Instruct advisor running on-device through ExecuTorch (Arm CPU, XNNPACK).

Runtime: the ExecuTorch Llama runner binary (`llama_main`), built from
executorch/examples/models/llama. Export and build steps: docs/EXECUTORCH_LLAMA.md.

Two modes:
* resident (preferred): `llama_server` (native/llama_server/llama_server.cpp, same ExecuTorch runner)
  is started once and keeps the .pte loaded; prompts go over stdin. Saves the per-call model load.
* one-shot (fallback): `llama_main` is started as a subprocess for every advice call.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from ..types import InspectionResult
from .prompts import llama3_prompt_parts, llama3_static_prefix, llama3_user_turn, log_line, split_log_entry

_SPECIAL = re.compile(r"<\|[a-z_]+\|>")
log = logging.getLogger(__name__)


class LlamaServer:
    """Client for the resident llama_server process (length-prefixed protocol, see llama_server.cpp)."""

    def __init__(self, server_path: Path, model: Path, tokenizer: Path, cpu_threads: int, timeout_s: float,
                 prefix: str | None = None, nice: int = 0, cpus: list[int] | None = None):
        self.cmd = [str(server_path), f"--model_path={model}", f"--tokenizer_path={tokenizer}",
                    f"--cpu_threads={cpu_threads}", "--num_bos=1"]
        self.timeout_s = timeout_s
        self.prefix = prefix
        self.nice = nice            # >0: lower CPU priority than the real-time vision loop (POSIX)
        self.cpus = cpus            # Linux: pin to these cores (keeps XNNPACK's threads off the vision core)
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.last_stats: dict = {}

    def _start(self) -> None:
        def child_setup():
            if self.nice and hasattr(os, "nice"):
                os.nice(self.nice)
            if self.cpus and hasattr(os, "sched_setaffinity"):
                os.sched_setaffinity(0, self.cpus)
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, start_new_session=True,  # Ctrl+C stops the app,
                                     preexec_fn=child_setup if (self.nice or self.cpus) else None)  # not the server
        line = self._readline()
        if not line.startswith("READY"):
            self.close()
            raise RuntimeError(f"llama_server failed to start: {line!r}")
        log.info("llama_server resident (model load %s ms)", line.split()[1])
        if self.prefix:
            # Register the static prompt prefix; the server prefills it lazily and reuses its KV cache for
            # every prompt that starts with it. Older servers answer ERR: then every request is a full prefill.
            data = self.prefix.encode("utf-8")
            self.proc.stdin.write(f"PREFIX {len(data)}\n".encode() + data)
            self.proc.stdin.flush()
            reply = self._readline()
            if not reply.startswith("OK"):
                log.info("llama_server has no prefix cache (%s)", reply)

    def _readline(self) -> str:
        assert self.proc and self.proc.stdout
        result: list[bytes] = []
        t = threading.Thread(target=lambda: result.append(self.proc.stdout.readline()), daemon=True)
        t.start()
        t.join(self.timeout_s)
        if not result:
            self.close()
            raise TimeoutError(f"llama_server gave no reply within {self.timeout_s}s")
        return result[0].decode("utf-8", "replace").strip()

    def generate(self, prompt: str, max_new_tokens: int, temperature: float,
                 on_text: Callable[[str], None] | None = None, segment: int = 0) -> str:
        """Full reply text. With on_text, it is also called with the reply-so-far as tokens arrive.
        segment: byte length of a reusable prompt head (prefix + class notes) the server may cache."""
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                self._start()
            data = prompt.encode("utf-8")
            stream = 1 if on_text else 0
            self.proc.stdin.write(f"GEN {len(data)} {max_new_tokens} {temperature} {stream} {segment}\n".encode()
                                  + data)
            self.proc.stdin.flush()
            head = self._readline()
            so_far = b""
            while head.startswith("TOK "):
                so_far += self.proc.stdout.read(int(head.split()[1]))
                try:
                    on_text(so_far.decode("utf-8", "ignore"))  # "ignore": a piece may end mid-character
                except Exception:  # noqa: BLE001 — a display callback must never break generation
                    log.exception("on_text callback failed")
                head = self._readline()
            if not head.startswith("OK"):
                raise RuntimeError(f"llama_server: {head}")
            _, n, n_prompt, n_gen, ttft, total, *rest = head.split()
            self.last_stats = {"prompt_tokens": int(n_prompt), "generated_tokens": int(n_gen),
                               "ttft_ms": int(ttft), "total_ms": int(total),
                               "cached_prefix_tokens": int(rest[0]) if rest else 0}
            return self.proc.stdout.read(int(n)).decode("utf-8", "replace")

    def close(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(b"QUIT\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()
        self.proc = None


def parse_runner_output(stdout: str, prompt: str) -> str:
    """Extract the assistant reply from llama_main stdout (it echoes the prompt first)."""
    text = stdout
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    elif prompt in text:
        text = text.split(prompt, 1)[1]
    text = text.split("<|eot_id|>", 1)[0]
    text = _SPECIAL.sub("", text)
    lines = [ln for ln in text.splitlines() if not ln.startswith(("PyTorchObserver", "I 0", "E 0", "W 0"))]
    return "\n".join(ln.rstrip() for ln in lines).strip()


class ExecuTorchLlamaAdvisor:
    name = "llama-executorch"

    def __init__(self, runner_path: str | Path, model_path: str | Path, tokenizer_path: str | Path,
                 kb: dict, max_new_tokens: int = 140, temperature: float = 0.2, cpu_threads: int = 4,
                 timeout_s: float = 90, server_path: str | Path | None = None, nice: int = 0,
                 cpus: list[int] | None = None):
        self.runner, self.model, self.tokenizer = Path(runner_path), Path(model_path), Path(tokenizer_path)
        for p in (self.model, self.tokenizer):
            if not p.exists():
                raise FileNotFoundError(p)
        self.kb, self.max_new_tokens, self.temperature = kb, max_new_tokens, temperature
        self.cpu_threads, self.timeout_s = cpu_threads, timeout_s
        self.last_latency_s = 0.0
        self.server = None
        if server_path and Path(server_path).exists():
            self.server = LlamaServer(Path(server_path), self.model, self.tokenizer, cpu_threads, timeout_s,
                                      prefix=llama3_static_prefix(), nice=nice, cpus=cpus)
            self.name = "llama-executorch-resident"
        elif not self.runner.exists():
            raise FileNotFoundError(self.runner)

    def warm_up(self) -> None:
        """Start the resident server now and prefill the static prompt prefix, so the first advice event
        pays neither the model load nor the prefix prefill (~40 s on a Pi 4)."""
        if self.server:
            self.server.generate(llama3_static_prefix() + llama3_user_turn("OK"), 2, 0.0)

    def generate(self, prompt: str, on_partial: Callable[[str], None] | None = None, segment: int = 0) -> str:
        if self.server:
            t0 = time.perf_counter()
            on_text = (lambda raw: on_partial(parse_runner_output(raw, prompt))) if on_partial else None
            try:
                text = self.server.generate(prompt, self.max_new_tokens, self.temperature, on_text, segment)
                self.last_latency_s = time.perf_counter() - t0
                return parse_runner_output(text, prompt)
            except Exception as e:  # fall through to the one-shot runner if it exists
                log.warning("llama_server failed (%s)%s", e, "; using llama_main" if self.runner.exists() else "")
                if not self.runner.exists():
                    raise
        return self._generate_oneshot(prompt)

    def _generate_oneshot(self, prompt: str) -> str:
        cmd = [str(self.runner), f"--model_path={self.model}", f"--tokenizer_path={self.tokenizer}",
               f"--prompt={prompt}", f"--max_new_tokens={self.max_new_tokens}",
               f"--temperature={self.temperature}", f"--cpu_threads={self.cpu_threads}",
               "--num_bos=1"]  # runner adds <|begin_of_text|> only when asked (ExecuTorch >= 1.x)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s)
        self.last_latency_s = time.perf_counter() - t0
        if proc.returncode != 0:
            # Skip the tokenizer's harmless "Error parsing json file" probe; surface real error lines.
            errs = [ln for ln in (proc.stdout + proc.stderr).splitlines()
                    if ln.startswith(("E ", "E0", "F ")) and "parsing json" not in ln]
            raise RuntimeError(f"llama_main exited {proc.returncode}: {' | '.join(errs[-3:]) or proc.stderr[-300:]}")
        return parse_runner_output(proc.stdout, prompt)

    def advise(self, result: InspectionResult, on_partial: Callable[[str], None] | None = None) -> str:
        """on_partial (resident server only) receives the cleaned reply-so-far while it is generated."""
        head, tail = llama3_prompt_parts(result, self.kb)
        reply = self.generate(head + tail, on_partial, segment=len(head.encode("utf-8")))
        reply, _ = split_log_entry(reply)  # drop a 'Log:' line if the model wrote one anyway
        if not reply:
            raise RuntimeError("empty LLM reply")
        return f"{reply}\nLog: {log_line(result)}"
