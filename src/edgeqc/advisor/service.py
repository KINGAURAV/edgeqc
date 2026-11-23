"""Async advisory service so LLM latency never blocks the real-time vision loop.

Advice depends only on the defect *signature* (product, verdict, defect classes + severities); the per-frame
'Log:' line is written by code. So the service generates each signature at most once:

- cached signature        → reused at once (fresh log line for this frame); no LLM call
- signature being generated → the frame waits for that reply instead of queueing a duplicate
- new signature           → queued; a bounded queue keeps the newest events, and an event pushed out of the
                             full queue gets the rule-based advice at once (it is never left without advice)
- line idle               → the LLM pre-generates advice for likely single-defect signatures (`pregenerate`)

LLM replies are persisted (data_dir/advice_cache.json, keyed by model + prompt + knowledge base), so a restart
keeps them. A rule-based fallback (LLM failed or ungrounded) is cached only for cooldown_s, then the LLM retries.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import yaml

from ..config import resolve_path
from ..types import InspectionResult, Severity
from . import prompts
from .executorch_llama import ExecuTorchLlamaAdvisor
from .prompts import is_grounded, log_line, split_log_entry
from .rule_based import RuleBasedAdvisor

log = logging.getLogger(__name__)
AdviceCallback = Callable[[InspectionResult, str, str, float], None]  # result, text, engine, latency_s
PartialCallback = Callable[[InspectionResult, str, str], None]        # result, reply-so-far, engine


def load_kb(path: str) -> dict:
    p = resolve_path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def build_advisors(cfg: dict) -> tuple[Optional[ExecuTorchLlamaAdvisor], RuleBasedAdvisor]:
    a = cfg["advisor"]
    kb = load_kb(a["knowledge_base"])
    rule = RuleBasedAdvisor(kb)
    llm = None
    if a["engine"] == "executorch":
        try:
            llm = ExecuTorchLlamaAdvisor(resolve_path(a["runner_path"]), resolve_path(a["model_path"]),
                                         resolve_path(a["tokenizer_path"]), kb, a["max_new_tokens"],
                                         a["temperature"], a.get("cpu_threads", 4), a["timeout_s"],
                                         resolve_path(a["server_path"]) if a.get("server_path") else None,
                                         nice=int(a.get("nice", 0)), cpus=a.get("cpu_affinity"))
            log.info("Advisor: Llama via ExecuTorch (%s)", a["model_path"])
        except FileNotFoundError as e:
            if not a.get("fallback_to_rule", True):
                raise
            log.warning("ExecuTorch artefact missing (%s); using rule-based advisor", e)
    return llm, rule


def signature(result: InspectionResult) -> tuple:
    """What the advice depends on: product, verdict and, per defect class, its worst severity. Untyped
    anomaly regions (anomaly_*) are ignored when a typed defect is present: on the steel line they are the
    classical detector's extra blobs around the YOLO-typed defect. Counting boxes or keeping every severity
    gave 23 signatures over 717 NEU frames and overflowed the LLM queue; this gives 7."""
    typed = [d for d in result.defects if not d.label.startswith("anomaly_")]
    worst: dict[str, Severity] = {}
    for d in typed or result.defects:
        worst[d.label] = Severity.max(worst.get(d.label, d.severity), d.severity)
    return (result.product, result.verdict.value, tuple(sorted((k, v.value) for k, v in worst.items())))


def with_fresh_log(advice: str, result: InspectionResult) -> str:
    """Reused advice text + the log line for THIS frame (location and score differ between repeats)."""
    text, _ = split_log_entry(advice)
    return f"{text}\nLog: {log_line(result)}"


class AdvisorService:
    def __init__(self, cfg: dict, on_advice: AdviceCallback, maxsize: int = 4,
                 on_partial: Optional[PartialCallback] = None, partial_interval_s: float = 0.3):
        """on_partial: optional live display of the LLM reply while it is generated (throttled to one call
        per partial_interval_s). The final text still goes through the grounding check and on_advice; if it is
        rejected, on_advice delivers the rule-based fallback, which replaces the streamed text."""
        a = cfg["advisor"]
        self.llm, self.rule = build_advisors(cfg)
        self.cooldown = float(a["cooldown_s"])
        # Greedy decoding: a reply the grounding check rejected comes back the same on a retry for the same
        # signature (Pi screw run, 19 Nov: 7 identical retries x 42 s). Keep the rule advice for the session then;
        # only timeouts / errors (load-dependent) are retried after cooldown_s.
        self.greedy = float(a.get("temperature", 0.0) or 0.0) == 0.0
        self._last_rejected = False
        self.idle_s = float(a.get("pregenerate_idle_s", 20))
        self.on_advice = on_advice
        self.on_partial, self.partial_interval_s = on_partial, partial_interval_s
        self.q: queue.Queue[InspectionResult] = queue.Queue(maxsize=maxsize)
        self._lock = threading.Lock()
        self._cache: dict[tuple, tuple[str, str, float]] = {}   # sig → (advice, engine, expires; inf = never)
        self._waiters: dict[tuple, list[InspectionResult]] = {}  # sig queued/generating → repeats waiting for it
        self._pregen: list[InspectionResult] = []
        self._last_submit = time.monotonic()
        self._generating = False
        self.ungrounded_count = 0  # LLM replies rejected by the grounding check (report metric)
        self.stats = {"generated": 0, "reused": 0, "joined": 0, "dropped": 0, "pregenerated": 0}
        self._cache_file: Optional[Path] = None
        if self.llm is not None:
            self._cache_file = resolve_path(cfg["app"]["data_dir"]) / "advice_cache.json"
            self._version = self._cache_version(cfg)
            self._load_cache()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, name="advisor", daemon=True)
        self._thread.start()
        if self.llm is not None and hasattr(self.llm, "warm_up"):
            threading.Thread(target=self._warm_up, name="advisor-warmup", daemon=True).start()

    # ---- persistence -------------------------------------------------------------------------------------
    @staticmethod
    def _cache_version(cfg: dict) -> str:
        """Advice is reusable only for the same model, prompt, knowledge base and decoding settings."""
        a = cfg["advisor"]
        kb = resolve_path(a["knowledge_base"])
        model = resolve_path(a["model_path"])
        parts = [model.name, str(model.stat().st_size) if model.exists() else "",
                 kb.read_text() if kb.exists() else "", prompts.SYSTEM_PROMPT, prompts.EXAMPLE_USER,
                 prompts.EXAMPLE_ASSISTANT, str(a["max_new_tokens"]), str(a["temperature"])]
        return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:16]

    def _load_cache(self) -> None:
        try:
            d = json.loads(self._cache_file.read_text())
        except (OSError, ValueError):
            return
        if d.get("version") != self._version:
            log.info("Advice cache is for another model/prompt; starting empty")
            return
        for e in d.get("entries", []):
            sig = (e["sig"][0], e["sig"][1], tuple(tuple(x) for x in e["sig"][2]))
            self._cache[sig] = (e["advice"], e["engine"], float("inf"))
        log.info("Advice cache: %d signatures loaded from %s", len(self._cache), self._cache_file)

    def _save_cache(self) -> None:
        if not self._cache_file:
            return
        with self._lock:
            entries = [{"sig": list(s), "advice": a, "engine": e}
                       for s, (a, e, exp) in self._cache.items()
                       if exp == float("inf") and self.llm is not None and e == self.llm.name]
        try:
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"version": self._version, "entries": entries}, indent=1))
            os.replace(tmp, self._cache_file)
        except OSError as e:
            log.warning("Could not save advice cache: %s", e)

    # ---- public API --------------------------------------------------------------------------------------
    def _warm_up(self) -> None:
        self._generating = True
        try:
            self.llm.warm_up()  # loads the resident model while the camera starts
        except Exception as e:
            log.warning("LLM warm-up failed: %s", e)
        finally:
            self._generating = False

    @property
    def busy(self) -> bool:
        """True while the LLM is loading or generating, or advice is queued. The app lends the LLM's cores to
        vision (vision.boost_threads) only while this is False."""
        return self.llm is not None and (self._generating or not self.q.empty())

    @property
    def engine_name(self) -> str:
        return self.llm.name if self.llm else self.rule.name

    def _cached_entry(self, sig: tuple) -> Optional[tuple[str, str]]:
        e = self._cache.get(sig)
        if e is None:
            return None
        if time.monotonic() > e[2]:  # expired rule fallback → let the LLM try again
            del self._cache[sig]
            return None
        return e[0], e[1]

    def submit(self, result: InspectionResult) -> bool:
        """Ask for advice for a non-PASS result. Returns True if a new generation was queued. False means
        the advice is already cached (get it with `cached`) or is being generated for an identical defect
        signature (it will arrive through on_advice for this result too)."""
        sig = signature(result)
        with self._lock:
            self._last_submit = time.monotonic()
            if self._cached_entry(sig):
                self.stats["reused"] += 1
                return False
            if sig in self._waiters:
                self._waiters[sig].append(result)
                self.stats["joined"] += 1
                return False
            self._waiters[sig] = []
        dropped = None
        while True:
            try:
                self.q.put_nowait(result)
                break
            except queue.Full:
                try:
                    dropped = self.q.get_nowait()
                except queue.Empty:
                    continue
                self._fallback_dropped(dropped)
        return True

    def cached(self, result: InspectionResult) -> Optional[tuple[str, str]]:
        """(advice for THIS frame, engine) if the signature is cached, else None."""
        with self._lock:
            e = self._cached_entry(signature(result))
        return (with_fresh_log(e[0], result), e[1]) if e else None

    def pregenerate(self, candidates: list[InspectionResult]) -> None:
        """Generate advice for these results while the line is idle (no submissions for pregenerate_idle_s)."""
        if self.llm is None:
            return
        with self._lock:
            have = set(self._cache)
            self._pregen = [r for r in candidates if signature(r) not in have]
        if self._pregen:
            log.info("Advice pre-generation: %d signatures queued for idle time", len(self._pregen))

    # ---- worker ------------------------------------------------------------------------------------------
    def _fallback_dropped(self, result: InspectionResult) -> None:
        """An event pushed out of the full queue still gets advice: the rule-based one, now."""
        self.stats["dropped"] += 1
        self._deliver(signature(result), result, self.rule.advise(result), self.rule.name, 0.0, cache=False)

    def _partial_fn(self, result: InspectionResult) -> Optional[Callable[[str], None]]:
        if not (self.on_partial and self.llm and "on_partial" in inspect.signature(self.llm.advise).parameters):
            return None
        last = [0.0]

        def fn(text: str) -> None:
            now = time.monotonic()
            if text and now - last[0] >= self.partial_interval_s:
                last[0] = now
                self.on_partial(result, text, self.llm.name)
        return fn

    def advise_sync(self, result: InspectionResult, stream: bool = True) -> tuple[str, str, float]:
        t0 = time.perf_counter()
        self._last_rejected = False
        if self.llm:
            try:
                partial = self._partial_fn(result) if stream else None
                reply = self.llm.advise(result, on_partial=partial) if partial else self.llm.advise(result)
                if is_grounded(reply, result):
                    return reply, self.llm.name, time.perf_counter() - t0
                log.warning("LLM reply failed grounding/format check; using rule fallback: %r", reply[:200])
                self.ungrounded_count += 1
                self._last_rejected = True
            except Exception as e:  # noqa: BLE001 — never let advice crash the line
                log.warning("LLM advice failed (%s); using rule fallback", e)
        return self.rule.advise(result), self.rule.name, time.perf_counter() - t0

    def _deliver(self, sig: tuple, result: Optional[InspectionResult], text: str, engine: str, latency: float,
                 cache: bool = True) -> None:
        llm_reply = self.llm is not None and engine == self.llm.name
        keep = llm_reply or self.llm is None or (self.greedy and self._last_rejected)
        with self._lock:
            if cache:
                self._cache[sig] = (text, engine, float("inf") if keep else time.monotonic() + self.cooldown)
            waiters = self._waiters.pop(sig, [])
        if cache and llm_reply:
            self._save_cache()
        for r, t, eng, lat in ([(result, text, engine, latency)] if result is not None else []) + \
                [(w, with_fresh_log(text, w), f"{engine} (repeat)", 0.0) for w in waiters]:
            try:
                self.on_advice(r, t, eng, lat)
            except Exception:  # noqa: BLE001
                log.exception("advice callback failed")

    def _next_pregen(self) -> Optional[InspectionResult]:
        with self._lock:
            if not self._pregen or time.monotonic() - self._last_submit < self.idle_s:
                return None
            while self._pregen:
                r = self._pregen.pop(0)
                sig = signature(r)
                if sig not in self._cache and sig not in self._waiters:
                    self._waiters[sig] = []
                    return r
        return None

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                result = self.q.get(timeout=0.05 if self._pregen else 0.5)
                pre = False
            except queue.Empty:
                result, pre = self._next_pregen(), True
                if result is None:
                    continue
            self._generating = True
            try:
                text, engine, latency = self.advise_sync(result, stream=not pre)
            finally:
                self._generating = False
            if pre:
                self.stats["pregenerated"] += 1
                log.info("Pre-generated advice for %s via %s (%.1fs)", signature(result)[2], engine, latency)
                self._deliver(signature(result), None, text, engine, latency)
            else:
                self.stats["generated"] += 1
                self._deliver(signature(result), result, text, engine, latency)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def pregeneration_candidates(cfg: dict, classifier_labels: list[str]) -> list[InspectionResult]:
    """One synthetic single-defect result per (class, reachable severity), most likely first: each class at
    its base severity, then escalated ones. Verdict follows the severity rules (MINOR → REVIEW, else REJECT)."""
    from ..types import Defect, Severity, Verdict

    sev_cfg = cfg["severity"]
    base = {k: Severity(v) for k, v in sev_cfg.get("class_criticality", {}).items()}
    default = Severity(sev_cfg.get("default", "MINOR"))
    labels = list(classifier_labels) + ["anomaly_linear", "anomaly_spot", "anomaly_area"]
    area = {Severity.MINOR: 0.01, Severity.MAJOR: 0.03, Severity.CRITICAL: 0.10}
    score = {Severity.MINOR: 1.3, Severity.MAJOR: 2.4, Severity.CRITICAL: 3.8}
    order = [Severity.MINOR, Severity.MAJOR, Severity.CRITICAL]
    tiers: list[list[InspectionResult]] = [[], [], []]
    for lab in labels:
        b = base.get(lab, default)
        for i, sev in enumerate(s for s in order if s.rank >= b.rank):
            side = int((area[sev] ** 0.5) * 256)
            d = Defect(lab, (128 - side // 2, 128 - side // 2, side, side), 0.0, score[sev], area[sev],
                       "anomaly" if lab.startswith("anomaly_") else "fused", sev)
            verdict = Verdict.REVIEW if sev == Severity.MINOR else Verdict.REJECT
            tiers[min(i, 2)].append(InspectionResult(0, time.time(), cfg["app"]["product"], verdict, score[sev],
                                                     [d], image_hw=(256, 256)))
    return [r for t in tiers for r in t]
