"""Advice reuse / pre-generation, the output path and storage counters (runtime optimisations)."""
import threading
import time

import numpy as np

from edgeqc.advisor import service as svc_mod
from edgeqc.advisor.rule_based import RuleBasedAdvisor
from edgeqc.advisor.service import AdvisorService, load_kb, pregeneration_candidates, signature
from edgeqc.storage import InspectionLog
from edgeqc.types import Defect, InspectionResult, Severity, Verdict
from edgeqc.vision.overlay import render


def _result(label="scratches", sev=Severity.MAJOR, x=1):
    return InspectionResult(1, time.time(), "steel_strip", Verdict.REJECT, 2.4,
                            [Defect(label, (x, 2, 30, 4), 0.8, 2.4, 0.03, "fused", sev)], image_hw=(100, 100))


class SlowLLM:
    name = "llama-executorch"

    def __init__(self, delay=0.3):
        self.delay, self.calls = delay, 0

    def advise(self, result):
        self.calls += 1
        time.sleep(self.delay)
        d = result.defects[0]
        return f"Cause: {d.label} from worn rolls.\nAction: inspect rolls.\nCheck: next coil."


def _service(cfg, monkeypatch, llm, cb, **kw):
    kb = load_kb(cfg["advisor"]["knowledge_base"])
    monkeypatch.setattr(svc_mod, "build_advisors", lambda c: (llm, RuleBasedAdvisor(kb)))
    return AdvisorService(cfg, cb, **kw)


def test_same_signature_generated_once_and_repeats_get_fresh_log(cfg, monkeypatch):
    got, lock = [], threading.Lock()

    def cb(r, text, engine, lat):
        with lock:
            got.append((r, text, engine))

    llm = SlowLLM()
    svc = _service(cfg, monkeypatch, llm, cb)
    first, repeat = _result(x=1), _result(x=80)        # same signature, different location
    assert svc.submit(first) is True
    assert svc.submit(repeat) is False                 # joins the in-flight generation
    for _ in range(50):
        if len(got) == 2:
            break
        time.sleep(0.05)
    assert llm.calls == 1 and len(got) == 2
    rep = next(g for g in got if g[0] is repeat)
    assert rep[2].endswith("(repeat)") and "top-right" in rep[1] and "Cause: scratches" in rep[1]
    later = _result(x=40)                              # after it is cached: immediate reuse, no LLM call
    assert svc.submit(later) is False and svc.cached(later)[0].startswith("Cause: scratches")
    assert llm.calls == 1 and svc.stats["joined"] == 1 and svc.stats["reused"] == 1
    svc.close()


def test_advice_cache_persists_across_restarts(cfg, monkeypatch):
    done = threading.Event()
    svc = _service(cfg, monkeypatch, SlowLLM(0.0), lambda *a: done.set())
    svc.submit(_result())
    assert done.wait(3)
    svc.close()
    llm2 = SlowLLM(0.0)
    svc2 = _service(cfg, monkeypatch, llm2, lambda *a: None)
    assert svc2.cached(_result()) is not None and llm2.calls == 0
    svc2.close()


def test_event_pushed_out_of_full_queue_gets_rule_advice(cfg, monkeypatch):
    got = []
    svc = _service(cfg, monkeypatch, SlowLLM(1.0), lambda r, t, e, l: got.append((r.defects[0].label, e)),
                   maxsize=1)
    svc.submit(_result("scratches"))   # worker picks this up
    time.sleep(0.1)
    svc.submit(_result("inclusion"))   # waits in the queue
    svc.submit(_result("patches"))     # pushes "inclusion" out → rule advice now
    assert ("inclusion", "rule") in got and svc.stats["dropped"] == 1
    svc.close()


def test_pregeneration_fills_cache_when_idle(cfg, monkeypatch):
    cfg["advisor"]["pregenerate_idle_s"] = 0
    llm = SlowLLM(0.0)
    svc = _service(cfg, monkeypatch, llm, lambda *a: None)
    cands = pregeneration_candidates(cfg, ["scratches"])
    # scratches is MINOR by default → MINOR/MAJOR/CRITICAL, plus 3 anomaly_* types
    assert {signature(c)[2][0] for c in cands} >= {("scratches", "MINOR"), ("scratches", "CRITICAL")}
    assert cands[0].verdict == Verdict.REVIEW
    svc.pregenerate(cands)
    for _ in range(60):
        if svc.stats["pregenerated"] == len(cands):
            break
        time.sleep(0.05)
    assert svc.stats["pregenerated"] == len(cands)
    r = _result("scratches", Severity.MAJOR)
    assert svc.submit(r) is False and svc.cached(r)
    svc.close()


def test_render_downscales_boxes():
    frame = np.zeros((1000, 800, 3), np.uint8)
    r = _result()
    r.defects[0].bbox = (500, 500, 100, 100)
    out = render(frame, r, np.zeros((64, 64), np.float32), 2.0, max_side=500)
    assert out.shape == (500, 400, 3)
    assert out[300, 250].any() and not out[300, 330].any()  # box edge drawn at scaled position x=250..300


def test_storage_counts_and_deferred_image(tmp_path):
    store = InspectionLog(tmp_path, save_images="non_pass")
    r = _result()
    p = store.image_path_for(r)
    store.log(r, image_path=p)
    ok = InspectionResult(2, time.time(), "steel_strip", Verdict.PASS, 0.3, [])
    assert store.image_path_for(ok) is None
    store.log(ok)
    assert store.stats()["REJECT"] == 1 and store.stats()["PASS"] == 1
    assert store.recent(2, only_defects=True)[0]["image_path"] == p
    store.close()
    assert InspectionLog(tmp_path).stats()["total"] == 2   # counts reloaded from the DB


def test_signature_ignores_box_count():
    one = _result()
    two = _result()
    two.defects = two.defects * 2
    assert signature(one) == signature(two)


def test_signature_keeps_worst_severity_and_ignores_untyped_blobs():
    r = _result("crazing", Severity.CRITICAL)
    plain = signature(r)
    r.defects += [Defect("crazing", (5, 5, 5, 5), 0.5, 1.2, 0.001, "fused", Severity.MAJOR),
                  Defect("anomaly_spot", (9, 9, 3, 3), 0.0, 1.5, 0.001, "anomaly", Severity.MINOR)]
    assert signature(r) == plain == ("steel_strip", "REJECT", (("crazing", "CRITICAL"),))
    only_blob = InspectionResult(1, 0.0, "steel_strip", Verdict.REVIEW, 1.2,
                                 [Defect("anomaly_spot", (9, 9, 3, 3), 0.0, 1.5, 0.001, "anomaly", Severity.MINOR)])
    assert signature(only_blob)[2] == (("anomaly_spot", "MINOR"),)
