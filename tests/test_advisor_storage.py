import threading
import time

from edgeqc.advisor.executorch_llama import parse_runner_output
from edgeqc.advisor.prompts import build_llama3_prompt
from edgeqc.advisor.service import AdvisorService, load_kb
from edgeqc.storage import InspectionLog
from edgeqc.types import Defect, InspectionResult, Severity, Verdict


def _result(label="scratches", sev=Severity.MAJOR):
    return InspectionResult(1, time.time(), "steel_strip", Verdict.REJECT, 2.4,
                            [Defect(label, (1, 2, 30, 4), 0.8, 2.4, 0.03, "fused", sev)])


def test_prompt_is_grounded(cfg):
    p = build_llama3_prompt(_result(), load_kb(cfg["advisor"]["knowledge_base"]))
    assert "type=scratches" in p and "misaligned guides" in p
    assert p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_parse_runner_output():
    prompt = "<|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    stdout = prompt + "Cause: debris.\nAction: stop line.<|eot_id|>\nPyTorchObserver {\"x\":1}\n"
    assert parse_runner_output(stdout, prompt) == "Cause: debris.\nAction: stop line."


def test_advisor_service_falls_back_and_dedupes(cfg):
    cfg["advisor"]["engine"] = "executorch"
    cfg["advisor"]["model_path"] = "models/llm/does_not_exist.pte"  # artefacts absent → rule fallback
    got, done = [], threading.Event()

    def cb(result, text, engine, latency):
        got.append((engine, text))
        done.set()

    svc = AdvisorService(cfg, cb)
    assert svc.engine_name == "rule"
    assert svc.submit(_result()) is True
    assert svc.submit(_result()) is False  # same signature within cooldown
    assert done.wait(3)
    svc.close()
    assert got[0][0] == "rule" and "Cause:" in got[0][1]


def test_inspection_log_roundtrip(tmp_path):
    import numpy as np

    store = InspectionLog(tmp_path, save_images="non_pass")
    r = _result()
    iid = store.log(r, np.zeros((10, 10, 3), np.uint8))
    store.set_advice(iid, "Cause: x", "rule", 0.01)
    assert store.acknowledge(iid)
    rows = store.recent(5)
    assert rows[0]["advice"] == "Cause: x" and rows[0]["defects"][0]["label"] == "scratches"
    assert rows[0]["image_path"] and rows[0]["acknowledged_by"] == "operator"
    assert store.stats()["REJECT"] == 1
    csv = store.export_csv(tmp_path / "log.csv")
    assert "scratches" in csv.read_text()


def test_prompt_includes_location(cfg):
    r = _result()
    r.image_hw = (100, 100)
    p = build_llama3_prompt(r, load_kb(cfg["advisor"]["knowledge_base"]))
    assert "location=top-left" in p.split("Findings:")[-1]
    assert "x=1,y=2" not in p  # pixel boxes are not sent to the model


def test_prompt_head_is_shared_by_same_class(cfg):
    """Second KV-cache level: same defect class -> byte-identical head; only the findings tail differs."""
    from edgeqc.advisor.prompts import llama3_prompt_parts, llama3_static_prefix

    kb = load_kb(cfg["advisor"]["knowledge_base"])
    a, b = _result("scratches"), _result("scratches")
    b.image_score = 3.3
    (ha, ta), (hb, tb) = llama3_prompt_parts(a, kb), llama3_prompt_parts(b, kb)
    assert ha == hb and ta != tb and ha.startswith(llama3_static_prefix())
    assert llama3_prompt_parts(_result("inclusion"), kb)[0] != ha


def test_log_line_is_appended_by_code():
    from edgeqc.advisor.prompts import log_line

    r = _result("rolled-in_scale")
    r.image_hw = (100, 100)
    assert log_line(r).startswith(f"{r.verdict.value} - ") and "rolled-in_scale at top-left" in log_line(r)


def test_grounding_guard_and_log_split():
    from edgeqc.advisor.prompts import is_grounded, split_log_entry

    r = _result("rolled-in_scale")
    good = "Cause: weak descaling left rolled-in scale.\nAction: hold coil.\nCheck: nozzles.\nLog: REJECT - MAJOR rolled-in scale top-left."
    assert is_grounded(good, r)
    # naming the defect only in the code-written Log line does not count
    assert not is_grounded("Cause: weak descaling.\nAction: hold coil.\nLog: REJECT - MAJOR rolled-in scale.", r)
    assert not is_grounded("Cause: x\nAction: y\nLog: z", r)  # never names the defect
    assert not is_grounded("The part looks fine to me.", r)  # wrong format
    # singular / -ing forms of the class name count (Pi advisor eval, 19 Nov: "a patch", "patching")
    p = _result("patches")
    assert is_grounded("Cause: Uneven scale removal causing a patch.\nAction: Re-inspect after cleaning.", p)
    assert not is_grounded("Cause: Uneven pickling.\nAction: Re-inspect after cleaning.", p)
    advice, entry = split_log_entry(good)
    assert entry.startswith("REJECT - MAJOR rolled-in") and "Log:" not in advice


def test_ungrounded_llm_reply_falls_back_to_rule(cfg):
    from edgeqc.advisor.service import AdvisorService

    class FakeLLM:
        name = "llama-executorch"

        def advise(self, result):
            return "Everything is fine."

    svc = AdvisorService(cfg, lambda *a: None)
    svc.llm = FakeLLM()
    text, engine, _ = svc.advise_sync(_result())
    svc.close()
    assert engine == "rule" and "Log:" in text and svc.ungrounded_count == 1


def test_greedy_rejection_not_retried_but_errors_are(cfg):
    import time as _t

    from edgeqc.advisor.service import AdvisorService

    class FakeLLM:
        name = "llama-executorch"
        calls = 0
        fail = False

        def advise(self, result):
            FakeLLM.calls += 1
            if FakeLLM.fail:
                raise RuntimeError("timeout")
            return "Everything is fine."  # rejected by the grounding check

    def run_once(svc, r):
        n = svc.stats["generated"]
        assert svc.submit(r)
        t0 = _t.monotonic()
        while svc.stats["generated"] == n and _t.monotonic() - t0 < 5:
            _t.sleep(0.01)

    svc = AdvisorService(cfg, lambda *a: None)
    svc.llm, svc.greedy, svc.cooldown = FakeLLM(), True, 60.0
    run_once(svc, _result())
    assert not svc.submit(_result()) and FakeLLM.calls == 1  # same reply would come back: keep rule advice
    FakeLLM.fail = True
    r2 = _result("inclusion")
    run_once(svc, r2)
    sig = next(s for s in svc._cache if "inclusion" in str(s))
    assert svc._cache[sig][2] != float("inf")  # an error is retried after cooldown_s
    svc.close()


def test_log_entry_stored_separately(tmp_path, cfg):
    from edgeqc.advisor.rule_based import RuleBasedAdvisor

    r = _result()
    r.image_hw = (100, 100)
    text = RuleBasedAdvisor(load_kb(cfg["advisor"]["knowledge_base"])).advise(r)
    store = InspectionLog(tmp_path)
    iid = store.log(r)
    store.set_advice(iid, text, "rule", 0.0)
    row = store.recent(1)[0]
    assert row["log_entry"].startswith("REJECT - MAJOR scratches at top-left")
    assert "Log:" not in row["advice"]


def test_product_specific_kb_notes(cfg):
    from edgeqc.advisor.prompts import kb_notes
    from edgeqc.advisor.rule_based import RuleBasedAdvisor

    kb = load_kb(cfg["advisor"]["knowledge_base"])
    r = InspectionResult(1, time.time(), "pcb", Verdict.REJECT, 2.0,
                         [Defect("anomaly_area", (1, 2, 30, 40), 0.0, 2.0, 0.1, "anomaly", Severity.MAJOR)])
    assert "pick-and-place" in kb_notes(r, kb) and "pick-and-place" in RuleBasedAdvisor(kb).advise(r)
    r.product = "bottle"  # no product entry -> generic note
    assert "lighting" in kb_notes(r, kb)
