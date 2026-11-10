# Checks the status dashboard's parsing of a quant_eval log and of checklist.md.
from scripts import status

LAYER = "gptq layer {:2d}: 88064 calib tokens, {{'orig': 47}}, 66s, peak {}GiB, RAM free {}GiB, total {} min"


def test_log_of_a_running_attempt():
    st = status.parse_log("\n".join(["[transformers] warning", LAYER.format(0, 9.4, 58.3, 1), LAYER.format(1, 9.6, 58.7, 3)]))
    assert st == {"layers": [(0, 9.4, 58.3, 1), (1, 9.6, 58.7, 3)], "result": None, "error": None}


def test_log_keeps_only_the_resumed_attempt():
    lines = [LAYER.format(i, 9.9, 60.0, i + 1) for i in range(18)] + ["torch.OutOfMemoryError: CUDA out of memory."]
    assert status.parse_log("\n".join(lines))["error"].startswith("torch.OutOfMemoryError")
    lines += [LAYER.format(16, 9.9, 60.0, 1), LAYER.format(17, 9.9, 60.0, 2), '{"config": "t", "kl_top64": 0.08}']
    st = status.parse_log("\n".join(lines))
    assert [layer[0] for layer in st["layers"]] == [16, 17]
    assert st["result"] == {"config": "t", "kl_top64": 0.08} and st["error"] is None


def test_checklist_sections_and_now():
    text = "# 체크리스트\n\n## 지금\n- 하는 일: 평가\n- 다음: 런타임\n\n## 설계\n- [x] a\n- [ ] b\n  - [x] c\n\n## Phase 2\n- [ ] d\n"
    assert status.parse_checklist(text) == (["하는 일: 평가", "다음: 런타임"], [["설계", 2, 3], ["Phase 2", 0, 1]])
