from pathlib import Path

from second_brain.agent.openvino_agent import RecallAgent
from second_brain.config import AppConfig, IngestionConfig, ModelConfig


class FakeTools:
    def __init__(self):
        self.calls = []

    def search_memory(self, query, start_date=None, end_date=None, file_type=None, limit=12):
        self.calls.append(("search_memory", query, start_date, end_date, file_type))
        return [{
            "document_id": 1, "chunk_id": 2, "event_date": "2026-08-21",
            "filename": "8.21 申论.docx", "path": r"D:\资料\8.21 申论.docx",
            "snippet": "复习申论和公文写作。", "source_kind": "text",
        }]

    def get_timeline(self, start_date, end_date=None, topic=None, limit=30):
        self.calls.append(("get_timeline", start_date, end_date, topic))
        return self.search_memory(topic or "", start_date, end_date)

    def trace_topic(self, topic, limit=12):
        self.calls.append(("trace_topic", topic))
        return self.search_memory(topic, limit=limit)


class FakeModel:
    def __init__(self, responses, available=True):
        self.responses = list(responses)
        self.available = available
        self.loaded = False
        self.unloaded = False
        self.prompts = []

    def warmup(self):
        self.loaded = True

    def generate(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        return self.responses.pop(0)

    def unload(self):
        self.unloaded = True


def config(tmp_path, enabled=True):
    return AppConfig(
        config_path=tmp_path / "config.toml", study_year=2026, data_dir=tmp_path / "data",
        model_dir=tmp_path / "models", device="CPU", source_roots=(tmp_path / "source",),
        ingestion=IngestionConfig(), models=ModelConfig(agent_enabled=enabled),
    )


def test_agent_plans_tool_and_synthesizes_grounded_answer(tmp_path):
    model = FakeModel([
        '{"tool":"search_memory","query":"申论","start_date":null,"end_date":null,"file_type":"docx"}',
        "复习过。明确记录在 2026-08-21，来源是 8.21 申论.docx。",
    ])
    tools = FakeTools()
    answer = RecallAgent(config(tmp_path), tools, model).answer("我以前复习过申论吗？")
    assert answer.mode == "openvino"
    assert answer.plan["tool"] == "search_memory"
    assert tools.calls[0][1] == "申论"
    assert all("/no_think" not in prompt for prompt in model.prompts)
    assert "不要写日期、文件名、路径" in model.prompts[1]
    assert "可追溯来源" in answer.answer


def test_agent_traces_topic_progress_instead_of_searching_whole_question(tmp_path):
    model = FakeModel([
        '{"tool":"search_memory","query":"高等代数学习路线"}',
    ])
    tools = FakeTools()
    answer = RecallAgent(config(tmp_path), tools, model).answer("我的高等代数的学习路线是什么情况？")
    assert answer.mode == "hybrid"
    assert answer.plan["tool"] == "trace_topic"
    assert answer.plan["topic"] == "高等代数"
    assert [step["tool"] for step in answer.plan["steps"]] == ["search_topic", "get_concept_state"]
    assert tools.calls[0] == ("trace_topic", "高等代数")
    assert "总体判断" in answer.answer


def test_repetitive_model_output_falls_back_to_structured_route(tmp_path):
    model = FakeModel([
        '{"tool":"trace_topic","topic":"高等代数"}',
    ])
    answer = RecallAgent(config(tmp_path), FakeTools(), model).answer("高等代数学习路线怎么样？")
    assert answer.mode == "hybrid"
    assert "总体判断" in answer.answer
    assert "学习轨迹" in answer.answer
    assert len(answer.answer) < 1200


def test_off_topic_route_generation_is_rejected(tmp_path):
    model = FakeModel([
        '{"tool":"trace_topic","topic":"高等代数"}',
    ])
    answer = RecallAgent(config(tmp_path), FakeTools(), model).answer("高等代数进展如何？")
    assert answer.mode == "hybrid"
    assert "学习轨迹" in answer.answer


def test_truncated_reasoning_forces_safe_fallback(tmp_path):
    model = FakeModel([
        '{"tool":"search_memory","query":"申论"}',
        "",
    ])
    answer = RecallAgent(config(tmp_path), FakeTools(), model).answer("申论")
    assert answer.mode == "deterministic"
    assert "找到了 1 份相关资料" in answer.answer


def test_hallucinated_date_forces_deterministic_fallback(tmp_path):
    model = FakeModel([
        '{"tool":"search_memory","query":"申论"}',
        "你在 2026-09-99 学过申论。",
    ])
    answer = RecallAgent(config(tmp_path), FakeTools(), model).answer("申论")
    assert answer.mode == "deterministic"
    assert "2026-08-21" in answer.answer


def test_program_appends_verified_source_when_model_omits_filename(tmp_path):
    model = FakeModel([
        '{"tool":"search_memory","query":"申论"}',
        "复习过，来源是证据 JSON。",
    ])
    answer = RecallAgent(config(tmp_path), FakeTools(), model).answer("申论")
    assert answer.mode == "openvino"
    assert "8.21 申论.docx" in answer.answer
    assert "JSON" not in answer.answer


def test_model_unavailable_keeps_recall_working(tmp_path):
    model = FakeModel([], available=False)
    answer = RecallAgent(config(tmp_path), FakeTools(), model).answer("8.21 那天学了什么？")
    assert answer.mode == "deterministic"
    assert answer.plan["tool"] == "get_timeline"
    assert "来源" in answer.answer


def test_close_unloads_model(tmp_path):
    model = FakeModel([], available=False)
    agent = RecallAgent(config(tmp_path), FakeTools(), model)
    agent.close()
    assert model.unloaded


def test_warmup_keeps_lfm_resident(tmp_path):
    model = FakeModel([])
    agent = RecallAgent(config(tmp_path), FakeTools(), model)
    assert agent.warmup()
    assert agent.resident


class FakeDeepModel(FakeModel):
    reasoner_available = True

    def analyze_deep(self, prompt, max_new_tokens=None):
        self.prompts.append(prompt)
        return (
            "总体判断：已经有连续记录，但不能视为完全掌握。[E1]\n"
            "发展轨迹：先暴露问题，后进入复习强化。[E1]\n"
            "当前卡点：证据显示独立调用知识仍不稳定。[E1]\n"
            "下一步行动：按错题与概念状态做针对性复盘。[E1]"
        )


class FakeDeepModelWithoutCitations(FakeDeepModel):
    def analyze_deep(self, prompt, max_new_tokens=None):
        return (
            "总体判断：已有连续学习记录，但还不能视为完全掌握。\n"
            "发展轨迹：从接触概念转入专题复习。\n"
            "当前卡点：独立组织证明仍不稳定。\n"
            "下一步行动：按概念状态复盘并做限时验证。"
        )


class FakeDeepModelWithUnknownDate(FakeDeepModel):
    def analyze_deep(self, prompt, max_new_tokens=None):
        return (
            "总体判断：在 2026-09-99 已经完全掌握。[E1]\n"
            "发展轨迹：记录显示持续练习。[E1]\n"
            "当前卡点：独立证明仍不稳定。[E1]\n"
            "下一步行动：回到原始证据复盘。[E1]"
        )


def test_deep_recall_routes_complex_question_to_qwen(tmp_path):
    tools = FakeTools()
    model = FakeDeepModel(['{"tool":"trace_topic","topic":"高等代数"}'])
    cfg = config(tmp_path)
    cfg = AppConfig(
        config_path=cfg.config_path, study_year=cfg.study_year, data_dir=cfg.data_dir,
        model_dir=cfg.model_dir, device=cfg.device, source_roots=cfg.source_roots,
        ingestion=cfg.ingestion, models=ModelConfig(agent_enabled=True, deep_enabled=True),
    )
    answer = RecallAgent(cfg, tools, model).answer("深度分析我的高等代数学习路线", recall_mode="deep")
    assert answer.mode == "deep"
    assert "当前卡点" in answer.answer
    assert "可追溯来源" in answer.answer
    assert any("概念状态" in prompt for prompt in model.prompts)


def test_deep_recall_adds_verified_references_when_model_omits_them(tmp_path):
    tools = FakeTools()
    model = FakeDeepModelWithoutCitations(['{"tool":"trace_topic","topic":"高等代数"}'])
    cfg = config(tmp_path)
    cfg = AppConfig(
        config_path=cfg.config_path, study_year=cfg.study_year, data_dir=cfg.data_dir,
        model_dir=cfg.model_dir, device=cfg.device, source_roots=cfg.source_roots,
        ingestion=cfg.ingestion, models=ModelConfig(agent_enabled=True, deep_enabled=True),
    )
    answer = RecallAgent(cfg, tools, model).answer("深度分析高等代数", recall_mode="deep")
    assert answer.mode == "deep"
    assert "[E1]" in answer.answer


def test_deep_recall_sanitizes_unverified_dates(tmp_path):
    tools = FakeTools()
    model = FakeDeepModelWithUnknownDate(['{"tool":"trace_topic","topic":"高等代数"}'])
    cfg = config(tmp_path)
    cfg = AppConfig(
        config_path=cfg.config_path, study_year=cfg.study_year, data_dir=cfg.data_dir,
        model_dir=cfg.model_dir, device=cfg.device, source_roots=cfg.source_roots,
        ingestion=cfg.ingestion, models=ModelConfig(agent_enabled=True, deep_enabled=True),
    )
    answer = RecallAgent(cfg, tools, model).answer("深度分析高等代数", recall_mode="deep")
    assert answer.mode == "deep"
    assert "2026-09-99" not in answer.answer
    assert "日期不确定" in answer.answer


def test_deep_sources_with_filename_without_spaces_pass_grounding(tmp_path):
    tools = FakeTools()
    original = tools.search_memory

    def no_space_results(*args, **kwargs):
        results = original(*args, **kwargs)
        results[0]["filename"] = "8.21申论.docx"
        results[0]["path"] = r"D:\资料\8.21申论.docx"
        return results

    tools.search_memory = no_space_results
    model = FakeDeepModel(['{"tool":"trace_topic","topic":"高等代数"}'])
    cfg = config(tmp_path)
    cfg = AppConfig(
        config_path=cfg.config_path, study_year=cfg.study_year, data_dir=cfg.data_dir,
        model_dir=cfg.model_dir, device=cfg.device, source_roots=cfg.source_roots,
        ingestion=cfg.ingestion, models=ModelConfig(agent_enabled=True, deep_enabled=True),
    )
    answer = RecallAgent(cfg, tools, model).answer("深度分析高等代数", recall_mode="deep")
    assert answer.mode == "deep"
    assert "文件：8.21申论.docx" in answer.answer
