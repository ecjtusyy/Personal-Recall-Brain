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
    assert answer.plan == {"tool": "trace_topic", "topic": "高等代数"}
    assert tools.calls[0] == ("trace_topic", "高等代数")
    assert "总体判断" in answer.answer


def test_repetitive_model_output_falls_back_to_structured_route(tmp_path):
    model = FakeModel([
        '{"tool":"trace_topic","topic":"高等代数"}',
    ])
    answer = RecallAgent(config(tmp_path), FakeTools(), model).answer("高等代数学习路线怎么样？")
    assert answer.mode == "hybrid"
    assert "总体判断" in answer.answer
    assert "关键节点" in answer.answer
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
