from pathlib import Path

from second_brain.agent.openvino_agent import OpenVINOTextModel, RecallAgent
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


class FakeModel:
    def __init__(self, responses, available=True):
        self.responses = list(responses)
        self.available = available
        self.unloaded = False
        self.prompts = []

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
    assert all("/no_think" in prompt for prompt in model.prompts)
    assert "可追溯来源" in answer.answer


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
