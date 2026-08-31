from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..dates import parse_date_query
from ..retrieval.lexical import extract_topic
from .tools import MemoryTools


DATE_PATTERN = re.compile(r"20\d{2}-\d{2}-\d{2}")
FILE_PATTERN = re.compile(r"[^\s：:，,。]+\.(?:docx|pdf|md|txt|png|jpe?g|webp|wav|mp3|m4a|flac)", re.I)


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    evidence: list[dict[str, Any]]
    plan: dict[str, Any]
    mode: str


class OpenVINOTextModel:
    def __init__(self, model_path: Path, device: str = "CPU", max_new_tokens: int = 512) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._pipeline = None

    @property
    def available(self) -> bool:
        return (self.model_path / "openvino_model.xml").exists()

    def _load(self):
        if self._pipeline is None:
            import openvino_genai as ov_genai

            self._pipeline = ov_genai.LLMPipeline(str(self.model_path), self.device)
        return self._pipeline

    def generate(self, prompt: str, max_new_tokens: int | None = None) -> str:
        result = self._load().generate(
            prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=False,
        )
        text = str(result).strip()
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    def unload(self) -> None:
        self._pipeline = None


def _json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class RecallAgent:
    def __init__(self, config: AppConfig, tools: MemoryTools, model: OpenVINOTextModel | None = None) -> None:
        self.config = config
        self.tools = tools
        model_path = config.model_dir / config.models.agent_id.split("/")[-1]
        self.model = model or OpenVINOTextModel(model_path, config.device, config.models.max_new_tokens)

    def _fallback_plan(self, question: str) -> dict[str, Any]:
        start, end = parse_date_query(question, self.config.study_year)
        topic = extract_topic(question)
        if start and not topic:
            return {"tool": "get_timeline", "start_date": start, "end_date": end, "topic": None}
        return {"tool": "search_memory", "query": topic or question, "start_date": start, "end_date": end,
                "file_type": "docx" if "Word" in question or "word" in question else None}

    def _plan(self, question: str) -> dict[str, Any]:
        fallback = self._fallback_plan(question)
        if not self.config.models.agent_enabled or not self.model.available:
            return fallback
        prompt = f"""你是本地记忆检索规划器。只能选择一个工具：search_memory 或 get_timeline。
严格输出单行 JSON，不要解释。日期用 YYYY-MM-DD。无法确定的字段为 null。
search_memory 参数：query,start_date,end_date,file_type。
get_timeline 参数：start_date,end_date,topic。
当前学习年份：{self.config.study_year}
问题：{question}
不要展开思考，只做工具选择。/no_think
JSON："""
        try:
            candidate = _json_object(self.model.generate(prompt, 180))
        except Exception:
            return fallback
        if not candidate or candidate.get("tool") not in {"search_memory", "get_timeline"}:
            return fallback
        if candidate["tool"] == "get_timeline" and not candidate.get("start_date"):
            return fallback
        candidate.setdefault("query", fallback.get("query"))
        return candidate

    def _execute(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        if plan["tool"] == "get_timeline":
            return self.tools.get_timeline(
                str(plan["start_date"]), plan.get("end_date"), plan.get("topic"), limit=30,
            )
        return self.tools.search_memory(
            str(plan.get("query") or ""), plan.get("start_date"), plan.get("end_date"),
            plan.get("file_type"), limit=12,
        )

    @staticmethod
    def _deterministic_answer(question: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return "目前没有检索到足够证据。可以换一个关键词，或先点击“立即扫描资料”。"
        grouped: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in evidence:
            if item["document_id"] not in seen:
                grouped.append(item)
                seen.add(item["document_id"])
        lines = [f"找到了 {len(grouped)} 份相关资料。"]
        for index, item in enumerate(grouped[:8], 1):
            lines.extend([
                "",
                f"{index}. {item.get('event_date') or '日期不确定'}｜{item['filename']}",
                f"证据：{item['snippet']}",
                f"来源：{item['path']}",
            ])
        return "\n".join(lines)

    def _grounded(self, answer: str, evidence: list[dict[str, Any]]) -> bool:
        allowed_dates = {item.get("event_date") for item in evidence if item.get("event_date")}
        allowed_files = {item["filename"] for item in evidence}
        if any(date not in allowed_dates for date in DATE_PATTERN.findall(answer)):
            return False
        for name in FILE_PATTERN.findall(answer):
            mentioned = Path(name).name
            if not any(filename == mentioned or filename.endswith(mentioned) for filename in allowed_files):
                return False
        return True

    def answer(self, question: str) -> AgentAnswer:
        plan = self._plan(question)
        evidence = self._execute(plan)
        if not evidence and plan != self._fallback_plan(question):
            plan = self._fallback_plan(question)
            evidence = self._execute(plan)
        fallback = self._deterministic_answer(question, evidence)
        if not evidence or not self.config.models.agent_enabled or not self.model.available:
            return AgentAnswer(fallback, evidence, plan, "deterministic")
        compact = [{key: item[key] for key in ("event_date", "filename", "path", "snippet", "source_kind")}
                   for item in evidence[:10]]
        prompt = f"""你是个人第二大脑。只根据下面证据回答用户，不得编造日期、文件名、路径或学习事实。
先直接回答，再按时间列出关键证据，最后列出来源。证据不足要明确说不足。使用中文。
用户问题：{question}
证据 JSON：{json.dumps(compact, ensure_ascii=False)}
不要展示思考过程，只输出简洁答案。/no_think
回答："""
        try:
            generated = self.model.generate(prompt, min(self.config.models.max_new_tokens, 384))
            if generated and self._grounded(generated, evidence):
                return AgentAnswer(generated, evidence, plan, "openvino")
        except Exception:
            pass
        return AgentAnswer(fallback, evidence, plan, "deterministic")

    def close(self) -> None:
        self.model.unload()
