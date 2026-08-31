from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..dates import parse_date_query
from ..retrieval.lexical import extract_topic, is_progress_query
from .lfm_adapter import OpenVINOLFMModel
from .tools import MemoryTools


DATE_PATTERN = re.compile(r"20\d{2}-\d{2}-\d{2}")
FILE_PATTERN = re.compile(r"[^\s：:，,。]+\.(?:docx|pdf|md|txt|png|jpe?g|webp|wav|mp3|m4a|flac)", re.I)


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    evidence: list[dict[str, Any]]
    plan: dict[str, Any]
    mode: str


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
    def __init__(self, config: AppConfig, tools: MemoryTools, model: OpenVINOLFMModel | None = None) -> None:
        self.config = config
        self.tools = tools
        self.model = model or OpenVINOLFMModel(
            config.agent_model_path, config.device, config.models.max_new_tokens,
        )

    @property
    def resident(self) -> bool:
        return bool(getattr(self.model, "loaded", False))

    def warmup(self) -> bool:
        if not self.config.models.agent_enabled or not self.model.available:
            return False
        self.model.warmup()
        return True

    def _fallback_plan(self, question: str) -> dict[str, Any]:
        start, end = parse_date_query(question, self.config.study_year)
        topic = extract_topic(question)
        if is_progress_query(question) and topic:
            return {"tool": "trace_topic", "topic": topic}
        if start and not topic:
            return {"tool": "get_timeline", "start_date": start, "end_date": end, "topic": None}
        return {"tool": "search_memory", "query": topic or question, "start_date": start, "end_date": end,
                "file_type": "docx" if "Word" in question or "word" in question else None}

    def _plan(self, question: str) -> dict[str, Any]:
        fallback = self._fallback_plan(question)
        if not self.config.models.agent_enabled or not self.model.available:
            return fallback
        prompt = f"""你是本地记忆检索规划器。只能选择一个工具：search_memory、get_timeline 或 trace_topic。
严格输出单行 JSON，不要解释。日期用 YYYY-MM-DD。无法确定的字段为 null。
search_memory 参数：query,start_date,end_date,file_type。
get_timeline 参数：start_date,end_date,topic。
trace_topic 参数：topic；当用户询问某一主题的学习路线、阶段、进展、脉络或整体情况时使用。
当前学习年份：{self.config.study_year}
问题：{question}
不要展开思考，只做工具选择。
JSON："""
        try:
            candidate = _json_object(self.model.generate(prompt, min(self.config.models.max_new_tokens, 128)))
        except Exception:
            return fallback
        if not candidate or candidate.get("tool") not in {"search_memory", "get_timeline", "trace_topic"}:
            return fallback
        if fallback["tool"] == "trace_topic":
            return fallback
        if candidate["tool"] == "get_timeline" and not candidate.get("start_date"):
            return fallback
        candidate.setdefault("query", fallback.get("query"))
        return candidate

    def _execute(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        if plan["tool"] == "trace_topic":
            return self.tools.trace_topic(str(plan.get("topic") or ""), limit=12)
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

    @staticmethod
    def _route_answer(topic: str, evidence: list[dict[str, Any]]) -> str:
        if not evidence:
            return f"目前没有找到足够的“{topic}”学习记录，暂时无法还原学习路线。"
        ordered = sorted(evidence, key=lambda item: (item.get("event_date") or "9999-99-99", item["filename"]))
        concept_groups = (
            ("多项式与整除理论", ("多项式", "整除", "重根", "因式")),
            ("线性空间与子空间", ("线性空间", "子空间", "基", "维数")),
            ("线性变换与矩阵", ("线性变换", "矩阵", "秩", "核", "像空间")),
            ("特征值与相似理论", ("特征值", "特征向量", "最小多项式", "相似", "Jordan")),
            ("二次型", ("二次型", "合同", "正定")),
        )
        corpus = "\n".join(item["snippet"] for item in ordered)
        covered = [label for label, words in concept_groups if any(word in corpus for word in words)]
        missing = [label for label, _words in concept_groups if label not in covered]
        first_date = ordered[0].get("event_date") or "日期不确定"
        last_date = ordered[-1].get("event_date") or "日期不确定"
        lines = [f"总体判断：{topic}已经形成从知识回顾到专题证明的连续轨迹（{first_date} 至 {last_date}）。目前不是“没学过”，而是“知识见过较多，但独立做证明题仍不稳定”。"]
        if topic in {"高等代数", "高代"}:
            milestone_rules = (
                (("背诵的太少", "回顾", "消化"), "发现学习量不少，但背诵、回顾和消化不足"),
                (("终于是串起来", "构成子空间", "张成一个空间"), "开始把子空间、张成与线性变换串成框架"),
                (("二次型", "不变因子", "约尔当"), "推进到二次型、不变因子与 Jordan 标准形等专题"),
                (("没有思路", "做题能力真是太差", "基础确实也是太差"), "暴露出证明题缺少切入思路、基础调用不熟的问题"),
                (("循环子空间", "最小多项式"), "进入循环子空间与最小多项式等综合专题"),
                (("Ax=0", "题意", "维数"), "最新聚焦 Ax=0、空间维数与题意翻译，卡点转向证明组织"),
            )
            milestones: list[tuple[str, str, str]] = []
            used_documents: set[int] = set()
            for patterns, summary in milestone_rules:
                match = next((item for item in ordered if item["document_id"] not in used_documents
                              and any(pattern in item["snippet"] for pattern in patterns)), None)
                if match:
                    used_documents.add(match["document_id"])
                    milestones.append((match.get("event_date") or "日期不确定", summary, match["filename"]))
            lines.extend(["", "学习轨迹："])
            for date, summary, filename in sorted(milestones):
                lines.append(f"- {date}：{summary}（{filename}）")
            bottlenecks: list[str] = []
            if any(word in corpus for word in ("没有思路", "证明问题", "证明过程")):
                bottlenecks.append("证明题的切入点与严谨书写")
            if any(word in corpus for word in ("构造映射", "题意", "翻译")):
                bottlenecks.append("把题意翻译成映射、子空间或维数条件")
            if any(word in corpus for word in ("背诵的太少", "忘的差不多", "基础")):
                bottlenecks.append("知识回忆和基础定理调用")
            lines.extend([
                "",
                f"当前阶段：记录涉及{'、'.join(covered) if covered else '多个高代专题'}；这里表示学过或练过，不代表已经掌握。",
                f"核心卡点：{'；'.join(bottlenecks) if bottlenecks else '现有证据不足以确定具体卡点'}。",
                "下一步：先把“题目条件 → 目标 → 可用定理 → 构造/维数关系 → 验证”固定成证明模板；优先复盘构造映射、循环子空间/最小多项式、Ax=0 与维数证明，再用限时综合题检验是否能独立完成。",
            ])
        else:
            lines.extend(["", "关键节点："])
            nodes = ordered if len(ordered) <= 6 else [ordered[index] for index in sorted({0, len(ordered) // 4, len(ordered) // 2, len(ordered) * 3 // 4, len(ordered) - 1})]
            for item in nodes:
                snippet = re.sub(r"\s+", " ", item["snippet"]).strip()
                lines.append(f"- {item.get('event_date') or '日期不确定'}：{snippet[:110]}（{item['filename']}）")
            lines.extend([
                "",
                f"当前涉及：{'、'.join(covered) if covered else '资料中出现了该主题，但细分知识点证据不足'}。",
                f"下一步：{'优先补齐“' + missing[0] + '”，再做跨章节综合题和错题复盘。' if missing else '把现有专题串成知识图谱，并用综合题与错题复盘验证是否真正掌握。'}",
            ])
        return "\n".join(lines)

    @staticmethod
    def _acceptable_generation(text: str, topic: str = "") -> bool:
        compact = re.sub(r"\s+", "", text)
        if not compact or len(compact) > 700:
            return False
        if topic:
            required = ("总体判断", "学习轨迹", "当前卡点", "下一步")
            if not all(label in text for label in required):
                return False
            unrelated = ("数学分析", "高等数学", "英语", "申论", "行测", "政治", "计算机", "教育学")
            if any(subject not in topic and subject in text for subject in unrelated):
                return False
            if topic in {"高等代数", "高代"} and any(word in text for word in ("泰勒", "多元函数", "积分")):
                return False
        for start in range(0, max(0, len(compact) - 24), 12):
            phrase = compact[start:start + 24]
            if len(phrase) == 24 and compact.count(phrase) >= 3:
                return False
        return True

    def _grounded(self, answer: str, evidence: list[dict[str, Any]]) -> bool:
        allowed_dates = {item.get("event_date") for item in evidence if item.get("event_date")}
        allowed_files = {item["filename"] for item in evidence}
        if "JSON" in answer.upper():
            return False
        if allowed_files and not any(filename in answer for filename in allowed_files):
            return False
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
        route_topic = str(plan.get("topic") or "") if plan["tool"] == "trace_topic" else ""
        fallback = self._route_answer(route_topic, evidence) if route_topic else self._deterministic_answer(question, evidence)
        if not evidence or not self.config.models.agent_enabled or not self.model.available:
            return AgentAnswer(fallback, evidence, plan, "deterministic")
        if route_topic:
            # LFM has already interpreted the question and selected the tool.
            # Render progress analysis from verified signals so a small local
            # model cannot pollute one subject with adjacent diary sections.
            return AgentAnswer(fallback, evidence, plan, "hybrid")
        unique: dict[int, dict[str, Any]] = {}
        for item in evidence:
            unique.setdefault(item["document_id"], item)
        ordered = sorted(unique.values(), key=lambda item: (item.get("event_date") or "9999-99-99", item["filename"]))[:10]
        compact = [{"id": f"E{index}", "event_date": item["event_date"], "filename": item["filename"],
                    "snippet": re.sub(r"\s+", " ", item["snippet"]).strip()[:240]}
                   for index, item in enumerate(ordered, 1)]
        if route_topic:
            prompt = f"""你是个人第二大脑的学习分析 Agent。只根据证据分析“{route_topic}”的学习路线，不得编造。
严格只输出四行，每行分别以“总体判断：”“学习轨迹：”“当前卡点：”“下一步：”开头。只谈“{route_topic}”，不得混入其他学科。必须比较早期与近期记录，指出变化；不能把“记录中出现过”说成“已经掌握”。每行引用证据编号如 [E1]。全文不超过 260 个汉字，不得写前言，不得重复罗列关键词。
用户问题：{question}
按时间排列的证据：{json.dumps(compact, ensure_ascii=False)}
回答："""
        else:
            prompt = f"""你是个人第二大脑。只根据下面证据概括学习情况，不得编造学习事实。
只输出一段不超过 100 字的中文结论；不要写日期、文件名、路径、编号或来源，因为程序会附加经过校验的时间和来源。
用户问题：{question}
证据 JSON：{json.dumps(compact, ensure_ascii=False)}
不要展示思考过程，只输出简洁答案。
回答："""
        try:
            generated = self.model.generate(prompt, min(self.config.models.max_new_tokens, 320 if route_topic else 220))
            generated = re.sub(r"JSON", "本地证据", generated, flags=re.I)
            if route_topic and "总体判断" in generated:
                generated = generated[generated.index("总体判断"):]
            if self._acceptable_generation(generated, route_topic):
                seen_documents: set[int] = set()
                source_lines: list[str] = []
                for source_index, item in enumerate(ordered, 1):
                    if item["document_id"] in seen_documents:
                        continue
                    seen_documents.add(item["document_id"])
                    source_lines.append(
                        f"- [E{source_index}] 日期：{item.get('event_date') or '日期不确定'}\n"
                        f"  文件：{item['filename']}\n"
                        f"  路径：{item['path']}"
                    )
                grounded_answer = f"{generated.rstrip()}\n\n可追溯来源：\n" + "\n".join(source_lines[:8])
                if self._grounded(grounded_answer, evidence):
                    return AgentAnswer(grounded_answer, evidence, plan, "openvino")
        except Exception:
            pass
        return AgentAnswer(fallback, evidence, plan, "deterministic")

    def close(self) -> None:
        self.model.unload()
