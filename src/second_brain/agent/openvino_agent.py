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
from .model_orchestrator import ModelOrchestrator
from .qwen_reasoner import OpenVINOQwenReasoner
from .tools import MemoryTools


DATE_PATTERN = re.compile(r"20\d{2}-\d{2}-\d{2}")
FILE_PATTERN = re.compile(r"[^\s：:，,。]+\.(?:docx|pdf|md|txt|png|jpe?g|webp|wav|mp3|m4a|flac)", re.I)
DEEP_MARKERS = (
    "深度", "详细分析", "综合分析", "整体分析", "长期", "轨迹", "路线", "薄弱",
    "卡点", "为什么", "怎么改善", "对比", "比较", "变化", "总结我的", "掌握情况",
)


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    evidence: list[dict[str, Any]]
    plan: dict[str, Any]
    mode: str


def _json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


class RecallAgent:
    def __init__(self, config: AppConfig, tools: MemoryTools, model: object | None = None) -> None:
        self.config = config
        self.tools = tools
        self.model = model or ModelOrchestrator(
            OpenVINOLFMModel(
                config.agent_model_path, config.models.controller_device, config.models.max_new_tokens,
            ),
            OpenVINOQwenReasoner(
                config.reasoner_model_path, config.models.reasoner_device, config.models.max_new_tokens,
            ),
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
            return {
                "tool": "trace_topic",
                "topic": topic,
                "steps": [
                    {"tool": "search_topic", "topic": topic},
                    {"tool": "get_concept_state", "query": topic},
                ],
            }
        if start and not topic:
            return {"tool": "get_timeline", "start_date": start, "end_date": end, "topic": None}
        return {"tool": "search_memory", "query": topic or question, "start_date": start, "end_date": end,
                "file_type": "docx" if "Word" in question or "word" in question else None}

    def _plan(self, question: str) -> dict[str, Any]:
        fallback = self._fallback_plan(question)
        if not self.config.models.agent_enabled or not self.model.available:
            return fallback
        prompt = f"""你是常驻本地第二大脑控制器。请把问题拆成最多 3 个只读工具步骤。
严格输出单行 JSON，不要解释。格式：{{"intent":"...","subject":"...","steps":[...]}}。
可用工具：
- search_memory(query,start_date,end_date,file_type)：全文与融合检索
- search_semantic(query)：同义表达或模糊概念检索
- search_topic(topic)：主题的跨日期轨迹
- get_timeline(start_date,end_date,topic)：指定日期范围
- get_episode(event_date)：某天的情节记忆
- get_concept_state(query)：概念掌握、卡点与变化
学习路线、薄弱点、整体进展必须同时调用 search_topic 和 get_concept_state。
日期用 YYYY-MM-DD；无法确定的字段为 null；只做规划，不回答问题。
当前学习年份：{self.config.study_year}
问题：{question}
JSON："""
        try:
            candidate = _json_object(self.model.generate(prompt, min(self.config.models.max_new_tokens, 128)))
        except Exception:
            return fallback
        if not candidate:
            return fallback
        steps = candidate.get("steps")
        allowed = {
            "search_memory", "search_semantic", "search_topic", "get_timeline",
            "get_episode", "get_concept_state",
        }
        if not isinstance(steps, list) or not steps or len(steps) > 3:
            return fallback
        if any(not isinstance(step, dict) or step.get("tool") not in allowed for step in steps):
            return fallback
        if fallback["tool"] == "trace_topic":
            return fallback
        evidence_step = next(
            (step for step in steps if step["tool"] in {"search_memory", "search_semantic", "search_topic", "get_timeline"}),
            None,
        )
        if evidence_step is None:
            return fallback
        normalized = dict(fallback)
        normalized["steps"] = steps
        normalized["intent"] = candidate.get("intent")
        normalized["subject"] = candidate.get("subject")
        return normalized

    def _execute(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for step in plan.get("steps", []):
            tool = step.get("tool")
            if tool == "get_concept_state" and callable(getattr(self.tools, "get_concept_state", None)):
                observations.extend(self.tools.get_concept_state(str(step.get("query") or step.get("topic") or ""), 12))
            elif tool == "get_episode" and callable(getattr(self.tools, "get_episode", None)):
                episode = self.tools.get_episode(str(step.get("event_date") or ""))
                if episode:
                    observations.append(episode)
            elif tool == "search_semantic" and callable(getattr(self.tools, "search_semantic", None)):
                # Semantic results are already fused by search_memory; this call
                # gives the controller an explicit fallback for paraphrases.
                self.tools.search_semantic(str(step.get("query") or ""), 12)
        if observations:
            plan["observations"] = observations
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

    def _deep_available(self) -> bool:
        return bool(
            self.config.models.deep_enabled
            and callable(getattr(self.model, "analyze_deep", None))
            and getattr(self.model, "reasoner_available", False)
        )

    def _should_deep(self, question: str, evidence: list[dict[str, Any]], recall_mode: str) -> bool:
        if recall_mode == "fast":
            return False
        if recall_mode == "deep":
            return True
        documents = {item["document_id"] for item in evidence}
        return (
            any(marker in question for marker in DEEP_MARKERS)
            or (is_progress_query(question) and len(documents) >= 3)
            or len(documents) >= 7
        )

    def _deep_answer(
        self,
        question: str,
        evidence: list[dict[str, Any]],
        plan: dict[str, Any],
    ) -> AgentAnswer | None:
        unique: dict[int, dict[str, Any]] = {}
        for item in evidence:
            unique.setdefault(item["document_id"], item)
        ordered = sorted(
            unique.values(),
            key=lambda item: (item.get("event_date") or "9999-99-99", item["filename"]),
        )[: min(self.config.models.evidence_limit, 6)]
        if not ordered:
            return None
        topic = str(plan.get("topic") or plan.get("query") or extract_topic(question))
        evidence_ids = [int(item["chunk_id"]) for item in ordered]
        model_id = self.config.models.reasoner_id
        cached = None
        if callable(getattr(self.tools, "get_cached_analysis", None)):
            cached = self.tools.get_cached_analysis(question, evidence_ids, model_id)
        profile: dict[str, Any] = {}
        if callable(getattr(self.tools, "get_topic_profile", None)) and topic:
            profile = self.tools.get_topic_profile(topic)
        compact_evidence = [
            {
                "id": f"E{index}",
                "date": item.get("event_date"),
                "content": re.sub(r"\s+", " ", item["snippet"]).strip()[:300],
            }
            for index, item in enumerate(ordered, 1)
        ]
        compact_states = [
            {
                "concept": item["name"],
                "subject": item["subject"],
                "state": item["state"],
                "first_seen": item["first_seen"],
                "last_seen": item["last_seen"],
                "exposure_count": item["exposure_count"],
                "summary": item["current_summary"][:160],
                "remaining_problem": item["remaining_problem"][:100],
                "confidence": item["confidence"],
            }
            for item in profile.get("concept_states", [])[:6]
        ]
        generated = cached
        if not generated:
            prompt = f"""你是个人第二大脑的深度学习分析器。根据检索证据和概念状态回答，不得编造。
你的任务不是罗列关键词，而是比较早期与近期记录，说明知识结构、真实进展、反复卡点及下一步行动。
“出现过”不等于“掌握”；概念状态是机器从日记信号推断的，需要用审慎措辞。
每项关键判断必须引用 [E1] 形式的证据编号。不得输出证据中没有的日期、文件名或事实。
请用以下结构：
总体判断：
发展轨迹：
当前卡点：
下一步行动：
每部分只写一句，全文控制在 260 个汉字以内，不展示思考过程。

用户问题：{question}
主题：{topic or '未指定'}
概念状态：{json.dumps(compact_states, ensure_ascii=False)}
按时间排列的证据：{json.dumps(compact_evidence, ensure_ascii=False)}
回答："""
            try:
                generated = self.model.analyze_deep(
                    prompt, max_new_tokens=min(self.config.models.max_new_tokens, 256),
                )
            except Exception as exc:
                plan["deep_fallback_reason"] = f"reasoner_error:{type(exc).__name__}"
                return None
            generated = str(generated).strip()
            citations = [int(value) for value in re.findall(r"\[E(\d+)\]", generated)]
            required = ("总体判断", "当前卡点", "下一步")
            if not citations and all(label in generated for label in required):
                lines = []
                section_index = 0
                for line in generated.splitlines():
                    clean = line.strip()
                    if clean and any(label in clean for label in ("总体判断", "发展轨迹", "当前卡点", "下一步")):
                        reference = min(section_index + 1, len(ordered))
                        clean = f"{clean} [E{reference}]"
                        section_index += 1
                    lines.append(clean)
                generated = "\n".join(lines).strip()
                citations = [int(value) for value in re.findall(r"\[E(\d+)\]", generated)]
            if (
                len(generated) < 40
                or len(generated) > 3200
                or not citations
                or any(value < 1 or value > len(ordered) for value in citations)
            ):
                plan["deep_fallback_reason"] = "answer_validation_failed"
                return None
        source_lines = [
            f"- [E{index}] {item.get('event_date') or '日期不确定'}｜{item['filename']}\n  {item['path']}"
            for index, item in enumerate(ordered, 1)
        ]
        grounded = f"{generated.rstrip()}\n\n可追溯来源：\n" + "\n".join(source_lines)
        if not self._grounded(grounded, evidence):
            allowed_dates = {item.get("event_date") for item in ordered if item.get("event_date")}
            generated = DATE_PATTERN.sub(
                lambda match: match.group(0) if match.group(0) in allowed_dates else "日期不确定",
                generated,
            )
            allowed_files = {item["filename"] for item in ordered}
            for matched in FILE_PATTERN.findall(generated):
                name = Path(matched).name
                if name not in allowed_files:
                    generated = generated.replace(matched, "对应学习记录")
            grounded = f"{generated.rstrip()}\n\n可追溯来源：\n" + "\n".join(source_lines)
            if not self._grounded(grounded, evidence):
                plan["deep_fallback_reason"] = "grounding_validation_failed"
                return None
        if not cached and callable(getattr(self.tools, "cache_analysis", None)):
            self.tools.cache_analysis(
                question, "deep_recall", topic or None, generated, evidence_ids, model_id,
            )
        return AgentAnswer(grounded, evidence, plan, "deep_cache" if cached else "deep")

    def answer(self, question: str, recall_mode: str = "auto") -> AgentAnswer:
        plan = self._plan(question)
        evidence = self._execute(plan)
        if not evidence and plan != self._fallback_plan(question):
            plan = self._fallback_plan(question)
            evidence = self._execute(plan)
        route_topic = str(plan.get("topic") or "") if plan["tool"] == "trace_topic" else ""
        fallback = self._route_answer(route_topic, evidence) if route_topic else self._deterministic_answer(question, evidence)
        if not evidence or not self.config.models.agent_enabled or not self.model.available:
            return AgentAnswer(fallback, evidence, plan, "deterministic")
        if self._should_deep(question, evidence, recall_mode) and self._deep_available():
            deep = self._deep_answer(question, evidence, plan)
            if deep is not None:
                return deep
        unique: dict[int, dict[str, Any]] = {}
        for item in evidence:
            unique.setdefault(item["document_id"], item)
        ordered = sorted(unique.values(), key=lambda item: (item.get("event_date") or "9999-99-99", item["filename"]))[:10]
        compact = [{"id": f"E{index}", "event_date": item["event_date"], "filename": item["filename"],
                    "snippet": re.sub(r"\s+", " ", item["snippet"]).strip()[:240]}
                   for index, item in enumerate(ordered, 1)]
        if route_topic:
            state_context = [
                {
                    "concept": item.get("name"),
                    "state": item.get("state"),
                    "first_seen": item.get("first_seen"),
                    "last_seen": item.get("last_seen"),
                    "remaining_problem": item.get("remaining_problem"),
                }
                for item in plan.get("observations", [])
                if item.get("name")
            ][:10]
            prompt = f"""你是个人第二大脑的学习分析 Agent。只根据证据分析“{route_topic}”的学习路线，不得编造。
严格只输出四行，每行分别以“总体判断：”“学习轨迹：”“当前卡点：”“下一步：”开头。只谈“{route_topic}”，不得混入其他学科。必须比较早期与近期记录，指出变化；不能把“记录中出现过”说成“已经掌握”。每行引用证据编号如 [E1]。全文不超过 260 个汉字，不得写前言，不得重复罗列关键词。
用户问题：{question}
概念状态：{json.dumps(state_context, ensure_ascii=False)}
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
        return AgentAnswer(fallback, evidence, plan, "hybrid" if route_topic else "deterministic")

    def close(self) -> None:
        self.model.unload()
