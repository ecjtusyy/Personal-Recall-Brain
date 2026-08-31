from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from second_brain.agent.openvino_agent import RecallAgent
from second_brain.agent.tools import MemoryTools
from second_brain.config import load_config
from second_brain.db import open_db
from second_brain.dates import parse_date_query
from second_brain.ingestion.scanner import Scanner
from second_brain.ingestion.enrichment import ImageEnricher
from second_brain.model_download import download_models


st.set_page_config(page_title="Personal Recall Brain", page_icon="🧠", layout="wide")


@st.cache_resource
def runtime(config_path: str):
    config = load_config(config_path)
    conn = open_db(config.db_path)
    return config, conn


@st.cache_resource
def agent_runtime(config_path: str):
    agent_config, agent_conn = runtime(config_path)
    return RecallAgent(agent_config, MemoryTools(agent_config, agent_conn))


CONFIG_PATH = os.environ.get("SECOND_BRAIN_CONFIG", "config.toml")
config, conn = runtime(str(Path(CONFIG_PATH).resolve()))
tools = MemoryTools(config, conn)


def open_document(document_id: int) -> None:
    result = tools.open_source(document_id)
    if not result.get("ok"):
        st.error(result.get("error") or "无法打开来源")
        return
    if os.name == "nt":
        os.startfile(result["path"])
    else:
        st.info(result["path"])


def show_evidence(items, prefix: str) -> None:
    seen = set()
    for item in items:
        if item["chunk_id"] in seen:
            continue
        seen.add(item["chunk_id"])
        with st.container(border=True):
            left, right = st.columns([5, 1])
            left.markdown(f"**{item.get('event_date') or '日期不确定'} · {item['filename']}**")
            left.caption(f"证据类型：{item['source_kind']}　日期来源：{item.get('date_source') or '未知'}")
            left.write(item["snippet"])
            left.code(item["path"], language=None)
            right.button("打开原文件", key=f"{prefix}-open-{item['chunk_id']}", on_click=open_document,
                         args=(item["document_id"],), use_container_width=True)


status = tools.status()
documents = status["documents"]

with st.sidebar:
    st.title("🧠 第二大脑")
    st.caption("本地运行 · 资料只读 · OpenVINO")
    c1, c2 = st.columns(2)
    c1.metric("已索引", documents.get("ready") or 0)
    c2.metric("知识块", status.get("chunks") or 0)
    if st.button("立即扫描资料", type="primary", use_container_width=True):
        progress = st.status("正在只读扫描…", expanded=True)
        stats = Scanner(config, conn, progress=progress.write).scan()
        progress.update(label=f"扫描完成：更新 {stats.files_changed}，跳过 {stats.files_skipped}，失败 {stats.files_failed}",
                        state="complete" if not stats.files_failed else "error")
        st.cache_resource.clear()
        st.rerun()
    images = status.get("images") or {}
    pending_images = images.get("pending") or 0
    if pending_images:
        st.caption(f"还有 {pending_images} 张图片可补充文字；每批完成后自动保存。")
        if st.button("补充下一批图片文字（最多 50 张）", use_container_width=True):
            progress = st.status("OpenVINO 正在识别图片…", expanded=True)
            result = ImageEnricher(config, conn, progress=progress.write).enrich(limit=50)
            progress.update(
                label=f"本批完成 {result.completed} 张，识别到文字 {result.with_text} 张，失败 {result.failed} 张",
                state="complete" if not result.failed else "error",
            )
            st.cache_resource.clear()
            st.rerun()
    st.divider()
    agent_path = config.model_dir / config.models.agent_id.split("/")[-1] / "openvino_model.xml"
    st.write("智能问答模型", "✅ 已就绪" if agent_path.exists() else "⬇️ 尚未下载")
    if not agent_path.exists() and st.button("下载核心 OpenVINO 模型", use_container_width=True):
        with st.status("正在下载约 1.2GB 的 INT4 模型…", expanded=True) as model_status:
            try:
                for model_path in download_models(config.config_path, "core"):
                    st.write(model_path)
                model_status.update(label="核心模型下载完成", state="complete")
                st.rerun()
            except Exception as exc:
                model_status.update(label=f"下载失败：{exc}", state="error")
    st.caption("模型只写入项目 models 目录；不会写入学习资料目录。")

st.title("Personal Recall Brain")
st.caption("先检索证据，再由本地 OpenVINO Agent 组织回答。每条事实都能回到原文件。")

chat_tab, search_tab, timeline_tab, status_tab = st.tabs(["💬 问第二大脑", "🔎 精确搜索", "🗓️ 学习时间轴", "🛡️ 状态与安全"])

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("evidence"):
                show_evidence(message["evidence"], f"history-{message['id']}")
    if question := st.chat_input("例如：我以前什么时候复习过申论？"):
        st.session_state.messages.append({"role": "user", "content": question, "id": len(st.session_state.messages)})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("正在检索本地证据…"):
                result = agent_runtime(str(Path(CONFIG_PATH).resolve())).answer(question)
            st.markdown(result.answer)
            st.caption("回答模式：OpenVINO 本地 Agent" if result.mode == "openvino" else "回答模式：确定性检索（模型未加载或回答校验未通过）")
            show_evidence(result.evidence, f"answer-{len(st.session_state.messages)}")
        st.session_state.messages.append({
            "role": "assistant", "content": result.answer, "evidence": result.evidence,
            "id": len(st.session_state.messages),
        })

with search_tab:
    query = st.text_input("搜索关键词", placeholder="申论、公文写作、最小多项式……")
    col1, col2, col3 = st.columns(3)
    start_date = col1.text_input("开始日期（可选）", placeholder="2026-08-21") or None
    end_date = col2.text_input("结束日期（可选）", placeholder="2026-08-31") or None
    file_type = col3.selectbox("资料类型", ["全部", "docx", "pdf", "md", "txt", "png", "jpg", "音频"])
    if query:
        selected_type = None if file_type == "全部" or file_type == "音频" else file_type
        results = tools.search_memory(query, start_date, end_date, selected_type, 30)
        if file_type == "音频":
            results = [item for item in results if item["file_type"] in {"wav", "mp3", "m4a", "flac"}]
        st.write(f"找到 {len(results)} 条证据")
        show_evidence(results, "search")

with timeline_tab:
    date_text = st.text_input("日期", value="8.21", placeholder="8.21 或 2026-08-21")
    topic = st.text_input("主题（可选）", placeholder="申论") or None
    start, end = parse_date_query(date_text, config.study_year)
    if start:
        results = tools.get_timeline(start, end, topic, 100)
        st.write(f"{start} 共找到 {len(results)} 条记录")
        show_evidence(results, "timeline")
    else:
        st.info("请输入有效日期，例如 8.21 或 2026-08-21。")

with status_tab:
    st.subheader("数据安全边界")
    st.success("学习资料目录只读：程序没有删除、移动、改名或写回原文件的功能。")
    for root in config.source_roots:
        st.code(str(root), language=None)
    st.write("数据库与缓存位置：")
    st.code(str(config.data_dir), language=None)
    if documents.get("failed"):
        st.warning(f"有 {documents['failed']} 个文件解析失败；其他文件不受影响，可在日志中查看原因。")
    st.subheader("图片文字补充")
    st.json(status.get("images") or {})
    latest = status.get("latest_run")
    if latest:
        st.json(latest)
