from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent.openvino_agent import RecallAgent
from .agent.tools import MemoryTools
from .config import load_config
from .db import open_db
from .ingestion.enrichment import ImageEnricher
from .ingestion.scanner import Scanner
from .model_download import download_models


def _runtime(config_path: str):
    config = load_config(config_path)
    return config, open_db(config.db_path)


def cmd_init(args) -> int:
    config = load_config(args.config)
    print(f"配置已就绪：{config.config_path}")
    print("资料目录只读：")
    for root in config.source_roots:
        print(f"  - {root}")
    return 0


def cmd_scan(args) -> int:
    config, conn = _runtime(args.config)
    stats = Scanner(config, conn, progress=print).scan()
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
    return 0 if stats.files_failed == 0 else 2


def cmd_search(args) -> int:
    config, conn = _runtime(args.config)
    results = MemoryTools(config, conn).search_memory(
        args.query, args.start_date, args.end_date, args.file_type, args.limit,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def cmd_enrich_images(args) -> int:
    config, conn = _runtime(args.config)
    stats = ImageEnricher(config, conn, progress=print).enrich(args.limit, args.retry_failed)
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))
    return 0 if stats.failed == 0 else 2


def cmd_ask(args) -> int:
    config, conn = _runtime(args.config)
    tools = MemoryTools(config, conn)
    agent = RecallAgent(config, tools)
    try:
        result = agent.answer(args.question)
    finally:
        agent.close()
    print(result.answer)
    labels = {
        "openvino": "LFM2.5-2.6B · OpenVINO 本地 Agent",
        "hybrid": "LFM Agent 规划 · 证据校验渲染",
        "deterministic": "确定性证据检索",
    }
    label = labels.get(result.mode, labels["deterministic"])
    print(f"回答模式：{label}", file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    config, conn = _runtime(args.config)
    print(json.dumps(MemoryTools(config, conn).status(), ensure_ascii=False, indent=2))
    return 0


def cmd_models(args) -> int:
    for path in download_models(args.config, args.profile):
        print(f"已准备 OpenVINO 模型：{path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="second-brain", description="本地个人智能第二大脑")
    parser.add_argument("--config", default="config.toml", help="配置文件路径")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="初始化配置与运行目录")
    init.set_defaults(func=cmd_init)
    scan = sub.add_parser("scan", help="只读扫描资料")
    scan.set_defaults(func=cmd_scan)
    enrich = sub.add_parser("enrich-images", help="用 OpenVINO 分批补充图片文字")
    enrich.add_argument("--limit", type=int, default=200, help="本批最多处理的图片数")
    enrich.add_argument("--retry-failed", action="store_true", help="同时重试上次失败的图片")
    enrich.set_defaults(func=cmd_enrich_images)
    search = sub.add_parser("search", help="证据检索")
    search.add_argument("query")
    search.add_argument("--start-date")
    search.add_argument("--end-date")
    search.add_argument("--file-type")
    search.add_argument("--limit", type=int, default=12)
    search.set_defaults(func=cmd_search)
    ask = sub.add_parser("ask", help="询问本地 Agent")
    ask.add_argument("question")
    ask.set_defaults(func=cmd_ask)
    status = sub.add_parser("status", help="查看索引状态")
    status.set_defaults(func=cmd_status)
    models = sub.add_parser("download-models", help="下载或转换 Intel OpenVINO 模型")
    models.add_argument("--profile", choices=("core", "audio", "semantic", "vision", "all"), default="core")
    models.set_defaults(func=cmd_models)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
