"""試験10件で、検索・振り分け・生成を別々に測り、条件ごとにMarkdownの表を出す。

    python bench.py                    # 見出し分割と固定長200字の比較(LLMなし、結果は毎回同じ)
    python bench.py --llm qwen3:1.7b   # Ollamaの生成の行を加える
    python bench.py --detail           # 1問ごとの結果も出す(失敗の切り分け用)
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from rag_agent import (
    NOT_FOUND, TOP_K, Chunker, ExtractiveGenerator, Generator, MarkdownRetriever,
    OllamaGenerator, RAGAgent, fixed_chunks, parse_sections,
)

# type: exact(ノートと同じ語) / paraphrase(別の言い方) / direct(検索不要) / unanswerable(ノートにない話題)
CASES = [
    {"id": "Q01", "type": "exact", "query": "研究室ミーティングの資料は何枚まで?", "route": "search",
     "gold": "sec_lab_meeting", "keywords": ["3枚"]},
    {"id": "Q02", "type": "exact", "query": "ゼミ発表の次の担当日はいつ?", "route": "search",
     "gold": "sec_seminar", "keywords": ["11月12日"]},
    {"id": "Q03", "type": "exact", "query": "修士論文の提出の締め切りは?", "route": "search",
     "gold": "sec_thesis_plan", "keywords": ["1月20日"]},
    {"id": "Q04", "type": "exact", "query": "学会の予稿の締め切りはいつ?", "route": "search",
     "gold": "sec_conference", "keywords": ["11月15日"]},
    {"id": "Q05", "type": "paraphrase", "query": "家庭教師で1時間にいくらもらえる?", "route": "search",
     "gold": "sec_part_time", "keywords": ["2,000円"]},
    {"id": "Q06", "type": "paraphrase", "query": "夜は何時に布団に入る?", "route": "search",
     "gold": "sec_health", "keywords": ["0時"]},
    {"id": "Q07", "type": "direct", "query": "こんにちは", "route": "direct"},
    {"id": "Q08", "type": "direct", "query": "12 * 8 は?", "route": "direct"},
    {"id": "Q09", "type": "unanswerable", "query": "旅行の予算はいくら?", "route": "search"},
    {"id": "Q10", "type": "unanswerable", "query": "飼っている猫の名前は?", "route": "search"},
]
RETRIEVAL_TYPES = ("exact", "paraphrase")


@dataclass
class Condition:
    label: str
    chunker: Chunker
    generator: Generator


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    answer = unicodedata.normalize("NFKC", answer).lower()
    return sum(unicodedata.normalize("NFKC", k).lower() in answer for k in keywords) / len(keywords)


def evaluate(cond: Condition, md: str, cases: list[dict]) -> tuple[dict, list[dict]]:
    retriever = MarkdownRetriever(cond.chunker(md))
    agent = RAGAgent(retriever, cond.generator)
    rows = []
    for c in cases:
        a = agent.answer(c["query"])
        row = {"id": c["id"], "type": c["type"], "route_ok": a.route == c["route"],
               "rank": None, "coverage": None, "answer": a.text,
               "t_retrieve": a.t_retrieve, "t_generate": a.t_generate}
        if c["type"] in RETRIEVAL_TYPES:
            # 検索はルーターや閾値と切り離して、全順位で測る
            ids = [ch.id for ch, _ in retriever.search(c["query"])]
            row["rank"] = ids.index(c["gold"]) + 1 if c["gold"] in ids else None
            row["coverage"] = keyword_coverage(a.text, c["keywords"])
        rows.append(row)

    ret = [r for r in rows if r["type"] in RETRIEVAL_TYPES]
    una = [r for r in rows if r["type"] == "unanswerable"]
    metrics = {
        "Hit@3": statistics.fmean(r["rank"] is not None and r["rank"] <= TOP_K for r in ret),
        "MRR": statistics.fmean(1 / r["rank"] if r["rank"] else 0.0 for r in ret),
        "Coverage": statistics.fmean(r["coverage"] for r in ret),
        "Routing": statistics.fmean(r["route_ok"] for r in rows),
        "Unans.": statistics.fmean(NOT_FOUND in r["answer"] for r in una),
        "検索ms": latency([r["t_retrieve"] for r in rows]),
        "生成ms": latency([r["t_generate"] for r in rows]),
    }
    return metrics, rows


def latency(ms: list[float]) -> str:
    return f"{statistics.mean(ms):.2f} / {max(ms):.2f}"


def table(header: list[str], rows: list[list]) -> str:
    cells = [header] + [[f"{v:.2f}" if isinstance(v, float) else str(v) for v in r] for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    return "\n".join([lines[0], "|" + "---|" * len(header), *lines[1:]])


def main() -> None:
    p = argparse.ArgumentParser(description="RAGエージェントの分離評価")
    p.add_argument("--note", default=Path(__file__).with_name("mock.md"))
    p.add_argument("--llm", metavar="MODEL", help="Ollamaのモデル名。指定したときだけLLMの行を加える")
    p.add_argument("--detail", action="store_true", help="1問ごとの結果も出す")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windowsでリダイレクトしても表が文字化けしないように

    md = Path(args.note).read_text(encoding="utf-8")
    conditions = [
        Condition("見出し分割 + 抽出", parse_sections, ExtractiveGenerator()),
        Condition("固定長200字 + 抽出", fixed_chunks, ExtractiveGenerator()),
    ]
    if args.llm:
        conditions.append(Condition(f"見出し分割 + LLM({args.llm})", parse_sections, OllamaGenerator(args.llm)))

    results = [(c.label, *evaluate(c, md, CASES)) for c in conditions]
    header = ["条件", *results[0][1]]
    print(table(header, [[label, *m.values()] for label, m, _ in results]))
    print(f"\n検索ms・生成msは「平均 / 最大」。Python {platform.python_version()}, {platform.system()} {platform.machine()}")

    if args.detail:
        for label, _, rows in results:
            print(f"\n### {label}\n")
            print(table(["ID", "種類", "ルート", "正解の順位", "Coverage", "回答"], [
                [r["id"], r["type"], "ok" if r["route_ok"] else "NG", r["rank"] or "-",
                 "-" if r["coverage"] is None else r["coverage"], r["answer"].replace("\n", " ")[:60]]
                for r in rows
            ]))


if __name__ == "__main__":
    main()
