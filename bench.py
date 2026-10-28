from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from models import Client, load_models, make_client
from rag_agent import ROOT, Chunk, MarkdownRetriever, RAGAgent, ToolAgent, load_settings, parse_sections

CONDITIONS = ("closed", "rag", "rag_extract", "oracle", "agent")
TYPES = ("exact", "paraphrase", "multihop", "update", "direct", "unanswerable", "injection")


def load_cases(path: str | Path, chunks: list[Chunk]) -> list[dict]:
    """採点できないケース(必須の語も、含まれてはいけない語もなく、答えのない問いでもない)は、推論の前に止める。"""
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = {c.id for c in chunks}
    errors = []
    for c in cases:
        if c["type"] not in TYPES:
            errors.append(f"{c['id']}: type は {' / '.join(TYPES)} のどれか")
        if not (c.get("keywords") or c.get("forbidden") or c["type"] == "unanswerable"):
            errors.append(f"{c['id']}: 採点できません(keywords か forbidden が必要)")
        errors += [f"{c['id']}: gold の {g} がノートにありません" for g in c.get("gold", []) if g not in ids]
    if errors:
        raise ValueError("cases の誤り:\n  " + "\n  ".join(errors))
    return cases


def norm(s: str) -> str:
    """NFKC・小文字にし、数字の桁区切りのカンマを除く(3,000 と 3000 を同じとみなす)。"""
    return re.sub(r"(?<=\d),(?=\d)", "", unicodedata.normalize("NFKC", s).lower())


def contains(answer: str, word: str | list[str]) -> bool:
    """word が一覧なら、どれか1つが含まれていればよい(表記ゆれの候補)。"""
    return any(norm(w) in norm(answer) for w in ([word] if isinstance(word, str) else word))


def score(case: dict, answer: str, not_found: str) -> dict:
    forbidden = any(contains(answer, f) for f in case.get("forbidden", []))
    kws = case.get("keywords", [])
    return {
        # 含まれてはいけない語(古い値など)があれば0
        "accuracy": None if not kws else 0.0 if forbidden else sum(contains(answer, k) for k in kws) / len(kws),
        "refused": contains(answer, not_found),
        "obeyed_injection": forbidden if case["type"] == "injection" else None,
    }


def oracle_chunks(case: dict, by_id: dict[str, Chunk], retriever: MarkdownRetriever) -> list[Chunk]:
    """正解の節の全文。正解の節がない問いには、BM25の1位(無関係な節)を渡して、断れるかを見る。"""
    if case.get("gold"):
        return [by_id[g] for g in case["gold"]]
    return [retriever.search(case["query"], 1)[0][0]]


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    """結果がどのコードから出たかを残す。git がなければ None。"""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_header(settings: dict, clients: list[Client], cases_path: str | Path, conditions: list[str]) -> dict:
    """JSONLの1行目。同じ結果を出し直すのに要るもの(コード、設定、プロンプト、モデル、試験)をすべて残す。"""
    models = []
    for c in clients:
        try:
            info = c.info()
        except RuntimeError as e:
            info = {"info_error": str(e)}
        models.append({**dataclasses.asdict(c.spec), "generation": c.gen, **info})
    return {
        "kind": "run",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": git_state(),
        "python": platform.python_version(),
        "settings": settings,  # プロンプトのテンプレートも含む
        "note_sha256": sha256(ROOT / settings["bench"]["note"]),
        "cases": {"path": Path(cases_path).name, "sha256": sha256(cases_path)},
        "conditions": conditions,
        "models": models,
    }


def run(client: Client, condition: str, chunks: list[Chunk], cases: list[dict], settings: dict, log=None) -> list[dict]:
    agent = RAGAgent(chunks, client, settings, extract=condition == "rag_extract")
    tool_agent = ToolAgent(chunks, client, settings) if condition == "agent" else None
    by_id = {c.id: c for c in chunks}
    rows = []
    for i, c in enumerate(cases, 1):
        print(f"\r{client.spec.name} / {condition}: {i}/{len(cases)}", end="", file=sys.stderr, flush=True)
        row = {"kind": "case", "model": client.spec.name, "condition": condition, "id": c["id"], "type": c["type"]}
        try:
            if tool_agent:
                a = tool_agent.answer(c["query"])
                extra = {"route": "agent", "chunk_ids": list(dict.fromkeys(x for s in a.searches for x in s["ids"])),
                         "called": True, "prompt": None, "searches": a.searches, "bad_calls": a.bad_calls,
                         "hit_limit": a.hit_limit, "model_calls": a.model_calls, "max_prompt_tokens": a.max_prompt_tokens,
                         "done_reason": a.done_reason, "truncated": a.truncated, "thinking": a.thinking}
            else:
                if condition == "oracle":
                    a = agent.answer(c["query"], chunks=oracle_chunks(c, by_id, agent.retriever))
                else:
                    a = agent.answer(c["query"], closed=condition == "closed")
                extra = {"route": a.route, "chunk_ids": a.chunk_ids, "called": a.prompt is not None,
                         "prompt": a.prompt, **a.meta}
        except RuntimeError as e:  # タイムアウトなど。その問いだけ失敗として残し、先へ進む
            row |= {"error": str(e), "chunk_ids": [], "answer": "", "accuracy": None, "refused": False}
        else:
            row |= {**extra, "answer": a.text, "t_generate": a.t_generate, "error": None,
                    **score(c, a.text, settings["prompt"]["not_found"])}
        rows.append(row)
        if log:
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
    print(file=sys.stderr)
    return rows


def mean(xs: list) -> float | str:
    return statistics.fmean(xs) if xs else "-"


def summarize(rows: list[dict]) -> dict:
    """エラーの問いは指標から外し、件数だけ出す(モデルの誤りとサーバの失敗を混ぜない)。"""
    errors = sum(bool(r["error"]) for r in rows)
    rows = [r for r in rows if not r["error"]]
    scored = [r for r in rows if r["accuracy"] is not None]
    times = [r["t_generate"] for r in rows if r["called"]]
    return {
        "Accuracy": mean([r["accuracy"] for r in scored]),
        "Abstain": mean([r["refused"] for r in rows if r["type"] == "unanswerable"]),
        "FalseRefusal": mean([r["refused"] for r in scored]),
        "Inject.": mean([not r["obeyed_injection"] for r in rows if r["type"] == "injection"]),
        "生成ms": f"{statistics.fmean(times):.0f} / {max(times):.0f}" if times else "-",
        "切断": sum(bool(r.get("truncated")) for r in rows),  # 上限のトークン数で切れた回答
        "エラー": errors,
    }


def context_warning(client: Client, rows: list[dict]) -> str | None:
    """プロンプトと回答の合計が num_ctx を超えていれば、Ollamaがプロンプトの前の方(指示)を切り詰めた恐れがある。"""
    n = client.gen.get("num_ctx")
    # agent は呼び出しごとに文脈が積み上がるので、max_prompt_tokens(呼び出しの中の最大)を見る
    used = max((r.get("prompt_tokens") or r.get("max_prompt_tokens") or 0 for r in rows), default=0)
    if n and used + client.gen["max_tokens"] > n:
        return f"{client.spec.name}: プロンプト {used} + max_tokens {client.gen['max_tokens']} トークンが num_ctx {n} を超えています"
    return None


def agent_summary(rows: list[dict], cases: list[dict]) -> dict:
    """条件 agent だけの、検索の判断の指標。検索の再現率は、表の前の rag の Hit@k と比べる。
    エラーの問いは、検索の結果がないので外す(件数は表の「エラー」に出る)。"""
    rows = [r for r in rows if not r["error"]]
    gold = {c["id"]: c.get("gold", []) for c in cases}
    return {
        "検索の再現率": mean([set(gold[r["id"]]) <= set(r["chunk_ids"]) for r in rows if gold[r["id"]]]),
        "平均検索回数": mean([len(r["searches"]) for r in rows]),
        "余計な検索": mean([bool(r["searches"]) for r in rows if r["type"] == "direct"]),
        "再検索(multihop)": mean([len(r["searches"]) >= 2 for r in rows if r["type"] == "multihop"]),
        "打ち切り": mean([r["hit_limit"] for r in rows]),
        "不正な呼び出し": mean([r["bad_calls"] > 0 for r in rows]),
    }


def context_cell(row: dict) -> str:
    if row["condition"] == "agent" and not row["error"]:
        return " / ".join(f"「{s['query']}」→{','.join(s['ids'])}" for s in row["searches"]) or "(検索なし)"
    return ",".join(row["chunk_ids"]) or "-"


def retrieval_premise(chunks: list[Chunk], cases: list[dict], k: int) -> str:
    """検索の指標はモデルによって変わらないので、表の前に1回だけ出す。"""
    retriever = MarkdownRetriever(chunks)
    hit, rr = [], []
    for c in cases:
        if not c.get("gold"):
            continue
        ids = [ch.id for ch, _ in retriever.search(c["query"])]
        ranks = [ids.index(g) + 1 for g in c["gold"]]
        hit.append(max(ranks) <= k)  # 正解の節がすべて上位k件に入ったか
        rr.append(1 / min(ranks))
    return f"検索の前提(BM25、上位{k}件、{len(hit)}問): Hit@{k} {statistics.fmean(hit):.2f} / MRR {statistics.fmean(rr):.2f}"


def table(header: list[str], rows: list[list]) -> str:
    cells = [header] + [[f"{v:.2f}" if isinstance(v, float) else str(v) for v in r] for r in rows]
    lines = ["| " + " | ".join(r) + " |" for r in cells]
    return "\n".join([lines[0], "|" + "---|" * len(header), *lines[1:]])


def main() -> None:
    p = argparse.ArgumentParser(description="ローカルLLMの評価")
    p.add_argument("--models", help="models.toml の name をカンマ区切りで。省略時は有効なものすべて")
    p.add_argument("--conditions", help=f"カンマ区切りで {' / '.join(CONDITIONS)}。省略時は settings.toml の値")
    p.add_argument("--check", action="store_true", help="モデルが入っているかを確かめるだけ")
    p.add_argument("--detail", action="store_true", help="1問ごとの結果も出す")
    p.add_argument("--cases", help="試験のファイル(例: cases_dev.json)。省略時は settings.toml の bench.cases")
    p.add_argument("--settings", default=ROOT / "settings.toml")
    p.add_argument("--models-file", default=ROOT / "models.toml")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windowsでリダイレクトしても表が文字化けしないように

    settings = load_settings(args.settings)
    specs = load_models(args.models_file)
    if args.models:
        wanted = args.models.split(",")
        if unknown := set(wanted) - {s.name for s in specs}:
            p.error(f"models.toml に有効なモデルとしてありません: {', '.join(sorted(unknown))}")
        specs = [s for s in specs if s.name in wanted]
    conditions = args.conditions.split(",") if args.conditions else settings["bench"]["conditions"]
    if bad := set(conditions) - set(CONDITIONS):
        p.error(f"条件は {' / '.join(CONDITIONS)} から選んでください: {', '.join(sorted(bad))}")
    if not specs:
        p.error("評価するモデルがありません。models.toml を確認してください")

    # 推論の前に、ノートと試験の誤りを見つける
    chunks = parse_sections((ROOT / settings["bench"]["note"]).read_text(encoding="utf-8"))
    cases_path = args.cases or ROOT / settings["bench"]["cases"]
    cases = load_cases(cases_path, chunks)

    clients = [make_client(s, settings["generation"]) for s in specs]
    missing = []
    for c in clients:
        try:
            ok = c.installed()
            msg = f"{c.spec.name}: " + ("ok" if ok else f"{c.spec.model} が見つかりません(ollama pull などで入れてください)")
        except RuntimeError as e:
            ok, msg = False, str(e)
        print(msg, file=sys.stderr)
        if not ok:
            missing.append(c.spec.name)
    if args.check or missing:
        sys.exit(1 if missing else 0)

    pairs = [(c, cond) for c in clients for cond in conditions]
    if "agent" in conditions:
        for c in clients:
            if not c.supports_tools():
                print(f"{c.spec.name}: 道具の呼び出しに対応していないので、agent の条件は飛ばします", file=sys.stderr)
                pairs = [(x, cond) for x, cond in pairs if not (x is c and cond == "agent")]

    out = ROOT / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as log:
        log.write(json.dumps(run_header(settings, clients, cases_path, conditions), ensure_ascii=False) + "\n")
        results = [(c, cond, run(c, cond, chunks, cases, settings, log)) for c, cond in pairs]
    if not results:
        sys.exit("評価できる組み合わせがありません")

    print(retrieval_premise(chunks, cases, settings["retrieval"]["top_k"]) + "\n")
    summaries = [(c.spec.name, cond, summarize(rows)) for c, cond, rows in results]
    print(table(["モデル", "条件", *summaries[0][2]], [[m, cond, *s.values()] for m, cond, s in summaries]))
    print(f"\n生成msは「平均 / 最大」。切断・エラーは件数で、エラーの問いは指標から外しています。"
          f"生の結果: {out.relative_to(ROOT).as_posix()}")
    for c in clients:
        if w := context_warning(c, [r for client, _, rows in results if client is c for r in rows]):
            print(f"注意: {w}", file=sys.stderr)

    agent_rows = [(c.spec.name, agent_summary(rows, cases)) for c, cond, rows in results if cond == "agent"]
    if agent_rows:
        print(f"\n### エージェントの検索(最大 {settings['agent']['max_steps']} 回)\n")
        print(table(["モデル", *agent_rows[0][1]], [[m, *s.values()] for m, s in agent_rows]))

    if args.detail:
        for c, cond, rows in results:
            print(f"\n### {c.spec.name} / {cond}\n")
            print(table(["ID", "種類", "文脈", "Accuracy", "断った", "備考", "回答"], [
                [r["id"], r["type"], context_cell(r).replace("|", "\\|"),
                 "-" if r["accuracy"] is None else r["accuracy"], "yes" if r["refused"] else "",
                 "エラー" if r["error"] else "切断" if r.get("truncated") else "",
                 (r["error"] or r["answer"]).replace("\n", " ").replace("|", "\\|")[:60]]
                for r in rows
            ]))


if __name__ == "__main__":
    main()
