"""Markdownノートに対するRAGの基盤(分割、BM25、ルーター、文脈の圧縮、プロンプト)。

評価の対象ではなく、モデルを同じ条件で比べるための土台。設定は settings.toml から読む。
    python rag_agent.py "受注番号はどんな形式?"                      # モデルに送るプロンプトを表示するだけ
    python rag_agent.py "受注番号はどんな形式?" --model qwen3-1.7b   # models.toml の名前で1問だけ聞く
"""
from __future__ import annotations

import argparse
import math
import re
import string
import tomllib
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).parent

H2 = re.compile(r"^##\s")
HEADING = re.compile(r"^##\s+(.+?)\s*\{#([a-z0-9_]+)\}\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str


@dataclass
class Answer:
    text: str
    route: str  # "search" | "direct"
    chunk_ids: list[str]
    prompt: str | None  # None は、τで足切りしてモデルを呼ばなかったとき
    t_generate: float  # ミリ秒
    meta: dict = field(default_factory=dict)  # 停止の理由やトークン数(models.Client.generate を参照)


class Generator(Protocol):
    def generate(self, prompt: str) -> tuple[str, float, dict]: ...


def load_settings(path: str | Path = ROOT / "settings.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def parse_sections(md: str) -> list[Chunk]:
    """`## 見出し {#id}` の節を1チャンクにする。`###` 以下は親の本文に含める。"""
    chunks: list[Chunk] = []
    seen: set[str] = set()
    head: tuple[str, str] | None = None
    body: list[str] = []
    in_code = False

    def flush() -> None:
        if head:
            chunks.append(Chunk(head[1], head[0], "\n".join(body).strip()))

    for lineno, line in enumerate(md.splitlines(), 1):
        if FENCE.match(line):
            in_code = not in_code
        if in_code or not H2.match(line):
            body.append(line)
            continue
        m = HEADING.match(line)
        if not m:
            raise ValueError(f"{lineno}行目: 見出しにID({{#sec_xxx}})がありません: {line}")
        if m[2] in seen:
            raise ValueError(f"{lineno}行目: IDが重複しています: {m[2]}")
        flush()
        seen.add(m[2])
        head, body = (m[1], m[2]), []
    flush()
    if not chunks:
        raise ValueError("`## 見出し {#id}` の節が1つもありません")
    return chunks


def bigrams(text: str) -> list[str]:
    s = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).lower())
    return [s[i:i + 2] for i in range(len(s) - 1)] or ([s] if s else [])


class MarkdownRetriever:
    """文字2-gramのBM25。"""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tfs = [Counter(bigrams(f"{c.title}\n{c.text}")) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(chunks)
        self.idf = {t: math.log((n - d + 0.5) / (d + 0.5) + 1) for t, d in df.items()}

    def search(self, query: str, k: int | None = None) -> list[tuple[Chunk, float]]:
        terms = set(bigrams(query))
        scored = []
        for c, tf, dl in zip(self.chunks, self.tfs, self.lens):
            norm = self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            s = sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)
            scored.append((c, s))
        scored.sort(key=lambda x: x[1], reverse=True)  # 同点は元の順(安定ソート)
        return scored[:k] if k else scored


class QueryRouter:
    """挨拶、お礼、自己紹介、計算式は "direct"。それ以外は "search"。"""

    DIRECT = [re.compile(p, re.I) for p in (
        r"^(こんにちは|こんばんは|おはよう(ございます)?|はじめまして|ありがとう(ございます)?|hello|hi|thanks?)[!。.、\s]*$",
        r"^(あなた|君)は(誰|だれ|何者)|自己紹介",
        r"^\s*\d+\s*[-+*/]\s*\d+\s*(=|は)?\s*(いくつ|何)?\s*\??\s*$",
    )]

    def route(self, query: str) -> str:
        q = unicodedata.normalize("NFKC", query).strip()
        return "direct" if any(p.search(q) for p in self.DIRECT) else "search"


def split_sentences(text: str) -> list[str]:
    sents = [s.strip().lstrip("-*#").strip() for s in re.split(r"(?<=。)|\n", text)]
    return [s for s in sents if s and not FENCE.match(s)]


class SentenceExtractor:
    """軽量モデル向けの文脈の圧縮。各節から、質問と2-gramが多く重なる文を、元の順のまま残す。"""

    def __init__(self, max_sentences: int):
        self.max_sentences = max_sentences

    def compress(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        q = set(bigrams(query))
        out = []
        for c in chunks:
            sents = split_sentences(c.text)
            ranked = sorted(range(len(sents)), key=lambda i: -len(q & set(bigrams(sents[i]))))
            out.append(Chunk(c.id, c.title, "\n".join(sents[i] for i in sorted(ranked[:self.max_sentences]))))
        return out


def build_prompt(prompts: dict, kind: str, question: str, chunks: list[Chunk] = ()) -> str:
    """kind は settings.toml の [prompt] のキー(with_context / closed / direct)。"""
    context = "\n\n".join(f"[{c.id}] {c.title}\n{c.text}" for c in chunks)
    return string.Template(prompts[kind]).safe_substitute(
        question=question, context=context, not_found=prompts["not_found"])


class RAGAgent:
    def __init__(self, chunks: list[Chunk], generator: Generator, settings: dict, extract: bool = False):
        self.retriever, self.router, self.generator = MarkdownRetriever(chunks), QueryRouter(), generator
        self.k, self.tau = settings["retrieval"]["top_k"], settings["retrieval"]["tau"]
        self.prompts = settings["prompt"]
        self.extractor = SentenceExtractor(settings["extract"]["max_sentences"]) if extract else None

    def answer(self, query: str, chunks: list[Chunk] | None = None, closed: bool = False) -> Answer:
        """chunks を渡すと、検索も圧縮もせずにその節をそのまま使う。closed なら文脈を付けない。"""
        route = self.router.route(query)
        if route == "direct" or closed:
            return self._ask("direct" if route == "direct" else "closed", query, [], route)
        if chunks is None:
            hits = self.retriever.search(query, self.k)
            if not hits or hits[0][1] < self.tau:
                return Answer(self.prompts["not_found"], route, [c.id for c, _ in hits], None, 0.0)
            chunks = [c for c, _ in hits]
            if self.extractor:
                chunks = self.extractor.compress(query, chunks)
        return self._ask("with_context", query, chunks, route)

    def _ask(self, kind: str, query: str, chunks: list[Chunk], route: str) -> Answer:
        prompt = build_prompt(self.prompts, kind, query, chunks)
        text, ms, meta = self.generator.generate(prompt)
        return Answer(text, route, [c.id for c in chunks], prompt, ms, meta)


class PromptOnly:
    """モデルを呼ばずに、送るはずのプロンプトを返す(--model を省いたときの確認用)。"""

    def generate(self, prompt: str) -> tuple[str, float, dict]:
        return prompt, 0.0, {}


def main() -> None:
    from models import load_models, make_client

    p = argparse.ArgumentParser(description="ノートに対して1問だけ質問する")
    p.add_argument("query")
    p.add_argument("--model", help="models.toml の name。省略すると、プロンプトを表示するだけ")
    p.add_argument("--extract", action="store_true", help="検索した節を、文の抽出で圧縮してから渡す")
    p.add_argument("--settings", default=ROOT / "settings.toml")
    p.add_argument("--models-file", default=ROOT / "models.toml")
    args = p.parse_args()

    settings = load_settings(args.settings)
    if args.model:
        specs = {s.name: s for s in load_models(args.models_file)}
        if args.model not in specs:
            p.error(f"models.toml に {args.model} がありません(有効なもの: {', '.join(specs) or 'なし'})")
        generator = make_client(specs[args.model], settings["generation"])
    else:
        generator = PromptOnly()
    chunks = parse_sections((ROOT / settings["bench"]["note"]).read_text(encoding="utf-8"))
    a = RAGAgent(chunks, generator, settings, extract=args.extract).answer(args.query)
    print(f"route : {a.route}")
    print(f"chunks: {', '.join(a.chunk_ids) or '-'}")
    if args.model:
        print(f"answer: {a.text}")
        print(f"time  : {a.t_generate:.0f}ms")
        if a.meta.get("truncated"):
            print("note  : 上限のトークン数で回答が切れました(max_tokens を増やしてください)")
    else:
        print(f"prompt:\n{a.prompt or '(τで足切り。モデルは呼ばない)'}")


if __name__ == "__main__":
    main()
