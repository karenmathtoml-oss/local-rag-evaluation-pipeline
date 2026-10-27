"""models.toml に書いたローカルのモデルを呼ぶ。Ollama(/api/chat)と OpenAI互換(/v1/chat/completions)に対応。

通信先は、ノートの中身を外に出さないよう、ローカルのホストに限る。
"""
from __future__ import annotations

import json
import re
import time
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
# 閉じタグのない <think> (上限のトークン数で切れた思考)も末尾まで取り除く。残すと、思考の中の語で正解と採点されてしまう
THINK = re.compile(r"<think>(.*?)(</think>|$)", re.S)


@dataclass
class ModelSpec:
    name: str
    backend: str  # "ollama" | "openai"
    model: str
    base_url: str
    think: bool | None = None  # Ollama だけ。None なら送らない
    options: dict = field(default_factory=dict)  # settings.toml の [generation] をモデルごとに上書きする
    enabled: bool = True


def load_models(path: str | Path) -> list[ModelSpec]:
    """enabled = false のモデルは除いて返す。書き間違いは、推論を始める前にエラーにする。"""
    with open(path, "rb") as f:
        entries = tomllib.load(f).get("models", [])
    specs = []
    for i, e in enumerate(entries, 1):
        try:
            spec = ModelSpec(**e)
        except TypeError as err:
            raise ValueError(f"{path} の {i} 番目のモデル: {err}") from err
        if spec.backend not in CLIENTS:
            raise ValueError(f"{spec.name}: backend は {' / '.join(CLIENTS)} のどれかにしてください")
        if urlsplit(spec.base_url).hostname not in LOCAL_HOSTS:
            raise ValueError(f"{spec.name}: base_url はローカル({', '.join(sorted(LOCAL_HOSTS))})に限ります")
        if spec.think is not None and spec.backend != "ollama":
            raise ValueError(f"{spec.name}: think は backend = \"ollama\" のときだけ使えます")
        specs.append(spec)
    names = [s.name for s in specs]
    if dup := {n for n in names if names.count(n) > 1}:
        raise ValueError(f"name が重複しています: {', '.join(sorted(dup))}")
    return [s for s in specs if s.enabled]


class Client:
    def __init__(self, spec: ModelSpec, generation: dict):
        self.spec = spec
        self.gen = {**generation, **spec.options}
        self.base = spec.base_url.rstrip("/")

    def _request(self, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.gen["timeout"]) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{self.spec.name}: {self.base + path} が {e.code} を返しました: "
                               f"{e.read().decode(errors='replace')[:200]}") from e
        except OSError as e:
            raise RuntimeError(f"{self.spec.name}: {self.base} に接続できません。サーバが動いているか確認してください: {e}") from e

    def generate(self, prompt: str) -> tuple[str, float, dict]:
        """回答、生成ms、メタ情報(stop、prompt_tokens、output_tokens、thinking、truncated)を返す。"""
        t0 = time.perf_counter()
        text, meta = self._chat(prompt)
        ms = (time.perf_counter() - t0) * 1e3
        thoughts = THINK.findall(text)  # [(思考, 閉じタグ)]
        thinking = "\n".join(t.strip() for t in [meta.get("thinking") or "", *(t for t, _ in thoughts)] if t.strip())
        unclosed = bool(thoughts) and not thoughts[-1][1]
        meta.update(thinking=thinking or None, truncated=meta.get("stop") == "length" or unclosed)
        return THINK.sub("", text).strip(), ms, meta

    def warmup(self) -> float:
        """モデルをサーバに読み込ませる(1トークンだけ生成し、採点しない)。かかったmsを返す。
        読み込みの時間が、1問目の生成msに入らないようにするため。"""
        gen, self.gen = self.gen, {**self.gen, "max_tokens": 1}
        try:
            return self.generate("こんにちは")[1]
        finally:
            self.gen = gen

    def _chat(self, prompt: str) -> tuple[str, dict]:
        raise NotImplementedError

    def installed(self) -> bool:
        raise NotImplementedError

    def info(self) -> dict:
        """ログに残す、サーバとモデルの情報(取れるものだけ)。"""
        return {}


class OllamaClient(Client):
    def _chat(self, prompt: str) -> tuple[str, dict]:
        payload = {
            "model": self.spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.gen["temperature"], "num_predict": self.gen["max_tokens"],
                        "seed": self.gen["seed"]},
        }
        if "num_ctx" in self.gen:
            payload["options"]["num_ctx"] = self.gen["num_ctx"]
        if self.spec.think is not None:
            payload["think"] = self.spec.think  # /api/chat では options ではなくトップレベルのキー
        r = self._request("/api/chat", payload)
        return r["message"]["content"], {
            "stop": r.get("done_reason"),
            "prompt_tokens": r.get("prompt_eval_count"),  # キャッシュが効くと、実際より少なく出る
            "output_tokens": r.get("eval_count"),
            "thinking": r["message"].get("thinking"),  # think = true のとき、content とは別に返る
            # サーバが測った時間(ナノ秒をミリ秒に)。eval_count / eval_ms でトークン/秒が出せる
            **{f"{k}_ms": r[f"{k}_duration"] / 1e6 for k in ("load", "prompt_eval", "eval") if f"{k}_duration" in r},
        }

    def _tag(self) -> dict | None:
        tags = {m["name"]: m for m in self._request("/api/tags")["models"]}
        return tags.get(self.spec.model) or tags.get(f"{self.spec.model}:latest")

    def installed(self) -> bool:
        return self._tag() is not None

    def info(self) -> dict:
        tag = self._tag() or {}
        details = tag.get("details", {})
        return {"server_version": self._request("/api/version").get("version"), "digest": tag.get("digest"),
                "parameter_size": details.get("parameter_size"), "quantization": details.get("quantization_level")}


class OpenAICompatClient(Client):
    def _chat(self, prompt: str) -> tuple[str, dict]:
        payload = {
            "model": self.spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.gen["temperature"],
            "max_tokens": self.gen["max_tokens"],
            "seed": self.gen["seed"],
        }
        r = self._request("/v1/chat/completions", payload)
        choice, usage = r["choices"][0], r.get("usage") or {}
        return choice["message"].get("content") or "", {
            "stop": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "thinking": choice["message"].get("reasoning_content"),  # llama.cpp などが思考を分けて返すとき
        }

    def installed(self) -> bool:
        return self.spec.model in {m["id"] for m in self._request("/v1/models")["data"]}


CLIENTS = {"ollama": OllamaClient, "openai": OpenAICompatClient}


def make_client(spec: ModelSpec, generation: dict) -> Client:
    return CLIENTS[spec.backend](spec, generation)
