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

    def generate(self, prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        text = self._chat(prompt)
        ms = (time.perf_counter() - t0) * 1e3
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip(), ms

    def _chat(self, prompt: str) -> str:
        raise NotImplementedError

    def installed(self) -> bool:
        raise NotImplementedError


class OllamaClient(Client):
    def _chat(self, prompt: str) -> str:
        payload = {
            "model": self.spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.gen["temperature"], "num_predict": self.gen["max_tokens"],
                        "seed": self.gen["seed"]},
        }
        if self.spec.think is not None:
            payload["think"] = self.spec.think  # /api/chat では options ではなくトップレベルのキー
        return self._request("/api/chat", payload)["message"]["content"]

    def installed(self) -> bool:
        names = {m["name"] for m in self._request("/api/tags")["models"]}
        return self.spec.model in names or f"{self.spec.model}:latest" in names


class OpenAICompatClient(Client):
    def _chat(self, prompt: str) -> str:
        payload = {
            "model": self.spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.gen["temperature"],
            "max_tokens": self.gen["max_tokens"],
            "seed": self.gen["seed"],
        }
        return self._request("/v1/chat/completions", payload)["choices"][0]["message"]["content"]

    def installed(self) -> bool:
        return self.spec.model in {m["id"] for m in self._request("/v1/models")["data"]}


CLIENTS = {"ollama": OllamaClient, "openai": OpenAICompatClient}


def make_client(spec: ModelSpec, generation: dict) -> Client:
    return CLIENTS[spec.backend](spec, generation)
