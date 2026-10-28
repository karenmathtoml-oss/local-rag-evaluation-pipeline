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


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict | None  # None は、引数を読めなかったとき(不正な呼び出し)


@dataclass
class ChatTurn:
    content: str
    tool_calls: list[ToolCall]
    done_reason: str | None  # "stop" 以外("length" など)は、途中で切れた可能性がある
    prompt_tokens: int | None
    message: dict  # 会話の履歴にそのまま足す、assistant の発言
    ms: float = 0.0
    output_tokens: int | None = None
    thinking: str | None = None  # <think> の中身と、サーバが別に返した思考。採点には使わない
    truncated: bool = False  # 上限のトークン数で切れた(done_reason が length か、<think> が閉じていない)


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


def parse_args(args) -> dict | None:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return args if isinstance(args, dict) else None


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

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> ChatTurn:
        t0 = time.perf_counter()
        turn = self._chat(messages, tools)
        turn.ms = (time.perf_counter() - t0) * 1e3
        thoughts = THINK.findall(turn.content)  # [(思考, 閉じタグ)]
        thinking = "\n".join(t.strip() for t in [turn.thinking or "", *(t for t, _ in thoughts)] if t.strip())
        turn.thinking = thinking or None
        turn.truncated = turn.done_reason == "length" or (bool(thoughts) and not thoughts[-1][1])
        turn.content = THINK.sub("", turn.content).strip()
        return turn

    def generate(self, prompt: str) -> tuple[str, float, dict]:
        """回答、生成ms、メタ情報(stop、prompt_tokens、output_tokens、thinking、truncated)を返す。"""
        turn = self.chat([{"role": "user", "content": prompt}])
        meta = {"stop": turn.done_reason, "prompt_tokens": turn.prompt_tokens, "output_tokens": turn.output_tokens,
                "thinking": turn.thinking, "truncated": turn.truncated}
        return turn.content, turn.ms, meta

    def _chat(self, messages: list[dict], tools: list[dict] | None) -> ChatTurn:
        raise NotImplementedError

    def tool_message(self, call: ToolCall, content: str) -> dict:
        """道具の結果を、会話の履歴に足す形にする(形式はサーバごとに違う)。"""
        raise NotImplementedError

    def installed(self) -> bool:
        raise NotImplementedError

    def info(self) -> dict:
        """ログに残す、サーバとモデルの情報(取れるものだけ)。"""
        return {}

    def supports_tools(self) -> bool:
        raise NotImplementedError


class OllamaClient(Client):
    def _chat(self, messages: list[dict], tools: list[dict] | None) -> ChatTurn:
        options = {"temperature": self.gen["temperature"], "num_predict": self.gen["max_tokens"], "seed": self.gen["seed"]}
        if "num_ctx" in self.gen:
            options["num_ctx"] = self.gen["num_ctx"]  # 超えると、Ollama は黙ってプロンプトの先頭を切り落とす
        payload = {"model": self.spec.model, "messages": messages, "stream": False, "options": options}
        if self.spec.think is not None:
            payload["think"] = self.spec.think  # /api/chat では options ではなくトップレベルのキー
        if tools:
            payload["tools"] = tools
        r = self._request("/api/chat", payload)
        m = r["message"]
        calls = [ToolCall(c.get("id") or f"call_{i}", c["function"]["name"], parse_args(c["function"].get("arguments")))
                 for i, c in enumerate(m.get("tool_calls") or [])]
        message = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            message["tool_calls"] = m["tool_calls"]
        return ChatTurn(m.get("content") or "", calls, r.get("done_reason"), r.get("prompt_eval_count"), message,
                        output_tokens=r.get("eval_count"), thinking=m.get("thinking"))  # think = true のとき、思考は content と別に返る

    def tool_message(self, call: ToolCall, content: str) -> dict:
        return {"role": "tool", "tool_name": call.name, "content": content}

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

    def supports_tools(self) -> bool:
        return "tools" in self._request("/api/show", {"model": self.spec.model}).get("capabilities", [])


class OpenAICompatClient(Client):
    def _chat(self, messages: list[dict], tools: list[dict] | None) -> ChatTurn:
        payload = {
            "model": self.spec.model,
            "messages": messages,
            "temperature": self.gen["temperature"],
            "max_tokens": self.gen["max_tokens"],
            "seed": self.gen["seed"],
        }
        if tools:
            payload["tools"] = tools
        r = self._request("/v1/chat/completions", payload)
        choice = r["choices"][0]
        m = choice["message"]
        calls = [ToolCall(c["id"], c["function"]["name"], parse_args(c["function"].get("arguments")))
                 for c in m.get("tool_calls") or []]
        message = {"role": "assistant", "content": m.get("content") or ""}
        if m.get("tool_calls"):
            message["tool_calls"] = m["tool_calls"]
        usage = r.get("usage") or {}
        return ChatTurn(m.get("content") or "", calls, choice.get("finish_reason"), usage.get("prompt_tokens"), message,
                        output_tokens=usage.get("completion_tokens"),
                        thinking=m.get("reasoning_content"))  # llama.cpp などが思考を分けて返すとき

    def tool_message(self, call: ToolCall, content: str) -> dict:
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    def installed(self) -> bool:
        return self.spec.model in {m["id"] for m in self._request("/v1/models")["data"]}

    def supports_tools(self) -> bool:
        return True  # OpenAI互換のサーバには、確かめる共通の手段がない


CLIENTS = {"ollama": OllamaClient, "openai": OpenAICompatClient}


def make_client(spec: ModelSpec, generation: dict) -> Client:
    return CLIENTS[spec.backend](spec, generation)
