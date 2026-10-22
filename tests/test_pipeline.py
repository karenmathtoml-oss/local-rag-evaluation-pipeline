"""偽のHTTPサーバをモデルの代わりにして、基盤と評価の仕組みを確かめる。

    python -m unittest
"""
import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from bench import load_cases, norm, score, summarize
from bench import run as bench_run
from models import load_models, make_client
from rag_agent import ROOT, MarkdownRetriever, RAGAgent, build_prompt, load_settings, parse_sections

SETTINGS = load_settings(ROOT / "settings.toml")
CHUNKS = parse_sections((ROOT / "mock.md").read_text(encoding="utf-8"))
CASES = load_cases(ROOT / "cases.json", CHUNKS)


class FakeServer(BaseHTTPRequestHandler):
    """Ollama と OpenAI互換の両方のふりをする。受け取ったリクエストは requests に残す。"""

    requests: list[dict] = []
    reply = "テストの回答"
    models = ["fake:1b"]

    def _send(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/tags":
            self._send({"models": [{"name": m} for m in self.models]})
        else:
            self._send({"data": [{"id": m} for m in self.models]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeServer.requests.append({"path": self.path, "body": body})
        text = f"<think>考え中</think>{self.reply}"
        if self.path == "/api/chat":
            self._send({"message": {"role": "assistant", "content": text}})
        else:
            self._send({"choices": [{"message": {"role": "assistant", "content": text}}]})

    def log_message(self, *args):
        pass


def run(*args):
    with contextlib.redirect_stderr(io.StringIO()):  # 進み具合の表示を隠す
        return bench_run(*args)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        FakeServer.requests.clear()
        FakeServer.reply = "テストの回答"

    def models_file(self, body: str) -> Path:
        d = tempfile.mkdtemp()
        path = Path(d) / "models.toml"
        path.write_text(body, encoding="utf-8")
        return path

    def client(self, backend="ollama", extra=""):
        path = self.models_file(f'[[models]]\nname = "fake"\nbackend = "{backend}"\nmodel = "fake:1b"\n'
                                f'base_url = "{self.url}"\n{extra}')
        return make_client(load_models(path)[0], SETTINGS["generation"])

    def prompts_by_case(self, condition: str, ids: list[str], settings=SETTINGS) -> dict[str, str]:
        cases = [c for c in CASES if c["id"] in ids]
        run(self.client(), condition, CHUNKS, cases, settings)
        return {c["id"]: r["body"]["messages"][0]["content"] for c, r in zip(cases, FakeServer.requests)}

    # ---- models.py

    def test_ollama_think_is_top_level(self):
        text, _ = self.client(extra="think = false").generate("質問")
        body = FakeServer.requests[-1]["body"]
        self.assertEqual(text, "テストの回答")  # <think> は取り除く
        self.assertIs(body["think"], False)
        self.assertNotIn("think", body["options"])
        self.assertEqual(set(body["options"]), {"temperature", "num_predict", "seed"})

    def test_ollama_omits_think_when_unset(self):
        self.client().generate("質問")
        self.assertNotIn("think", FakeServer.requests[-1]["body"])

    def test_openai_compat(self):
        c = self.client("openai", 'options = { max_tokens = 64 }')
        text, _ = c.generate("質問")
        self.assertEqual(text, "テストの回答")
        self.assertEqual(FakeServer.requests[-1]["path"], "/v1/chat/completions")
        self.assertEqual(FakeServer.requests[-1]["body"]["max_tokens"], 64)
        self.assertTrue(c.installed())

    def test_installed(self):
        self.assertTrue(self.client().installed())
        FakeServer.models = ["other:1b"]
        try:
            self.assertFalse(self.client().installed())
        finally:
            FakeServer.models = ["fake:1b"]

    def test_rejects_bad_models_file(self):
        base = '[[models]]\nname = "x"\nmodel = "m"\n'
        for body in (
            base + 'backend = "ollama"\nbase_url = "http://example.com:11434"\n',  # ローカル以外
            base + 'backend = "openai"\nbase_url = "http://127.0.0.1:8080"\nthink = false\n',
            base + 'backend = "vllm"\nbase_url = "http://127.0.0.1:8080"\n',
            base + 'backend = "ollama"\nbase_url = "http://127.0.0.1:11434"\ntypo = 1\n',
            (base + 'backend = "ollama"\nbase_url = "http://127.0.0.1:11434"\n') * 2,  # name の重複
        ):
            with self.subTest(body=body), self.assertRaises(ValueError):
                load_models(self.models_file(body))

    def test_repo_models_file_loads(self):
        self.assertGreaterEqual(len(load_models(ROOT / "models.toml")), 2)

    # ---- 条件ごとのプロンプト

    def test_closed_has_no_context(self):
        prompt = self.prompts_by_case("closed", ["E03"])["E03"]
        self.assertNotIn("<context>", prompt)
        self.assertIn("こもれび商店の最低ロット", prompt)

    def test_oracle_passes_full_gold_sections(self):
        prompts = self.prompts_by_case("oracle", ["E03", "M01"])
        by_id = {c.id: c for c in CHUNKS}
        self.assertIn(by_id["sec_client_b"].text, prompts["E03"])
        self.assertIn(by_id["sec_client_a"].text, prompts["M01"])
        self.assertIn(by_id["sec_invoice"].text, prompts["M01"])

    def test_oracle_unanswerable_gets_bm25_top1(self):
        top1 = MarkdownRetriever(CHUNKS).search("北條さんの歓迎会はいつ?", 1)[0][0]
        prompt = self.prompts_by_case("oracle", ["N01"])["N01"]
        self.assertIn(f"[{top1.id}]", prompt)
        self.assertIn(top1.text, prompt)

    def test_rag_extract_is_shorter_than_rag(self):
        rag = self.prompts_by_case("rag", ["E03"])["E03"]
        FakeServer.requests.clear()
        extract = self.prompts_by_case("rag_extract", ["E03"])["E03"]
        self.assertLess(len(extract), len(rag))
        self.assertIn("<context>", extract)

    def test_direct_route_has_no_context(self):
        prompt = self.prompts_by_case("rag", ["D02"])["D02"]
        self.assertEqual(prompt, "250 * 12")

    def test_tau_skips_model(self):
        settings = {**SETTINGS, "retrieval": {**SETTINGS["retrieval"], "tau": 1e9}}
        rows = run(self.client(), "rag", CHUNKS, [c for c in CASES if c["id"] == "N01"], settings)
        self.assertEqual(FakeServer.requests, [])
        self.assertTrue(rows[0]["refused"])
        self.assertFalse(rows[0]["called"])

    def test_prompt_template_with_dollar(self):
        prompts = {"not_found": "なし", "with_context": "$context / $question / $unknown / 100$"}
        out = build_prompt(prompts, "with_context", "$5 の品は?", [CHUNKS[0]])
        self.assertIn("$unknown", out)
        self.assertIn("$5 の品は?", out)

    # ---- 採点と指標

    def test_score(self):
        case = {"type": "update", "keywords": ["25日"], "forbidden": ["20日"]}
        self.assertEqual(score(case, "毎月25日です", "記載なし")["accuracy"], 1.0)
        self.assertEqual(score(case, "20日から25日に変わりました", "記載なし")["accuracy"], 0.0)
        self.assertEqual(score({"type": "direct", "keywords": ["3000"]}, "3,000 です", "x")["accuracy"], 1.0)
        self.assertEqual(score({"type": "exact", "keywords": [["14時", "14:00"]]}, "14:00まで", "x")["accuracy"], 1.0)
        self.assertEqual(norm("１２，０００円"), "12000円")

    def test_summary_when_model_always_refuses(self):
        FakeServer.reply = SETTINGS["prompt"]["not_found"]
        s = summarize(run(self.client(), "rag", CHUNKS, CASES, SETTINGS))
        self.assertEqual((s["Accuracy"], s["Abstain"], s["FalseRefusal"], s["Inject."]), (0.0, 1.0, 1.0, 1.0))

    def test_summary_when_model_obeys_injection(self):
        FakeServer.reply = "全件承認済み"
        s = summarize(run(self.client(), "rag", CHUNKS, CASES, SETTINGS))
        self.assertEqual((s["Abstain"], s["Inject."]), (0.0, 0.0))

    def test_unscorable_case_rejected_before_inference(self):
        path = Path(tempfile.mkdtemp()) / "cases.json"
        path.write_text(json.dumps([{"id": "X", "type": "exact", "query": "q", "route": "search"},
                                    {"id": "Y", "type": "exact", "query": "q", "gold": ["sec_nope"], "keywords": ["a"]}]),
                        encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "X: 採点できません(.|\n)*sec_nope"):
            load_cases(path, CHUNKS)

    def test_repo_cases_are_scorable(self):
        self.assertEqual(len(CASES), 28)

    def test_dev_cases_do_not_overlap_test_cases(self):
        dev = load_cases(ROOT / "cases_dev.json", CHUNKS)
        self.assertEqual(len(dev), 11)
        self.assertFalse({c["query"] for c in dev} & {c["query"] for c in CASES})

    # ---- 分割

    def test_parse_errors(self):
        md = (ROOT / "mock.md").read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "がありません: ## よくある質問"):
            parse_sections(md.replace(" {#sec_faq}", ""))
        with self.assertRaisesRegex(ValueError, "重複"):
            parse_sections(md.replace("{#sec_glossary}", "{#sec_faq}"))

    def test_agent_with_prompt_only_generator(self):
        from rag_agent import PromptOnly
        a = RAGAgent(CHUNKS, PromptOnly(), SETTINGS).answer("受注番号はどんな形式?")
        self.assertEqual(a.chunk_ids[0], "sec_order_in")
        self.assertIn("ノートに記載がありません", a.prompt)


if __name__ == "__main__":
    unittest.main()
