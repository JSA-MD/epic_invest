#!/usr/bin/env python3
"""Local mock OpenAI verifier for the fractal-genome auto LLM review path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fractal genome search against a local mock OpenAI chat-completions server.",
    )
    parser.add_argument(
        "--search-script",
        default=str(Path(__file__).resolve().with_name("search_pair_subset_fractal_genome.py")),
    )
    parser.add_argument(
        "--summary-out",
        default="/tmp/fractal_genome_mock_openai_summary.json",
    )
    parser.add_argument(
        "--review-out",
        default="/tmp/fractal_genome_mock_openai_reviews.jsonl",
    )
    parser.add_argument(
        "--command-log",
        default="/tmp/fractal_genome_mock_openai_command.json",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    return value


def build_mock_handler(state: dict[str, Any]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw_body = self.rfile.read(length).decode("utf-8") if length else ""
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                payload = {"_raw": raw_body}

            auth_header = self.headers.get("Authorization", "")
            record = {
                "method": self.command,
                "path": self.path,
                "auth_header": auth_header,
                "body": payload,
            }
            state.setdefault("requests", []).append(record)

            if not auth_header.startswith("Bearer "):
                body = json.dumps({"error": "missing bearer auth"}).encode("utf-8")
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            review_content = json.dumps({"accepted": True, "reason": "mock_semantic_pass"})
            response = {
                "id": "mock-chatcmpl-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": review_content,
                        },
                        "finish_reason": "stop",
                    }
                ],
            }
            data = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def start_mock_server() -> tuple[ThreadingHTTPServer, threading.Thread, str, dict[str, Any]]:
    state: dict[str, Any] = {"requests": []}
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_mock_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    return server, thread, base_url, state


def run_search_with_mock_openai(
    python: str,
    search_script: str,
    summary_out: str,
    review_out: str,
    base_url: str,
) -> dict[str, Any]:
    summary_path = Path(summary_out)
    review_path = Path(review_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        search_script,
        "--pairs",
        "BTCUSDT,BNBUSDT",
        "--expert-pool-size",
        "2",
        "--population",
        "4",
        "--generations",
        "1",
        "--elite-count",
        "1",
        "--top-k",
        "1",
        "--max-depth",
        "2",
        "--logic-max-depth",
        "1",
        "--seed",
        "123",
        "--filter-mode",
        "auto",
        "--auto-llm-review-top-n",
        "1",
        "--auto-llm-review-model",
        "mock-openai",
        "--summary-out",
        str(summary_path),
        "--llm-review-out",
        str(review_path),
    ]
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["OPENAI_API_KEY"] = "local-mock-key"
    env["OPENAI_BASE_URL"] = base_url
    env["OPENAI_MODEL"] = "mock-openai"
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
    report = json.loads(summary_path.read_text())
    feature_set = report.get("search", {}).get("feature_set", {})
    feature_names = {str(item.get("feature")) for item in feature_set.get("features", [])}
    required_features = {
        "btc_momentum_1d",
        "btc_momentum_3d",
        "btc_momentum_accel_1d_3d",
        "btc_drawdown_7d",
        "breadth_change_1d",
        "regime_spread_btc_minus_bnb",
        "rel_strength_bnb_btc_3d",
    }
    missing = sorted(required_features - feature_names)
    assert not missing, f"mock OpenAI smoke is missing required expanded features: {missing}"
    assert feature_set.get("feature_context", {}).get("single_asset_mode") is False, "mock OpenAI smoke must stay multi-pair"
    auto_events = report.get("auto_llm_review_events", [])
    assert auto_events, "mock OpenAI smoke should record auto review events"
    first_event = auto_events[0]
    assert first_event.get("enabled") is True, "auto review must be enabled"
    assert first_event.get("attempted", 0) >= 1, "auto review must attempt at least one request"
    assert first_event.get("added", 0) >= 1, "auto review must add at least one reviewed candidate"

    command_log = {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "summary_path": str(summary_path),
        "review_path": str(review_path),
        "base_url": base_url,
    }
    return {
        "command": cmd,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "summary_path": str(summary_path),
        "review_path": str(review_path),
        "command_log": command_log,
        "report": report,
    }


def main() -> None:
    args = parse_args()
    server, thread, base_url, state = start_mock_server()
    try:
        result = run_search_with_mock_openai(args.python, args.search_script, args.summary_out, args.review_out, base_url)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    requests = state.get("requests", [])
    assert requests, "mock OpenAI server should receive at least one request"
    first_request = requests[0]
    body = first_request.get("body", {})
    assert first_request.get("path") == "/v1/chat/completions", "search client must target chat completions"
    assert first_request.get("auth_header", "").startswith("Bearer "), "search client must send bearer auth"
    assert body.get("model") == "mock-openai", "search client must forward the mock model"
    assert body.get("temperature") == 0, "search client must send deterministic temperature"
    response_format = body.get("response_format", {})
    assert response_format.get("type") == "json_object", "search client must request JSON object responses"
    messages = body.get("messages", [])
    assert len(messages) >= 2, "search client must send system and user messages"
    assert messages[0].get("role") == "system", "first LLM message must be system"
    assert messages[1].get("role") == "user", "second LLM message must be user"

    report = result["report"]
    auto_events = report.get("auto_llm_review_events", [])
    assert Path(args.review_out).exists(), "mock OpenAI smoke should write review prompts"

    command_log_path = Path(args.command_log)
    command_log_path.parent.mkdir(parents=True, exist_ok=True)
    command_log_path.write_text(json.dumps(json_safe(result["command_log"]), ensure_ascii=False, indent=2) + "\n")

    wrapper = {
        "mock_openai": {
            "enabled": True,
            "base_url": base_url,
            "request_count": len(requests),
            "path": first_request.get("path"),
            "auth_header": first_request.get("auth_header"),
            "first_request_body": body,
            "used_external_key": False,
        },
        "search": {
            **report.get("search", {}),
            "auto_llm_review_events": auto_events,
        },
        "selection": report.get("selection", {}),
        "top_candidates": report.get("top_candidates", []),
        "runtime": report.get("runtime", {}),
        "auto_llm_review_events": auto_events,
    }
    Path(args.summary_out).write_text(json.dumps(json_safe(wrapper), ensure_ascii=False, indent=2) + "\n")

    print(json.dumps(json_safe(wrapper), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
