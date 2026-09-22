"""Resumable Ollama batch collection. Importing this module never calls a model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_MODELS = {
    "gpt_oss_120b": "gpt-oss:120b",
    "gemma": "gemma4:31b",
    "qwen": "qwen3.5:397b",
}
FIELDS = ["request_id", "prompt", "model", "model_id", "response", "status",
          "latency_sec", "attempts", "error", "completed_at"]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_text(path, text):
    """Replace a checkpoint only after its complete contents have been flushed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, data):
    atomic_text(path, json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def read_prompts(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "prompt" not in reader.fieldnames:
            raise ValueError("Prompt CSV must contain a 'prompt' column.")
        prompts = []
        for line, row in enumerate(reader, 2):
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                raise ValueError(f"Empty prompt on CSV line {line}.")
            if prompt not in prompts:
                prompts.append(prompt)
    if not prompts:
        raise ValueError("The prompt CSV is empty.")
    return prompts


def load_models(path=None):
    models = json.loads(Path(path).read_text()) if path else DEFAULT_MODELS.copy()
    if not isinstance(models, dict) or not models:
        raise ValueError("Models must be a JSON object mapping aliases to API model IDs.")
    if any(not isinstance(k, str) or not k.strip() or not isinstance(v, str) or not v.strip()
           for k, v in models.items()):
        raise ValueError("Model aliases and API IDs must be nonempty strings.")
    if len(set(models.values())) != len(models):
        raise ValueError("Use distinct API model IDs; aliases of one model are not independent peers.")
    return models


def read_results(path):
    """Also accept the original four-column CSV, while excluding legacy error text."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"prompt", "model", "response"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Results CSV requires prompt, model, and response columns.")
        rows, seen = [], set()
        for row in reader:
            if not row.get("prompt") or not row.get("model"):
                raise ValueError("Results contain a missing prompt or model.")
            key = (row["prompt"], row["model"])
            if key in seen:
                raise ValueError(f"Duplicate result for model {row['model']!r} and the same prompt.")
            seen.add(key)
            response = row.get("response") or ""
            status = row.get("status") or ("error" if response.startswith("OLLAMA_CLOUD_ERROR:") else "success")
            if status not in {"success", "error"}:
                raise ValueError(f"Unknown result status: {status}")
            row["status"] = "error" if not response.strip() else status
            row["model_id"] = row.get("model_id") or row["model"]
            rows.append(row)
    if not rows:
        raise ValueError("Results CSV contains no responses.")
    return rows


@dataclass
class CallResult:
    response: str = ""
    status: str = "error"
    latency_sec: float = 0.0
    attempts: int = 0
    error: str = ""


class OllamaClient:
    """Small non-streaming HTTP client with bounded retries and request timeouts."""

    def __init__(self, host=None, api_key=None, timeout=60.0, retries=2,
                 temperature=0.0, max_tokens=1024, transport=None, sleeper=None):
        self.host = (host or os.getenv("OLLAMA_HOST", "https://ollama.com")).rstrip("/")
        parsed = urlparse(self.host)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("OLLAMA_HOST must be an HTTP(S) origin without credentials.")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("OLLAMA_HOST must be an origin, without a path or query string.")
        if timeout <= 0 or not math.isfinite(timeout) or not 0 <= retries <= 5:
            raise ValueError("Timeout must be positive and finite; retries must be between 0 and 5.")
        if not math.isfinite(temperature) or temperature < 0 or max_tokens <= 0:
            raise ValueError("Temperature must be nonnegative and max_tokens must be positive.")
        self.api_key = os.getenv("OLLAMA_API_KEY", "") if api_key is None else api_key
        if self.api_key and parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use HTTPS when sending an API key to a remote host.")
        self.timeout, self.retries = timeout, retries
        self.temperature, self.max_tokens = temperature, max_tokens
        self.transport, self.sleeper = transport or urlopen, sleeper or time.sleep

    def settings(self):
        return {"host": self.host, "temperature": self.temperature, "max_tokens": self.max_tokens}

    def _request(self, path, payload=None):
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = Request(self.host + path, data=data, headers=headers)
        with self.transport(request, timeout=self.timeout) as response:
            return json.load(response)

    def list_models(self):
        return sorted({item["name"] for item in self._request("/api/tags")["models"]})

    def preflight(self, models):
        if urlparse(self.host).hostname == "ollama.com" and not self.api_key:
            raise ValueError("Set OLLAMA_API_KEY for a live cloud run, or use pipeline.py --demo.")
        available = self.list_models()
        missing = sorted(set(models) - set(available))
        if missing:
            raise ValueError(f"Models not advertised by {self.host}: {missing}. Run --list-models and update your model JSON.")

    def generate(self, model, prompt, *, json_mode=False):
        if urlparse(self.host).hostname == "ollama.com" and not self.api_key:
            raise ValueError("Set OLLAMA_API_KEY before making a live cloud request.")
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
                   "stream": False, "options": {"temperature": self.temperature, "num_predict": self.max_tokens}}
        if json_mode:
            payload["format"] = "json"
        started = time.monotonic()
        error = ""
        for attempt in range(1, self.retries + 2):
            retryable, retry_after = False, None
            try:
                result = self._request("/api/chat", payload)
                if result.get("error"):
                    raise ValueError(str(result["error"]))
                if result.get("done") is False or result.get("done_reason") == "length":
                    raise ValueError("Incomplete/truncated response; increase max_tokens if needed.")
                content = result.get("message", {}).get("content")
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("API returned no answer text.")
                return CallResult(content.strip(), "success", round(time.monotonic() - started, 4), attempt)
            except HTTPError as exc:
                error = f"HTTP {exc.code}: {exc.reason}"
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                try:
                    retry_after = float(exc.headers.get("Retry-After", ""))
                except (ValueError, TypeError, AttributeError):
                    pass
            except (URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
                error, retryable = f"{type(exc).__name__}: {exc}", True
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                error = f"Invalid API response: {exc}"
            if not retryable or attempt > self.retries:
                break
            delay = retry_after if retry_after is not None and math.isfinite(retry_after) else 2 ** (attempt - 1)
            self.sleeper(max(0, min(delay, 30)))
        if self.api_key:
            error = error.replace(self.api_key, "[REDACTED]")
        return CallResult(status="error", latency_sec=round(time.monotonic() - started, 4),
                          attempts=attempt, error=error[:500])


def run_batch(prompt_csv="examples/prompts.csv", output_csv="results.csv", *, models=None,
              client=None, resume=False, retry_failed=False):
    models = models or DEFAULT_MODELS.copy()
    if len(set(models.values())) != len(models):
        raise ValueError("Each model alias must refer to a different API model ID.")
    prompts = read_prompts(prompt_csv)
    output = Path(output_csv)
    if output.resolve() == Path(prompt_csv).resolve():
        raise ValueError("Output CSV cannot overwrite the prompt input.")
    client = client or OllamaClient()
    config = {"schema": 1, "prompts": prompts, "models": models, "generation": client.settings()}
    signature = fingerprint(config)
    manifest_path = output.with_suffix(output.suffix + ".meta.json")
    rows = []
    if retry_failed and not resume:
        raise ValueError("Use --resume with --retry-failed.")
    if output.exists() or manifest_path.exists():
        if not resume:
            raise ValueError(f"{output} already has run data. Use --resume or choose a new output path.")
        if not manifest_path.exists() or json.loads(manifest_path.read_text()).get("signature") != signature:
            raise ValueError("Resume settings/input differ from this run. Choose a new output path.")
        if output.exists() and output.stat().st_size:
            with output.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            keys = [row.get("request_id") for row in rows]
            if None in keys or len(set(keys)) != len(keys):
                raise ValueError("Invalid or duplicate checkpoint IDs.")
    else:
        write_json(manifest_path, {"signature": signature, "config": config,
                                  "python": sys.version.split()[0], "created_at": datetime.now(timezone.utc).isoformat()})
    completed = {row["request_id"]: row for row in rows}
    for prompt in prompts:
        for alias, model_id in models.items():
            request_id = fingerprint([signature, prompt, alias])
            prior = completed.get(request_id)
            if prior and (prior["status"] == "success" or not retry_failed):
                continue
            result = client.generate(model_id, prompt)
            record = {"request_id": request_id, "prompt": prompt, "model": alias, "model_id": model_id,
                      **asdict(result), "completed_at": datetime.now(timezone.utc).isoformat()}
            completed[request_id] = record
            rows = list(completed.values())
            write_csv(output, rows)
            print(f"[{len(rows)}/{len(prompts) * len(models)}] {alias}: {result.status}")
    return list(completed.values())


def call_ollama_cloud(prompt, model):
    """Compatibility helper: raise on failure instead of returning an error as an answer."""
    result = OllamaClient().generate(model, prompt)
    if result.status != "success":
        raise RuntimeError(result.error)
    return result.response


def add_client_arguments(parser):
    parser.add_argument("--host", default=None, help="Ollama origin; defaults to OLLAMA_HOST or https://ollama.com")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)


def client_from_args(args):
    return OllamaClient(host=args.host, timeout=args.timeout, retries=args.retries,
                        max_tokens=args.max_tokens, temperature=args.temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="examples/prompts.csv")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--models-file", help="JSON mapping aliases to API IDs")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    add_client_arguments(parser)
    args = parser.parse_args()
    try:
        client = client_from_args(args)
        if args.list_models:
            print("\n".join(client.list_models()))
            return 0
        models = load_models(args.models_file)
        client.preflight(models.values())
        rows = run_batch(args.prompts, args.output, models=models, client=client,
                         resume=args.resume, retry_failed=args.retry_failed)
        print(f"Saved {len(rows)} responses to {args.output}")
        return int(any(row["status"] == "error" for row in rows))
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed responses are saved; resume with the same settings.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
