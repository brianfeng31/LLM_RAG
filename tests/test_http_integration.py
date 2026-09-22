"""Exercise real HTTP serialization against a local Ollama-compatible stub."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from batch_llm_runner import OllamaClient


class HttpIntegrationTests(unittest.TestCase):
    def test_model_list_retry_and_chat_over_real_local_http(self):
        captured = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def respond(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path != "/api/tags":
                    self.respond(404, {"error": "unknown path"})
                    return
                self.respond(200, {"models": [{"name": "test-model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured.append((self.path, self.headers["Content-Type"], body))
                if len(captured) == 1:
                    self.respond(503, {"error": "temporary"})
                else:
                    self.respond(200, {"message": {"content": "Paris is the capital of France."}, "done": True})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            client = OllamaClient(host=f"http://127.0.0.1:{server.server_port}", api_key="",
                                  timeout=2, retries=1, sleeper=lambda _: None)
            client.preflight(["test-model"])
            result = client.generate("test-model", "What is the capital of France?")
            self.assertEqual(result.status, "success")
            self.assertEqual(result.attempts, 2)
            self.assertEqual(len(captured), 2)
            self.assertEqual(captured[0][0], "/api/chat")
            self.assertEqual(captured[0][1], "application/json")
            self.assertEqual(captured[0][2]["messages"][0]["content"], "What is the capital of France?")
            self.assertEqual(result.response, "Paris is the capital of France.")
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
