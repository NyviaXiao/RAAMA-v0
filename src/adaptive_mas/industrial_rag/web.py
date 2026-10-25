"""Loopback-only Chinese query workbench for an explicitly isolated financial corpus."""

from __future__ import annotations

import ipaddress
import json
import re
import socket
import threading
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Mapping
from urllib.parse import parse_qs, urlparse

from adaptive_mas.industrial_rag.models import QueryPlan
from adaptive_mas.industrial_rag.repository import IndustrialRagStore, ReleaseNotReady
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService
from adaptive_mas.runtime_limits import http_thread_limit


def _json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


class RagRequestHandler(BaseHTTPRequestHandler):
    server_version = "RAAMAIndustrialRAG/1"
    protocol_version = "HTTP/1.1"

    def _context(self) -> dict[str, object]:
        return self.server.context  # type: ignore[attr-defined]

    def _local_only(self) -> bool:
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                return False
            parsed = urlparse(f"//{self.headers.get('Host', '')}")
            if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
                return False
            parsed.port
            return True
        except ValueError:
            return False

    def _send(self, status: int, payload: object, content_type: str = "application/json; charset=utf-8") -> None:
        body = payload if isinstance(payload, bytes) else _json(payload)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict[str, object]:
        raw_length = self.headers.get("Content-Length", "0")
        if not re.fullmatch(r"\d{1,8}", raw_length):
            raise ValueError("request body length is invalid")
        length = int(raw_length)
        if length < 1 or length > 1_000_000:
            raise ValueError("request body must be between 1 byte and 1 MB")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request must be a JSON object")
        return value

    def _plan(self, payload: Mapping[str, object]) -> QueryPlan:
        allowed = {"security_code", "query", "as_of", "task_type", "accounting_basis", "report_periods", "channels", "top_k"}
        if set(payload) - allowed:
            raise ValueError("query contains fields controlled by server policy")
        channels = payload.get("channels", ["structured_facts", "bm25"])
        periods = payload.get("report_periods", [])
        if not isinstance(channels, list) or not isinstance(periods, list):
            raise ValueError("channels and report_periods must be arrays")
        return QueryPlan(
            plan_id=uuid.uuid4().hex,
            query=str(payload.get("query", "")),
            security_code=str(payload.get("security_code", "")),
            as_of=str(payload.get("as_of", "")),
            principal_id=str(self._context()["principal_id"]),
            task_type=str(payload.get("task_type", "fact")),  # type: ignore[arg-type]
            accounting_basis=str(payload["accounting_basis"]) if payload.get("accounting_basis") else None,
            report_periods=tuple(str(value) for value in periods),
            release_id=None,
            channels=tuple(str(value) for value in channels),
            top_k=int(payload.get("top_k", 10)),
            time_mode=str(self._context()["time_mode"]),  # type: ignore[arg-type]
        )

    def do_GET(self) -> None:
        if not self._local_only():
            self._send(403, {"error": "loopback clients only"})
            return
        path = urlparse(self.path).path
        context = self._context()
        if path == "/":
            content = (Path(__file__).parent / "web" / "index.html").read_bytes()
            self._send(200, content, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            store: IndustrialRagStore = context["store"]  # type: ignore[assignment]
            self._send(200, {"ok": True, "product": "industrial_financial_rag", "backends": store.backend_status(), "release": store.release_summaries(), "time_mode": context["time_mode"], "timestamp": datetime.now().astimezone().isoformat()})
            return
        if path == "/api/releases":
            self._send(200, context["store"].release_summaries())  # type: ignore[union-attr]
            return
        if path == "/api/quarantine":
            self._send(200, context["store"].quarantine_summary())  # type: ignore[union-attr]
            return
        if path.startswith("/api/original/"):
            pieces = path.removeprefix("/api/original/").split("/")
            if len(pieces) != 2 or any(not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", item) for item in pieces):
                self._send(404, {"error": "document not found"})
                return
            values = parse_qs(urlparse(self.path).query)
            try:
                plan = self._plan({
                    "security_code": values.get("security_code", [""])[0],
                    "query": values.get("query", [""])[0],
                    "as_of": values.get("as_of", [""])[0],
                })
                payload = context["service"].download_original(plan, pieces[0], pieces[1])  # type: ignore[union-attr]
            except PermissionError:
                self._send(404, {"error": "document not available"})
                return
            except ValueError as error:
                self._send(400, {"error": str(error)})
                return
            self._send(200, payload, "application/pdf" if payload.startswith(b"%PDF-") else "application/octet-stream")
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._local_only():
            self._send(403, {"error": "loopback clients only"})
            return
        path = urlparse(self.path).path
        try:
            payload = self._body()
            if path == "/api/search":
                plan = self._plan(payload)
                result = self._context()["service"].search(plan)  # type: ignore[union-attr]
                self._send(200, result)
                return
            if path == "/api/evidence":
                if set(payload) != {"security_code", "query", "as_of", "span_id", "report_periods", "accounting_basis"}:
                    raise ValueError("evidence expansion requires a cited span and the original query scope")
                plan_payload = {key: payload[key] for key in ("security_code", "query", "as_of", "report_periods", "accounting_basis") if key in payload}
                plan = self._plan(plan_payload)
                result = self._context()["service"].read_span(plan, str(payload["span_id"]))  # type: ignore[union-attr]
                self._send(200, result)
                return
            self._send(404, {"error": "not found"})
        except PermissionError:
            self._send(403, {"error": "source is not available to the current principal or time policy"})
        except ReleaseNotReady as error:
            self._send(409, {"error": str(error)})
        except FileNotFoundError:
            self._send(404, {"error": "source not found"})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send(400, {"error": str(error)})
        except Exception:
            self._send(503, {"error": "industrial RAG operation failed; see local service status"})

    def log_message(self, format: str, *args: object) -> None:
        # Query text and evidence identifiers may be sensitive; the local UI keeps no access log.
        return


class LocalRagHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler]):
        self.address_family = socket.AF_INET6 if ":" in server_address[0] else socket.AF_INET
        self._request_slots = threading.BoundedSemaphore(http_thread_limit())
        super().__init__(server_address, handler_class)

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._request_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\nContent-Length: 0\r\n\r\n"
                )
            except OSError:
                pass
            finally:
                request.close()
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._request_slots.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._request_slots.release()


def serve(
    *,
    database_path: str | Path,
    object_root: str | Path,
    host: str = "127.0.0.1",
    port: int = 8877,
    principal_id: str = "local-public",
    time_mode: str = "strict_forward",
    qwen_model_dir: str | Path | None = None,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("RAG API must bind to a loopback interface")
    if time_mode not in {"strict_forward", "retrospective_diagnostic"}:
        raise ValueError("time_mode is a fixed server policy")
    from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore

    store = IndustrialRagStore(database_path, LocalContentAddressedStore(object_root))
    channel_adapters = None
    reranker = None
    if qwen_model_dir is not None:
        from adaptive_mas.industrial_rag.qwen_models import create_local_qwen_adapters

        channel_adapters, reranker = create_local_qwen_adapters(store, qwen_model_dir)
    service = IndustrialRagService(
        store,
        channel_adapters=channel_adapters,
        reranker=reranker,
        allow_retrospective_diagnostic=time_mode == "retrospective_diagnostic",
    )
    server = LocalRagHTTPServer((host, port), RagRequestHandler)
    server.context = {"store": store, "service": service, "principal_id": principal_id, "time_mode": time_mode}
    print(f"RAAMA 金融资料工作台已启动：http://{host}:{port}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
