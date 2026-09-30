"""Real-Time Server Request Logs & Live Telemetry Stream Endpoints.

Provides:
- GET /logs, /server/logs: Interactive Dark-Mode Real-Time Web Dashboard.
- GET /api/logs: Recent request records and live telemetry metrics in JSON.
- GET /api/logs/stream: Server-Sent Events (SSE) stream for instant real-time telemetry.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncGenerator

try:
    from fastapi import APIRouter, Header, HTTPException, Query, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
except (ImportError, ModuleNotFoundError):
    class APIRouter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.routes = []
        def get(self, *args: Any, **kwargs: Any) -> Any:
            return lambda f: f
        def post(self, *args: Any, **kwargs: Any) -> Any:
            return lambda f: f

    def Query(default: Any = None, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return default

    def Header(default: Any = None, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return default

    class HTTPException(Exception):  # type: ignore[no-redef]
        def __init__(self, status_code: int = 400, detail: str = "", *args: Any, **kwargs: Any) -> None:
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Request:  # type: ignore[no-redef]
        pass

    class HTMLResponse:  # type: ignore[no-redef]
        def __init__(self, content: Any = None, status_code: int = 200, **kw: Any) -> None:
            self.body = content.encode("utf-8") if isinstance(content, str) else content
            self.status_code = status_code

    class JSONResponse:  # type: ignore[no-redef]
        def __init__(self, content: Any = None, status_code: int = 200, **kw: Any) -> None:
            self.body = json.dumps(content, default=str).encode("utf-8")
            self.status_code = status_code

    class StreamingResponse:  # type: ignore[no-redef]
        def __init__(self, content: Any = None, status_code: int = 200, **kw: Any) -> None:
            self.body = content
            self.status_code = status_code

from app.core.logging_middleware import RequestRecord, get_request_log_store

router = APIRouter(tags=["Server Live Logs & Telemetry"])


_HTML_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>⚡ Real-Time Server Request Logs | Bot Voice</title>
  <style>
    :root {
      --bg: #090d16;
      --card-bg: #131b2e;
      --border: #222f4c;
      --text: #e2e8f0;
      --text-dim: #94a3b8;
      --cyan: #38bdf8;
      --green: #4ade80;
      --yellow: #facc15;
      --red: #f87171;
      --purple: #c084fc;
      --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      padding: 16px;
      line-height: 1.5;
    }
    .header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border);
      gap: 12px;
    }
    .title-group { display: flex; align-items: center; gap: 12px; }
    .title-group h1 { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.5px; }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(74, 222, 128, 0.12);
      color: var(--green);
      border: 1px solid rgba(74, 222, 128, 0.3);
    }
    .status-dot {
      width: 8px; height: 8px; border-radius: 50%; background: var(--green);
      animation: pulse 2s infinite ease-in-out;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.4; transform: scale(0.85); }
    }
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 16px 0;
    }
    .stat-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      padding: 12px 14px;
      border-radius: 8px;
    }
    .stat-label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; font-weight: 600; }
    .stat-val { font-size: 1.5rem; font-weight: 700; font-family: var(--font-mono); margin-top: 4px; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      margin-bottom: 12px;
    }
    .filter-tabs { display: flex; gap: 6px; }
    .btn {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.82rem;
      cursor: pointer;
      transition: all 0.15s;
    }
    .btn:hover { border-color: var(--cyan); }
    .btn.active { background: #1e293b; border-color: var(--cyan); color: var(--cyan); font-weight: 600; }
    .search-box {
      flex: 1;
      max-width: 320px;
      min-width: 180px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.85rem;
      outline: none;
    }
    .search-box:focus { border-color: var(--cyan); }
    .actions-group { display: flex; gap: 8px; }
    .terminal-container {
      background: #060911;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      height: calc(100vh - 240px);
      min-height: 400px;
    }
    .terminal-header {
      background: #0f172a;
      border-bottom: 1px solid var(--border);
      padding: 8px 14px;
      font-size: 0.75rem;
      font-family: var(--font-mono);
      display: flex;
      justify-content: space-between;
      color: var(--text-dim);
    }
    .terminal-body {
      flex: 1;
      overflow-y: auto;
      padding: 6px;
      font-family: var(--font-mono);
      font-size: 0.82rem;
    }
    .log-row {
      display: grid;
      grid-template-columns: 85px 80px 90px 1fr 140px 105px 75px;
      align-items: center;
      gap: 8px;
      padding: 5px 8px;
      border-radius: 4px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.03);
      transition: background 0.1s;
    }
    .log-row:hover { background: rgba(255, 255, 255, 0.04); }
    .log-row.pending { opacity: 0.75; }
    .log-time { color: var(--text-dim); font-size: 0.75rem; }
    .badge {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 0.7rem;
      font-weight: 700;
      text-align: center;
      letter-spacing: 0.3px;
    }
    .badge-http { background: rgba(56, 189, 248, 0.15); color: var(--cyan); border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-telegram { background: rgba(192, 132, 252, 0.15); color: var(--purple); border: 1px solid rgba(192, 132, 252, 0.3); }
    .badge-get { background: rgba(74, 222, 128, 0.15); color: var(--green); }
    .badge-post { background: rgba(250, 204, 21, 0.15); color: var(--yellow); }
    .badge-command { background: rgba(192, 132, 252, 0.2); color: var(--purple); }
    .badge-callback { background: rgba(56, 189, 248, 0.2); color: var(--cyan); }
    .badge-status-ok { color: var(--green); font-weight: 600; }
    .badge-status-err { color: var(--red); font-weight: 700; }
    .badge-status-pending { color: var(--yellow); }
    .log-path { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-weight: 500; }
    .log-client { color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 0.76rem; }
    .log-dur { text-align: right; color: var(--cyan); font-weight: 600; font-size: 0.76rem; }
    @media (max-width: 768px) {
      .log-row { grid-template-columns: 65px 70px 1fr 75px; }
      .log-cat, .log-client, .log-time { display: none; }
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="title-group">
      <h1>⚡ Server Real-Time Request Stream</h1>
      <div id="connectionStatus" class="status-badge">
        <span class="status-dot"></span>
        <span id="connText">LIVE SSE CONNECTED</span>
      </div>
    </div>
    <div class="actions-group">
      <button id="toggleScrollBtn" class="btn" onclick="toggleAutoScroll()">⏬ Auto-Scroll: ON</button>
      <button class="btn" onclick="clearLogs()">🗑️ Clear</button>
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Active In-Flight</div>
      <div id="statActive" class="stat-val" style="color: var(--cyan)">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Total Requests</div>
      <div id="statTotal" class="stat-val">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Avg Latency</div>
      <div id="statLatency" class="stat-val" style="color: var(--green)">0 ms</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Req / Minute</div>
      <div id="statRpm" class="stat-val" style="color: var(--yellow)">0</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Errors (4xx/5xx)</div>
      <div id="statErrors" class="stat-val" style="color: var(--red)">0</div>
    </div>
  </div>

  <div class="toolbar">
    <div class="filter-tabs">
      <button class="btn active" onclick="setFilter('ALL', this)">All Requests</button>
      <button class="btn" onclick="setFilter('HTTP', this)">🌐 HTTP</button>
      <button class="btn" onclick="setFilter('TELEGRAM', this)">🤖 Telegram</button>
      <button class="btn" onclick="setFilter('ERRORS', this)">🚨 Errors Only</button>
    </div>
    <input type="text" id="searchInput" class="search-box" placeholder="🔍 Filter by path, IP, user..." oninput="renderFiltered()">
  </div>

  <div class="terminal-container">
    <div class="terminal-header">
      <span>REALTIME REQUEST EVENT STREAM</span>
      <span id="recordCount">0 events</span>
    </div>
    <div id="terminalBody" class="terminal-body"></div>
  </div>

  <script>
    let allRecords = [];
    let activeFilter = 'ALL';
    let autoScroll = true;
    const recordsMap = new Map();

    function setFilter(filter, el) {
      activeFilter = filter;
      document.querySelectorAll('.filter-tabs .btn').forEach(b => b.classList.remove('active'));
      if (el) el.classList.add('active');
      renderFiltered();
    }

    function toggleAutoScroll() {
      autoScroll = !autoScroll;
      const btn = document.getElementById('toggleScrollBtn');
      btn.innerText = autoScroll ? '⏬ Auto-Scroll: ON' : '⏸️ Auto-Scroll: OFF';
      btn.style.color = autoScroll ? 'var(--green)' : 'var(--yellow)';
    }

    function clearLogs() {
      allRecords = [];
      recordsMap.clear();
      renderFiltered();
    }

    function updateStats(metrics) {
      if (!metrics) return;
      document.getElementById('statActive').innerText = metrics.active_requests || 0;
      document.getElementById('statTotal').innerText = (metrics.total_requests || 0).toLocaleString();
      document.getElementById('statLatency').innerText = (metrics.average_latency_ms || 0) + ' ms';
      document.getElementById('statRpm').innerText = metrics.requests_per_minute || 0;
      document.getElementById('statErrors').innerText = metrics.total_errors || 0;
    }

    function createRowHtml(rec) {
      const isHttp = rec.category === 'HTTP';
      const catBadge = isHttp ? '<span class="badge badge-http">HTTP</span>' : '<span class="badge badge-telegram">TG</span>';
      
      let methodClass = 'badge-get';
      if (rec.method === 'POST') methodClass = 'badge-post';
      else if (rec.method === 'COMMAND') methodClass = 'badge-command';
      else if (rec.method === 'CALLBACK') methodClass = 'badge-callback';
      const methodBadge = `<span class="badge ${methodClass}">${rec.method}</span>`;

      let statusClass = 'badge-status-ok';
      if (rec.status === 'PENDING') statusClass = 'badge-status-pending';
      else if (rec.status_code >= 400 || rec.status.includes('FAIL') || rec.status.includes('ERROR')) statusClass = 'badge-status-err';

      const durStr = rec.duration_ms > 0 ? rec.duration_ms.toFixed(1) + 'ms' : (rec.status === 'PENDING' ? '⏳ pending' : '-');

      return `
        <div class="log-row ${rec.status === 'PENDING' ? 'pending' : ''}" id="row-${rec.id}">
          <span class="log-time">${rec.time_str || ''}</span>
          <span class="log-cat">${catBadge}</span>
          <span>${methodBadge}</span>
          <span class="log-path" title="${rec.path}">${rec.path}</span>
          <span class="log-client" title="${rec.client}">${rec.client}</span>
          <span class="${statusClass}">${rec.status}</span>
          <span class="log-dur">${durStr}</span>
        </div>
      `;
    }

    function renderFiltered() {
      const q = document.getElementById('searchInput').value.toLowerCase().trim();
      const container = document.getElementById('terminalBody');
      
      const filtered = allRecords.filter(rec => {
        if (activeFilter === 'HTTP' && rec.category !== 'HTTP') return false;
        if (activeFilter === 'TELEGRAM' && rec.category !== 'TELEGRAM') return false;
        if (activeFilter === 'ERRORS' && rec.status_code < 400 && !rec.status.includes('FAIL') && !rec.status.includes('ERROR')) return false;
        if (q) {
          const hay = `${rec.method} ${rec.path} ${rec.client} ${rec.status} ${rec.detail}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });

      container.innerHTML = filtered.map(createRowHtml).join('');
      document.getElementById('recordCount').innerText = `${filtered.length} of ${allRecords.length} events`;
      
      if (autoScroll) {
        container.scrollTop = container.scrollHeight;
      }
    }

    function upsertRecord(rec) {
      if (recordsMap.has(rec.id)) {
        const idx = allRecords.findIndex(r => r.id === rec.id);
        if (idx !== -1) {
          allRecords[idx] = rec;
        }
      } else {
        allRecords.push(rec);
        if (allRecords.length > 500) {
          const removed = allRecords.shift();
          if (removed) recordsMap.delete(removed.id);
        }
      }
      recordsMap.set(rec.id, rec);
      renderFiltered();
    }

    const urlParams = new URLSearchParams(window.location.search);
    const apiKey = urlParams.get('api_key') || '';
    const authQuery = apiKey ? ('?api_key=' + encodeURIComponent(apiKey)) : '';

    // Connect to Server-Sent Events (SSE) Stream
    function initEventSource() {
      const source = authQuery ? new EventSource('/api/logs/stream' + authQuery) : new EventSource('/api/logs/stream');

      source.onopen = function() {
        const badge = document.getElementById('connectionStatus');
        badge.style.color = 'var(--green)';
        badge.style.borderColor = 'rgba(74, 222, 128, 0.3)';
        document.getElementById('connText').innerText = 'LIVE SSE CONNECTED';
      };

      source.onmessage = function(event) {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'record' && data.record) {
            upsertRecord(data.record);
          } else if (data.type === 'initial') {
            if (Array.isArray(data.records)) {
              allRecords = data.records.reverse();
              allRecords.forEach(r => recordsMap.set(r.id, r));
            }
            if (data.metrics) updateStats(data.metrics);
            renderFiltered();
          } else if (data.type === 'metrics' && data.metrics) {
            updateStats(data.metrics);
          }
        } catch (e) {
          console.error("SSE parse error:", e);
        }
      };

      source.onerror = function() {
        const badge = document.getElementById('connectionStatus');
        badge.style.color = 'var(--yellow)';
        badge.style.borderColor = 'rgba(250, 204, 21, 0.3)';
        document.getElementById('connText').innerText = 'RECONNECTING...';
      };
    }

    // Initial load
    initEventSource();
    // Poll metrics every 3s
    setInterval(async () => {
      try {
        const res = await fetch('/api/logs?limit=1' + (apiKey ? ('&api_key=' + encodeURIComponent(apiKey)) : ''));
        const data = await res.json();
        if (data.metrics) updateStats(data.metrics);
      } catch (e) {}
    }, 3000);
  </script>
</body>
</html>
"""


def _check_logs_auth(
    x_api_key: str | None = None,
    authorization: str | None = None,
    api_key: str | None = None,
) -> None:
    """Ensure requester is authorized when API keys are configured."""
    from app.core.security import get_allowed_api_keys, validate_api_key

    allowed_keys = get_allowed_api_keys()
    if not allowed_keys:
        return
    token = x_api_key or api_key
    if not validate_api_key(token, authorization):
        raise HTTPException(status_code=401, detail="Unauthorized: Valid API key required")


@router.get("/logs", response_class=HTMLResponse)
@router.get("/server/logs", response_class=HTMLResponse)
@router.get("/live-logs", response_class=HTMLResponse)
async def live_logs_dashboard(
    api_key: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> HTMLResponse:
    """Serve the interactive real-time server telemetry and request stream dashboard."""
    _check_logs_auth(x_api_key=x_api_key, authorization=authorization, api_key=api_key)
    return HTMLResponse(content=_HTML_DASHBOARD)


@router.get("/api/logs")
async def get_recent_logs_api(
    limit: int = Query(default=100, ge=1, le=500),
    category: str | None = Query(default=None),
    only_errors: bool = Query(default=False),
    api_key: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """Retrieve recent request records and telemetry metrics in JSON format."""
    _check_logs_auth(x_api_key=x_api_key, authorization=authorization, api_key=api_key)
    store = get_request_log_store()
    records = store.get_recent(limit=limit, category=category, only_errors=only_errors)
    metrics = store.get_metrics()
    return JSONResponse({
        "metrics": metrics,
        "count": len(records),
        "records": records,
    })


@router.get("/api/logs/stream")
async def stream_logs_sse(
    request: Request = None,
    api_key: str | None = Query(default=None),
    x_api_key: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    """Server-Sent Events (SSE) streaming endpoint pushing real-time request records."""
    _check_logs_auth(x_api_key=x_api_key, authorization=authorization, api_key=api_key)
    store = get_request_log_store()
    queue = store.subscribe()

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # 1. Send initial burst with recent records and current metrics
            initial_records = store.get_recent(limit=60)
            initial_metrics = store.get_metrics()
            yield f"data: {json.dumps({'type': 'initial', 'records': initial_records, 'metrics': initial_metrics})}\n\n"

            last_ping = asyncio.get_running_loop().time()

            while True:
                # Disconnect check
                try:
                    disc_check = getattr(request, "is_disconnected", None)
                    if callable(disc_check):
                        res = disc_check()
                        if asyncio.iscoroutine(res):
                            if await res:
                                break
                        elif res:
                            break
                except Exception:
                    pass

                try:
                    record: RequestRecord = await asyncio.wait_for(queue.get(), timeout=2.0)
                    yield f"data: {json.dumps({'type': 'record', 'record': record.to_dict()})}\n\n"
                except asyncio.TimeoutError:
                    pass

                # Send heartbeat every 15s to keep connection alive through reverse proxies
                now = asyncio.get_running_loop().time()
                if now - last_ping >= 15.0:
                    metrics = store.get_metrics()
                    yield f"data: {json.dumps({'type': 'metrics', 'metrics': metrics})}\n\n"
                    last_ping = now

        except asyncio.CancelledError:
            pass
        finally:
            store.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
