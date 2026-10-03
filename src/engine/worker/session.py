"""Worker-side per-session usage tracker.

The worker keeps one entry per session id and, for each request carrying one,
annotates that request's produced metrics with:
  * session_context      -- absolute current context measured this turn
  * session_delta        -- growth since the previous turn in THIS process
  * session_recalibrated -- True when this process has never seen the session
                            (a fresh worker after a crash/respawn, or a brand
                            new session): the server re-bases, rather than adds
A server-side Session proxies it so the reported value survives a worker restart.
The conversation itself stays in the (openvino_genai) ChatHistory in process.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class WorkerSession:
    context: int = 0
    turns: int = 0


class WorkerSessionManager:
    def __init__(self) -> None:
        self._sessions: Dict[str, WorkerSession] = {}

    def augment(self, session_id: str, metrics: Dict[str, Any]) -> None:
        """Add this session's usage to a produced metrics dict, in place. No-op
        (leaves the dict untouched) when it has no `input_token` -- only chat
        (LLM/VLM) requests carry one, so non-chat engines are unaffected."""
        if "input_token" not in metrics:
            return
        prev = self._sessions.get(session_id)
        context = int(metrics.get("input_token", 0))
        if prev is None:
            delta: Optional[int] = None
            recalibrated = True
        else:
            delta = max(0, context - prev.context)
            recalibrated = False
        self._sessions[session_id] = WorkerSession(
            context=context,
            turns=(prev.turns if prev else 0) + 1,
        )
        metrics["session_id"] = session_id
        metrics["session_context"] = context
        metrics["session_delta"] = delta
        metrics["session_recalibrated"] = recalibrated
