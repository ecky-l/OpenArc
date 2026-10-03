"""Unit tests for the worker-side per-session usage tracker (worker/session.py).

The worker keeps one entry per session id and, for each request carrying a
session id, annotates that request's produced metrics with:
  * session_context      -- the absolute current context measured this turn
  * session_delta        -- growth since the previous turn in THIS process
  * session_recalibrated -- True when this process has never seen the session
                            (a fresh worker after a crash/respawn, or a brand
                            -new session): the server re-bases to it.
"""

from __future__ import annotations

from src.engine.worker.session import WorkerSessionManager


def _metrics(input_token: int, **extra) -> dict:
    m = {"input_token": input_token, "new_token": 3, "stream": False}
    m.update(extra)
    return m


def test_first_turn_recalibrates() -> None:
    m = WorkerSessionManager()
    metrics = _metrics(5000)
    m.augment("sess-1", metrics)
    assert metrics["session_id"] == "sess-1"
    assert metrics["session_context"] == 5000
    assert metrics["session_delta"] is None        # no prior turn
    assert metrics["session_recalibrated"] is True


def test_continued_turn_reports_delta() -> None:
    m = WorkerSessionManager()
    m.augment("sess-1", _metrics(5000))
    metrics = _metrics(6800)
    m.augment("sess-1", metrics)
    assert metrics["session_context"] == 6800
    assert metrics["session_delta"] == 1800        # 6800 - 5000
    assert metrics["session_recalibrated"] is False


def test_fresh_process_resees_session_as_recalibrated() -> None:
    # The key property: a worker process has a fresh, empty tracker, so when it
    # sees a session it already knows about (goose re-sent the conversation after
    # a crash) it reports recalibrated=True -> the server re-bases, not adds.
    m = WorkerSessionManager()
    metrics = _metrics(9000)
    m.augment("sess-1", metrics)
    assert metrics["session_recalibrated"] is True
    assert metrics["session_delta"] is None


def test_independent_sessions() -> None:
    m = WorkerSessionManager()
    a = _metrics(1000)
    b = _metrics(2000)
    m.augment("sa", a)
    m.augment("sb", b)
    # sa first-seen -> recalibrate; sb first-seen -> recalibrate (independent).
    assert a["session_recalibrated"] is True
    assert b["session_recalibrated"] is True
    a2 = _metrics(1500)
    b2 = _metrics(3500)
    m.augment("sa", a2)
    m.augment("sb", b2)
    assert a2["session_delta"] == 500
    assert b2["session_delta"] == 1500
    assert a2["session_recalibrated"] is False
    assert b2["session_recalibrated"] is False


def test_no_input_token_is_a_noop() -> None:
    # Engines that never report a context (embeddings, rerank, ...) must be
    # untouched: their metrics carry no input_token, so no session keys are added
    # and nothing is tracked. (Sessions only ever arrive with LLM/VLM chat.)
    m = WorkerSessionManager()
    metrics = {"prompt_tokens": 10, "stream": False}
    m.augment("sess-1", metrics)
    assert "session_context" not in metrics
    assert "session_delta" not in metrics
    assert "session_recalibrated" not in metrics
    assert m._sessions == {}


def test_negative_growth_clamped_to_zero() -> None:
    m = WorkerSessionManager()
    m.augment("sess-1", _metrics(5000))
    metrics = _metrics(1000)          # context shrank (e.g. re-template)
    m.augment("sess-1", metrics)
    assert metrics["session_delta"] == 0                 # clamped, not negative
    assert metrics["session_recalibrated"] is False
