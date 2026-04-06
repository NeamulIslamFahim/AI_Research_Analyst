"""Persistence helpers for Streamlit workspaces.

Supabase is the preferred durable backend for deployment.
When Supabase is not configured, the app falls back to local JSON files.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
import streamlit as st

from backend.helpers import load_env_var


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def get_or_create_owner_id() -> str:
    """Return a stable anonymous owner id for the current user session.

    We keep it in the URL query params so refreshes and shared deployment links
    retain the same logical workspace owner id without requiring authentication.
    """
    owner_id = st.session_state.get("workspace_owner_id", "").strip()
    if owner_id:
        return owner_id

    try:
        owner_id = str(st.query_params.get("u", "")).strip()
    except Exception:
        owner_id = ""

    if not owner_id:
        owner_id = f"user-{uuid.uuid4().hex}"
        try:
            st.query_params["u"] = owner_id
        except Exception:
            pass

    st.session_state["workspace_owner_id"] = owner_id
    return owner_id


def _chat_logs_dir(owner_id: str) -> Path:
    path = Path("data/chat_logs") / owner_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _chat_log_path(owner_id: str, session_id: str) -> Path:
    return _chat_logs_dir(owner_id) / f"{session_id}.json"


def _serialize_session(session: dict[str, Any]) -> dict[str, Any]:
    record = {
        "id": session.get("id"),
        "title": session.get("title"),
        "mode": session.get("mode"),
        "paper_text": session.get("paper_text", ""),
        "last_uploaded_pdf_signature": session.get("last_uploaded_pdf_signature", ""),
        "writer_state": session.get("writer_state") or {"phase": "start"},
        "writer_intro_shown": bool(session.get("writer_intro_shown")),
        "updated_at": _utc_now(),
        "messages": [],
    }
    for msg in session.get("messages", []):
        record["messages"].append(
            {
                "role": msg.get("role"),
                "type": msg.get("type"),
                "display_text": msg.get("display_text"),
                "content": msg.get("content"),
                "effective_query": msg.get("effective_query"),
            }
        )
    return record


def _deserialize_session(record: dict[str, Any], default_mode: str) -> dict[str, Any]:
    return {
        "id": record.get("id") or f"chat-{uuid.uuid4().hex[:8]}",
        "title": record.get("title") or "New Workspace",
        "mode": record.get("mode") or default_mode,
        "messages": list(record.get("messages") or []),
        "paper_text": record.get("paper_text", "") or "",
        "last_uploaded_pdf_signature": record.get("last_uploaded_pdf_signature", "") or "",
        "writer_state": record.get("writer_state") or {"phase": "start"},
        "writer_intro_shown": bool(record.get("writer_intro_shown")),
    }


def _local_load_sessions(owner_id: str, default_mode: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(_chat_logs_dir(owner_id).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        records.append(payload)

    if not records:
        legacy_path = Path("data/chat_logs/chat-1.json")
        if legacy_path.exists():
            try:
                payload = json.loads(legacy_path.read_text(encoding="utf-8"))
                records.append(payload)
            except Exception:
                pass

    records.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return [_deserialize_session(item, default_mode) for item in records]


def _local_save_sessions(owner_id: str, sessions: list[dict[str, Any]]) -> None:
    target_dir = _chat_logs_dir(owner_id)
    keep_ids = {str(session.get("id", "")).strip() for session in sessions if session.get("id")}

    for session in sessions:
        try:
            path = _chat_log_path(owner_id, str(session.get("id", "")).strip())
            path.write_text(json.dumps(_serialize_session(session), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            continue

    for path in target_dir.glob("*.json"):
        if path.stem not in keep_ids:
            try:
                path.unlink()
            except OSError:
                pass


def _supabase_config() -> tuple[str, str, str] | None:
    url = (load_env_var("SUPABASE_URL") or "").rstrip("/")
    key = (load_env_var("SUPABASE_SERVICE_ROLE_KEY") or load_env_var("SUPABASE_ANON_KEY") or "").strip()
    table = (load_env_var("SUPABASE_SESSIONS_TABLE", "app_sessions") or "app_sessions").strip()
    if not url or not key:
        return None
    return url, key, table


def _supabase_headers(api_key: str) -> dict[str, str]:
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _remote_load_sessions(owner_id: str, default_mode: str) -> list[dict[str, Any]]:
    config = _supabase_config()
    if not config:
        return []

    url, api_key, table = config
    endpoint = f"{url}/rest/v1/{table}"
    params = {
        "owner_id": f"eq.{owner_id}",
        "select": "payload,updated_at",
        "order": "updated_at.desc",
    }
    response = requests.get(endpoint, headers=_supabase_headers(api_key), params=params, timeout=15)
    response.raise_for_status()
    rows = response.json() or []
    sessions: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload")
        if isinstance(payload, dict):
            sessions.append(_deserialize_session(payload, default_mode))
    return sessions


def _remote_save_sessions(owner_id: str, sessions: list[dict[str, Any]]) -> None:
    config = _supabase_config()
    if not config:
        raise RuntimeError("Supabase is not configured.")

    url, api_key, table = config
    endpoint = f"{url}/rest/v1/{table}"
    headers = _supabase_headers(api_key)
    serialized = [_serialize_session(session) for session in sessions]

    payload = [
        {
            "owner_id": owner_id,
            "session_id": session["id"],
            "title": session.get("title", "New Workspace"),
            "mode": session.get("mode", "Research Explorer"),
            "updated_at": session.get("updated_at") or _utc_now(),
            "payload": session,
        }
        for session in serialized
    ]

    if payload:
        upsert_query = urlencode({"on_conflict": "owner_id,session_id"})
        upsert_headers = {
            **headers,
            "Prefer": "resolution=merge-duplicates,return=minimal",
        }
        response = requests.post(f"{endpoint}?{upsert_query}", headers=upsert_headers, data=json.dumps(payload), timeout=20)
        response.raise_for_status()

    existing_response = requests.get(
        endpoint,
        headers=headers,
        params={"owner_id": f"eq.{owner_id}", "select": "session_id"},
        timeout=15,
    )
    existing_response.raise_for_status()
    existing_rows = existing_response.json() or []
    existing_ids = {str(row.get("session_id", "")).strip() for row in existing_rows if row.get("session_id")}
    keep_ids = {str(session.get("id", "")).strip() for session in serialized if session.get("id")}
    stale_ids = sorted(existing_ids - keep_ids)
    if stale_ids:
        quoted_ids = ",".join(f'"{session_id}"' for session_id in stale_ids)
        delete_params = {
            "owner_id": f"eq.{owner_id}",
            "session_id": f"in.({quoted_ids})",
        }
        delete_response = requests.delete(endpoint, headers=headers, params=delete_params, timeout=15)
        delete_response.raise_for_status()


def load_sessions(default_mode: str) -> list[dict[str, Any]]:
    owner_id = get_or_create_owner_id()
    try:
        sessions = _remote_load_sessions(owner_id, default_mode)
        if sessions:
            return sessions
    except Exception:
        pass
    return _local_load_sessions(owner_id, default_mode)


def save_sessions(sessions: list[dict[str, Any]]) -> None:
    owner_id = get_or_create_owner_id()
    try:
        _remote_save_sessions(owner_id, sessions)
    except Exception:
        _local_save_sessions(owner_id, sessions)
