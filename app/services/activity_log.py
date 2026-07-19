"""
activity_log.py

Append-only activity log (JSONL). Now records which registered user
performed each action, so the admin panel can show "who did what."
"""

import json
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.core.config import settings


def log_activity(action: str, face_id: str, detail: str = "", username: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(settings.activity_log_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "face_id": face_id,
        "detail": detail,
        "username": username or "unknown",
    }
    with open(settings.activity_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def get_recent_activity(limit: int = 50) -> List[Dict[str, Any]]:
    if not os.path.exists(settings.activity_log_path):
        return []
    with open(settings.activity_log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    entries.reverse()
    return entries[:limit]


def get_activity_for_user(username: str, limit: int = 50) -> List[Dict[str, Any]]:
    all_entries = get_recent_activity(limit=10_000)
    filtered = [e for e in all_entries if e.get("username") == username]
    return filtered[:limit]