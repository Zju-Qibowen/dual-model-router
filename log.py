"""结构化路由日志 — 本地 SQLite 记录每次路由决策。

轻量、容错、不阻塞主流程。日志仅用于回溯分析，不参与路由逻辑。
"""

import os
import hashlib
import sqlite3
import threading
import logging
from datetime import datetime

logger = logging.getLogger("dual-model-router")

DB_DIR = os.path.join(os.path.expanduser("~"), ".claude", "dual-model-router")
DB_PATH = os.path.join(DB_DIR, "routing_log.db")
MAX_ROWS = 1000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS route_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT DEFAULT (datetime('now','localtime')),
    task_hash TEXT,
    task_len INTEGER,
    has_images INTEGER,
    image_count INTEGER,
    route_decision TEXT,
    route_method TEXT,
    exec_model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    success INTEGER,
    latency_ms INTEGER
)
"""


class RouteLogger:
    """单例路由日志器。写入失败绝不抛异常。"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        os.makedirs(DB_DIR, exist_ok=True)
        self._conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def log(self, **kwargs) -> None:
        """写入一条日志。失败时只打 warning，不抛异常。"""
        fields = [
            "task_hash", "task_len", "has_images", "image_count",
            "route_decision", "route_method", "exec_model",
            "input_tokens", "output_tokens", "success", "latency_ms",
        ]
        values = {f: kwargs.get(f) for f in fields}
        try:
            with self._lock:
                self._conn.execute(
                    f"INSERT INTO route_log ({', '.join(fields)}) "
                    f"VALUES ({', '.join('?' * len(fields))})",
                    [values[f] for f in fields],
                )
                self._conn.commit()
                self._prune()
        except Exception:
            logger.warning("路由日志写入失败", exc_info=True)

    def recent(self, n: int = 10) -> list[dict]:
        """查询最近 N 条记录。"""
        try:
            rows = self._conn.execute(
                "SELECT * FROM route_log ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
            cols = [d[0] for d in rows[0].cursor.description] if rows else []
            return [dict(zip(cols, row)) for row in rows]
        except Exception:
            return []

    def _prune(self) -> None:
        """保留最近 MAX_ROWS 条，超出删除。"""
        try:
            row_count = self._conn.execute(
                "SELECT COUNT(*) FROM route_log"
            ).fetchone()[0]
            if row_count > MAX_ROWS:
                delete_n = row_count - MAX_ROWS
                self._conn.execute(
                    "DELETE FROM route_log WHERE id IN ("
                    "SELECT id FROM route_log ORDER BY id ASC LIMIT ?"
                    ")", (delete_n,)
                )
                self._conn.commit()
        except Exception:
            pass


def task_hash(text: str) -> str:
    """计算 task 文本的短哈希。"""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


# 模块级单例
_route_logger = RouteLogger()
