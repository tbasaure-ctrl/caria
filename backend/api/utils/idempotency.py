"""
Idempotency utilities per audit document (4.1).
Ensures critical endpoints can be called multiple times without side effects.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional
from uuid import UUID

import psycopg2
import os


class IdempotencyKey:
    """Generate idempotency keys for requests."""

    @staticmethod
    def generate(
        user_id: UUID,
        endpoint: str,
        request_data: dict,
    ) -> str:
        """
        Generate idempotency key from user, endpoint, and request data.
        
        Per audit document (4.1): Critical endpoints should be idempotent.
        """
        # Create deterministic key
        key_data = {
            "user_id": str(user_id),
            "endpoint": endpoint,
            "data": sorted(request_data.items()),
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()


class IdempotencyStore:
    """Store and check idempotency keys."""

    def __init__(self, db_connection):
        self.db = db_connection
        self._ensure_table()

    def _ensure_table(self):
        """Ensure idempotency_keys table exists."""
        with self.db.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    key_hash VARCHAR(64) UNIQUE NOT NULL,
                    user_id UUID NOT NULL,
                    endpoint VARCHAR(255) NOT NULL,
                    response_status INTEGER NOT NULL,
                    response_body JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_idempotency_key_hash ON idempotency_keys(key_hash);
                CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_keys(expires_at);
                """
            )
            self.db.commit()

    def check_and_store(
        self,
        key_hash: str,
        user_id: UUID,
        endpoint: str,
    ) -> tuple[bool, Optional[dict]]:
        """
        Check if request is duplicate and store if new.
        
        Returns:
            (is_duplicate, cached_response)
        """
        with self.db.cursor() as cursor:
            # Check for existing key
            cursor.execute(
                """
                SELECT response_status, response_body
                FROM idempotency_keys
                WHERE key_hash = %s AND expires_at > CURRENT_TIMESTAMP
                """,
                (key_hash,),
            )
            result = cursor.fetchone()

            if result:
                # Duplicate request - return cached response
                return True, {
                    "status_code": result[0],
                    "body": result[1],
                }

            # New request - will be stored after processing
            return False, None

    def store_response(
        self,
        key_hash: str,
        user_id: UUID,
        endpoint: str,
        status_code: int,
        response_body: dict,
        ttl_seconds: int = 3600,  # 1 hour default
    ):
        """Store response for idempotency."""
        from datetime import datetime, timedelta

        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)

        with self.db.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO idempotency_keys (key_hash, user_id, endpoint, response_status, response_body, expires_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (key_hash) DO UPDATE SET
                    response_status = EXCLUDED.response_status,
                    response_body = EXCLUDED.response_body,
                    expires_at = EXCLUDED.expires_at
                """,
                (
                    key_hash,
                    str(user_id),
                    endpoint,
                    status_code,
                    json.dumps(response_body),
                    expires_at,
                ),
            )
            self.db.commit()

    def cleanup_expired(self):
        """Remove expired idempotency keys."""
        with self.db.cursor() as cursor:
            cursor.execute(
                "DELETE FROM idempotency_keys WHERE expires_at < CURRENT_TIMESTAMP"
            )
            self.db.commit()


def get_idempotency_store():
    """Get idempotency store instance."""
    db_conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        user=os.getenv("POSTGRES_USER", "caria_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB", "caria"),
    )
    return IdempotencyStore(db_conn)

