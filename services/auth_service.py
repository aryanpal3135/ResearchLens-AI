"""
ResearchLens AI - Authentication & User Management Service.
Handles user registration, login, secure PBKDF2 password hashing, and user profile persistence in SQLite.
"""

from contextlib import contextmanager
import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

DB_PATH = (
    Path(os.getenv("SQLITE_DB_PATH"))
    if os.getenv("SQLITE_DB_PATH")
    else Path(__file__).resolve().parent.parent / "data" / "users.db"
)


class AuthService:
    """Manages user persistence, password hashing, and authentication."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes the users table if it does not already exist."""
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    institution TEXT NOT NULL DEFAULT 'Academic Institution',
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    @staticmethod
    def _hash_password(password: str, salt_bytes: Optional[bytes] = None) -> Tuple[str, str]:
        """Hashes a password with PBKDF2-HMAC-SHA256 and a 16-byte random salt."""
        salt = salt_bytes or os.urandom(16)
        pwd_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations=100000,
        )
        return pwd_hash.hex(), salt.hex()

    def register_user(
        self,
        name: str,
        email: str,
        password: str,
        institution: str = "Independent Researcher",
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Registers a new user account. Returns (success, message, user_dict)."""
        name = name.strip()
        email = email.strip().lower()
        institution = institution.strip() or "Independent Researcher"

        if not name or len(name) < 2:
            return False, "Please enter a valid name (at least 2 characters).", None

        if not email or "@" not in email or "." not in email:
            return False, "Please enter a valid email address.", None

        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters long.", None

        pwd_hash, salt_hex = self._hash_password(password)
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        try:
            with self._connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO users (name, email, password_hash, salt, institution, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (name, email, pwd_hash, salt_hex, institution, now_str),
                )
                conn.commit()
                user_id = cursor.lastrowid

            user = {
                "id": user_id,
                "name": name,
                "email": email,
                "institution": institution,
                "created_at": now_str,
            }
            return True, "Account created successfully!", user
        except sqlite3.IntegrityError:
            return False, "An account with this email address already exists. Please log in.", None
        except Exception as e:
            return False, f"Registration error: {str(e)}", None

    def authenticate_user(self, email: str, password: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Authenticates user credentials. Returns (success, message, user_dict)."""
        email = email.strip().lower()
        if not email or not password:
            return False, "Please enter both email and password.", None

        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT id, name, email, password_hash, salt, institution, created_at FROM users WHERE email = ?",
                (email,),
            )
            row = cursor.fetchone()

        if not row:
            return False, "No account found with this email address.", None

        stored_hash = row["password_hash"]
        salt_bytes = bytes.fromhex(row["salt"])
        computed_hash, _ = self._hash_password(password, salt_bytes=salt_bytes)

        if computed_hash != stored_hash:
            return False, "Incorrect password. Please try again.", None

        user = {
            "id": row["id"],
            "name": row["name"],
            "email": row["email"],
            "institution": row["institution"],
            "created_at": row["created_at"],
        }
        return True, "Login successful!", user

    def get_or_create_demo_user(self) -> Dict[str, Any]:
        """Ensures a default demo researcher account exists for instant preview."""
        demo_email = "researcher@researchlens.ai"
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT id, name, email, institution, created_at FROM users WHERE email = ?",
                (demo_email,),
            )
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "name": row["name"],
                    "email": row["email"],
                    "institution": row["institution"],
                    "created_at": row["created_at"],
                }

        _, _, user = self.register_user(
            name="Dr. Aryan Sharma",
            email=demo_email,
            password="DemoPassword2026!",
            institution="AI Research Institute",
        )
        return user or {
            "id": 1,
            "name": "Dr. Aryan Sharma",
            "email": demo_email,
            "institution": "AI Research Institute",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
