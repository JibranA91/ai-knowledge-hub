#!/usr/bin/env python3
"""Reset an admin (or any) user's password directly in the database.

Usage
-----
  # Interactive — prompts for new password (input is hidden)
  python scripts/reset_password.py admin@example.com

  # Non-interactive — pass password as argument (useful in CI/scripts)
  python scripts/reset_password.py admin@example.com --password newpassword123

  # Using a custom DATABASE_URL instead of the one in .env
  DATABASE_URL=postgresql+asyncpg://... python scripts/reset_password.py admin

The script connects directly to the database, so the app does not need to be
running.  It reads DATABASE_URL from the environment / .env file automatically.
"""
import argparse
import asyncio
import getpass
import hashlib
import sys
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


async def _find_user(email: str) -> dict | None:
    import sqlalchemy as sa
    from app.db import get_db

    query = sa.text("""
        SELECT id, email, role, org_id
        FROM users
        WHERE email = :email
    """)
    async with get_db() as db:
        result = await db.execute(query, {"email": email})
        row = result.fetchone()
    if not row:
        return None
    return {
        "id": str(row.id),
        "email": row.email,
        "role": row.role,
        "org_id": str(row.org_id) if row.org_id else None,
    }


async def _update_password(user_id: str, new_hash: str) -> None:
    import sqlalchemy as sa
    from app.db import get_db

    query = sa.text("""
        UPDATE users SET password_hash = :hash WHERE id = CAST(:id AS UUID)
    """)
    async with get_db() as db:
        await db.execute(query, {"id": user_id, "hash": new_hash})


def _hash_password(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _validate_password(password: str) -> str | None:
    """Return an error message, or None if valid."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset a user's password directly in the database."
    )
    parser.add_argument("email", help="Email (or username) of the user to update")
    parser.add_argument(
        "--password", "-p",
        help="New password (if omitted you will be prompted interactively)",
    )
    args = parser.parse_args()

    # ── Resolve new password ──────────────────────────────────────────────────
    if args.password:
        new_password = args.password
    else:
        try:
            new_password = getpass.getpass(f"New password for '{args.email}': ")
            confirm      = getpass.getpass("Confirm new password: ")
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(1)
        if new_password != confirm:
            print("Error: passwords do not match.", file=sys.stderr)
            sys.exit(1)

    err = _validate_password(new_password)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # ── Run async work ────────────────────────────────────────────────────────
    async def _run():
        user = await _find_user(args.email)
        if not user:
            print(f"Error: no user found with email '{args.email}'.", file=sys.stderr)
            sys.exit(1)

        role_tag = f"[{user['role']}]"
        org_tag  = f"org={user['org_id']}" if user['org_id'] else "no org (admin)"
        print(f"Found: {user['email']}  {role_tag}  {org_tag}")

        new_hash = _hash_password(new_password)
        await _update_password(user["id"], new_hash)
        print(f"Password updated successfully for '{user['email']}'.")

    asyncio.run(_run())


if __name__ == "__main__":
    main()
