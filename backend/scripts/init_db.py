"""
Initialize the MySQL database tables for this project.

Prerequisites:
    1. A MySQL server is running and reachable.
    2. The database itself has been created, e.g.:
           mysql -u root -p -e "CREATE DATABASE crop_disease_detection;"
    3. backend/.env exists (copy from backend/.env.example) with correct
       DB_HOST / DB_PORT / DB_USER / DB_PASSWORD / DB_NAME values.

Usage (from the backend/ directory):
    python scripts/init_db.py
"""

import os
import sys

# Allow running this script directly (`python scripts/init_db.py`) by adding
# the backend/ root to sys.path so `import app...` resolves correctly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings  # noqa: E402
from app.database.session import init_db  # noqa: E402


def main():
    print(f"Connecting to MySQL database '{settings.DB_NAME}' at {settings.DB_HOST}:{settings.DB_PORT} ...")
    try:
        init_db()
    except Exception as exc:
        print(f"\nFailed to initialize the database: {exc}\n")
        print(
            "Checklist:\n"
            "  1. Is MySQL running?\n"
            f"  2. Does the database '{settings.DB_NAME}' already exist? "
            f"(CREATE DATABASE {settings.DB_NAME};)\n"
            "  3. Are DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME in backend/.env correct?\n"
        )
        sys.exit(1)

    print("Success. Tables created (or already existed):")
    print("  - predictions")


if __name__ == "__main__":
    main()
