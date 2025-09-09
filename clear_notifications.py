#!/usr/bin/env python3
"""Clear last_notified_date for all NOCs to force re-notification."""
import sqlite3
from pathlib import Path

# Update this path if needed
db_path = Path("data/noc_current.db")

if db_path.exists():
    conn = sqlite3.connect(db_path)
    try:
        # Clear all last_notified_date values
        conn.execute("UPDATE noc_extractions SET last_notified_date = NULL")
        conn.commit()
        
        # Check how many were updated
        cursor = conn.execute("SELECT COUNT(*) FROM noc_extractions WHERE last_notified_date IS NULL")
        count = cursor.fetchone()[0]
        
        print(f"Cleared last_notified_date for {count} NOCs")
        print("Run the cron job again to send notifications")
    finally:
        conn.close()
else:
    print(f"Database not found at {db_path}")