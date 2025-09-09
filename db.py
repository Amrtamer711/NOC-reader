import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
import os
import logging
import json
import shutil

from config import (
    CURRENT_DB_PATH, 
    HISTORY_DB_PATH, 
    CURRENT_DOCS_DIR, 
    EXPIRED_DOCS_DIR,
    IS_PRODUCTION
)

logger = logging.getLogger(__name__)

CURRENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS noc_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    noc_number TEXT NOT NULL UNIQUE,
    project_name TEXT,
    issue_date TEXT,
    noc_type TEXT,
    validity_end_date TEXT,
    comments TEXT,
    submitted_by TEXT NOT NULL,
    raw_data TEXT,
    document_path TEXT,
    document_filename TEXT,
    last_notified_date TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_noc_number ON noc_extractions(noc_number);
CREATE INDEX IF NOT EXISTS idx_validity_end_date ON noc_extractions(validity_end_date);
CREATE INDEX IF NOT EXISTS idx_timestamp ON noc_extractions(timestamp);
CREATE INDEX IF NOT EXISTS idx_submitted_by ON noc_extractions(submitted_by);

CREATE TRIGGER IF NOT EXISTS update_timestamp 
AFTER UPDATE ON noc_extractions
BEGIN
    UPDATE noc_extractions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
"""

HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS noc_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_id INTEGER,
    timestamp TEXT NOT NULL,
    noc_number TEXT NOT NULL,
    project_name TEXT,
    issue_date TEXT,
    noc_type TEXT,
    validity_end_date TEXT,
    comments TEXT,
    submitted_by TEXT NOT NULL,
    raw_data TEXT,
    document_path TEXT,
    document_filename TEXT,
    archived_at TEXT DEFAULT CURRENT_TIMESTAMP,
    archive_reason TEXT
);

CREATE INDEX IF NOT EXISTS idx_history_noc_number ON noc_history(noc_number);
CREATE INDEX IF NOT EXISTS idx_history_validity_end_date ON noc_history(validity_end_date);
CREATE INDEX IF NOT EXISTS idx_history_archived_at ON noc_history(archived_at);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    """Create a connection with proper WAL and concurrency settings."""
    # Use default isolation level to allow explicit transactions
    conn = sqlite3.connect(db_path, timeout=10.0)
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")  # 10 second timeout
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA cache_size=-2000;")  # 2MB cache
    conn.execute("PRAGMA temp_store=MEMORY;")
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    """Initialize both databases with the required schemas."""
    # Initialize current database
    conn = _connect(CURRENT_DB_PATH)
    try:
        conn.executescript(CURRENT_SCHEMA)
        conn.commit()
        logger.info("[DB] Current database initialized successfully")
    except Exception as e:
        logger.error(f"[DB] Error initializing current database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()
    
    # Initialize history database
    conn = _connect(HISTORY_DB_PATH)
    try:
        conn.executescript(HISTORY_SCHEMA)
        conn.commit()
        logger.info("[DB] History database initialized successfully")
    except Exception as e:
        logger.error(f"[DB] Error initializing history database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def save_noc_extraction(
    noc_number: str,
    project_name: str = "",
    issue_date: str = "",
    noc_type: str = "",
    validity_end_date: str = "",
    comments: str = "",
    submitted_by: str = "",
    timestamp: Optional[str] = None,
    raw_data: Optional[Dict[str, Any]] = None,
    document_data: Optional[bytes] = None,
    document_filename: Optional[str] = None
) -> int:
    """Save NOC extraction data to database. Returns the ID of the inserted record."""
    if not timestamp:
        timestamp = datetime.now().isoformat()
    
    document_path = None
    
    # Save document to file system if provided
    if document_data and document_filename:
        try:
            # Generate unique filename using NOC number and timestamp
            safe_noc = noc_number.replace('/', '_').replace(' ', '_')
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_ext = Path(document_filename).suffix or '.pdf'
            stored_filename = f"{safe_noc}_{timestamp_str}{file_ext}"
            
            # Save to current docs directory
            document_path = CURRENT_DOCS_DIR / stored_filename
            with open(document_path, 'wb') as f:
                f.write(document_data)
            logger.info(f"[DB] Saved document to {document_path}")
        except Exception as e:
            logger.error(f"[DB] Error saving document file: {e}")
            document_path = None
    
    conn = _connect(CURRENT_DB_PATH)
    try:
        conn.execute("BEGIN")
        cursor = conn.execute(
            """
            INSERT OR REPLACE INTO noc_extractions 
            (timestamp, noc_number, project_name, issue_date, noc_type, validity_end_date, 
             comments, submitted_by, raw_data, document_path, document_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                noc_number,
                project_name or "",
                issue_date or "",
                noc_type or "",
                validity_end_date or "",
                comments or "",
                submitted_by,
                json.dumps(raw_data) if raw_data else "",
                str(document_path) if document_path else None,
                document_filename
            ),
        )
        record_id = cursor.lastrowid
        conn.commit()
        logger.info(f"[DB] Saved NOC extraction with ID {record_id}")
        return record_id
    except Exception as e:
        logger.error(f"[DB] Error saving NOC extraction: {e}")
        conn.rollback()
        # Clean up document file on failure
        if document_path and document_path.exists():
            try:
                os.unlink(document_path)
            except:
                pass
        return 0
    finally:
        conn.close()


def get_noc_by_number(noc_number: str) -> Optional[Dict[str, Any]]:
    """Get NOC by its reference number from current database."""
    conn = _connect(CURRENT_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM noc_extractions WHERE noc_number = ?",
            (noc_number,)
        )
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    except Exception as e:
        logger.error(f"[DB] Error getting NOC by number: {e}")
        return None
    finally:
        conn.close()


def get_noc_document(noc_number: str) -> Optional[Tuple[bytes, str]]:
    """Get NOC document data and filename. Returns (document_data, filename) or None."""
    conn = _connect(CURRENT_DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT document_path, document_filename FROM noc_extractions WHERE noc_number = ?",
            (noc_number,)
        )
        row = cursor.fetchone()
        if row and row[0]:
            doc_path = Path(row[0])
            if doc_path.exists():
                try:
                    with open(doc_path, 'rb') as f:
                        return (f.read(), row[1] or f"NOC_{noc_number}.pdf")
                except Exception as e:
                    logger.error(f"[DB] Error reading document file: {e}")
        
        # Check history if not in current
        conn_hist = _connect(HISTORY_DB_PATH)
        try:
            cursor = conn_hist.execute(
                "SELECT document_path, document_filename FROM noc_history WHERE noc_number = ? ORDER BY archived_at DESC LIMIT 1",
                (noc_number,)
            )
            row = cursor.fetchone()
            if row and row[0]:
                doc_path = Path(row[0])
                if doc_path.exists():
                    try:
                        with open(doc_path, 'rb') as f:
                            return (f.read(), row[1] or f"NOC_{noc_number}.pdf")
                    except Exception as e:
                        logger.error(f"[DB] Error reading document file: {e}")
        finally:
            conn_hist.close()
            
        return None
    except Exception as e:
        logger.error(f"[DB] Error getting NOC document: {e}")
        return None
    finally:
        conn.close()


def archive_expired_nocs() -> List[str]:
    """Move expired NOCs to history database. Returns list of archived NOC numbers."""
    today = datetime.now().date()
    archived = []
    
    conn_current = _connect(CURRENT_DB_PATH)
    conn_history = _connect(HISTORY_DB_PATH)
    
    try:
        # Find expired NOCs
        conn_current.row_factory = sqlite3.Row
        cursor = conn_current.execute(
            """
            SELECT * FROM noc_extractions 
            WHERE validity_end_date != ''
            """
        )
        all_nocs = [dict(row) for row in cursor.fetchall()]
        
        # Filter expired NOCs by parsing DD-MM-YYYY dates
        expired_nocs = []
        for noc in all_nocs:
            try:
                validity_date = datetime.strptime(noc['validity_end_date'], '%d-%m-%Y').date()
                if validity_date < today:
                    expired_nocs.append(noc)
            except ValueError:
                # Skip if date format is invalid
                continue
        
        if expired_nocs:
            conn_current.execute("BEGIN")
            conn_history.execute("BEGIN")
            
            for noc in expired_nocs:
                # Move document file if exists
                new_doc_path = None
                if noc['document_path']:
                    old_path = Path(noc['document_path'])
                    if old_path.exists():
                        try:
                            filename = old_path.name
                            new_doc_path = EXPIRED_DOCS_DIR / filename
                            shutil.move(str(old_path), str(new_doc_path))
                            logger.info(f"[DB] Moved document from {old_path} to {new_doc_path}")
                        except Exception as e:
                            logger.error(f"[DB] Error moving document: {e}")
                            new_doc_path = noc['document_path']  # Keep original path
                
                # Insert into history
                conn_history.execute(
                    """
                    INSERT INTO noc_history 
                    (original_id, timestamp, noc_number, project_name, issue_date, noc_type, 
                     validity_end_date, comments, submitted_by, raw_data, document_path, 
                     document_filename, archive_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        noc['id'],
                        noc['timestamp'],
                        noc['noc_number'],
                        noc['project_name'],
                        noc['issue_date'],
                        noc['noc_type'],
                        noc['validity_end_date'],
                        noc['comments'],
                        noc['submitted_by'],
                        noc['raw_data'],
                        str(new_doc_path) if new_doc_path else noc['document_path'],
                        noc['document_filename'],
                        'Validity expired'
                    )
                )
                
                # Delete from current
                conn_current.execute(
                    "DELETE FROM noc_extractions WHERE id = ?",
                    (noc['id'],)
                )
                
                archived.append(noc['noc_number'])
            
            conn_current.commit()
            conn_history.commit()
            logger.info(f"[DB] Archived {len(archived)} expired NOCs")
    except Exception as e:
        logger.error(f"[DB] Error archiving expired NOCs: {e}")
        conn_current.rollback()
        conn_history.rollback()
    finally:
        conn_current.close()
        conn_history.close()
    
    return archived


def get_all_noc_extractions() -> List[Dict[str, Any]]:
    """Get all NOC extractions from current database."""
    conn = _connect(CURRENT_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT * FROM noc_extractions ORDER BY timestamp DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"[DB] Error reading NOC extractions: {e}")
        return []
    finally:
        conn.close()


def get_expiring_nocs(days_ahead: int = 14) -> List[Dict[str, Any]]:
    """Get NOCs expiring within the specified number of days."""
    conn = _connect(CURRENT_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        # Calculate date range
        today = datetime.now().date()
        future_date = today + timedelta(days=days_ahead)
        
        cursor = conn.execute(
            """
            SELECT * FROM noc_extractions 
            WHERE validity_end_date != ''
            """
        )
        all_nocs = [dict(row) for row in cursor.fetchall()]
        
        # Filter NOCs expiring within the specified days
        expiring_nocs = []
        for noc in all_nocs:
            try:
                validity_date = datetime.strptime(noc['validity_end_date'], '%d-%m-%Y').date()
                if today <= validity_date <= future_date:
                    expiring_nocs.append(noc)
            except ValueError:
                # Skip if date format is invalid
                continue
        
        # Sort by validity date
        expiring_nocs.sort(key=lambda x: datetime.strptime(x['validity_end_date'], '%d-%m-%Y'))
        return expiring_nocs
    except Exception as e:
        logger.error(f"[DB] Error getting expiring NOCs: {e}")
        return []
    finally:
        conn.close()


def update_last_notified(noc_number: str, notified_date: str) -> bool:
    """Update the last notified date for a NOC."""
    conn = _connect(CURRENT_DB_PATH)
    try:
        conn.execute("BEGIN")
        conn.execute(
            """
            UPDATE noc_extractions 
            SET last_notified_date = ? 
            WHERE noc_number = ?
            """,
            (notified_date, noc_number)
        )
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"[DB] Error updating last notified date: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()


def export_to_excel() -> str:
    """Export NOC extractions to Excel file and return the file path."""
    import pandas as pd
    import tempfile
    from datetime import datetime
    
    conn_current = _connect(CURRENT_DB_PATH)
    conn_history = _connect(HISTORY_DB_PATH)
    try:
        # Read current NOC extractions
        df_current = pd.read_sql_query(
            """
            SELECT 
                id as ID,
                timestamp as Timestamp,
                noc_number as "NOC Number",
                project_name as "Project Name",
                issue_date as "Issue Date",
                noc_type as "NOC Type",
                validity_end_date as "Validity End Date",
                comments as Comments,
                submitted_by as "Submitted By",
                'Active' as Status
            FROM noc_extractions 
            ORDER BY timestamp DESC
            """,
            conn_current
        )
        
        # Read expired NOC extractions from history
        df_expired = pd.read_sql_query(
            """
            SELECT 
                original_id as ID,
                timestamp as Timestamp,
                noc_number as "NOC Number",
                project_name as "Project Name",
                issue_date as "Issue Date",
                noc_type as "NOC Type",
                validity_end_date as "Validity End Date",
                comments as Comments,
                submitted_by as "Submitted By",
                'Expired' as Status
            FROM noc_history 
            ORDER BY timestamp DESC
            """,
            conn_history
        )
        
        # Combine both dataframes
        df = pd.concat([df_current, df_expired], ignore_index=True)
        df = df.sort_values('Timestamp', ascending=False)
        
        # Create a temporary Excel file
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f'_noc_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        )
        temp_file.close()
        
        # Write to Excel with formatting
        with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:
            # Write all data to first sheet
            df.to_excel(writer, sheet_name='All NOCs', index=False)
            
            # Write current NOCs to second sheet
            df_current.to_excel(writer, sheet_name='Active NOCs', index=False)
            
            # Write expired NOCs to third sheet
            df_expired.to_excel(writer, sheet_name='Expired NOCs', index=False)
            
            # Format all sheets
            workbook = writer.book
            from openpyxl.styles import Font, PatternFill
            header_font = Font(bold=True)
            header_fill = PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")
            
            for sheet_name in ['All NOCs', 'Active NOCs', 'Expired NOCs']:
                worksheet = writer.sheets[sheet_name]
                
                # Auto-adjust column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
                
                # Add filters
                worksheet.auto_filter.ref = worksheet.dimensions
                
                # Format header row
                for cell in worksheet[1]:
                    cell.font = header_font
                    cell.fill = header_fill
        
        logger.info(f"[DB] Exported {len(df)} records to {temp_file.name}")
        return temp_file.name
        
    except Exception as e:
        logger.error(f"[DB] Error exporting to Excel: {e}")
        raise
    finally:
        conn_current.close()
        conn_history.close()


def get_noc_summary() -> Dict[str, Any]:
    """Get a summary of NOC data for display."""
    conn = _connect(CURRENT_DB_PATH)
    try:
        cursor = conn.cursor()
        
        # Total count in current
        cursor.execute("SELECT COUNT(*) FROM noc_extractions")
        total_count = cursor.fetchone()[0]
        
        # Total in history
        conn_hist = _connect(HISTORY_DB_PATH)
        cursor_hist = conn_hist.cursor()
        cursor_hist.execute("SELECT COUNT(*) FROM noc_history")
        history_count = cursor_hist.fetchone()[0]
        conn_hist.close()
        
        # Count by NOC type
        cursor.execute("""
            SELECT noc_type, COUNT(*) 
            FROM noc_extractions 
            WHERE noc_type != ''
            GROUP BY noc_type
        """)
        by_type = dict(cursor.fetchall())
        
        # Recent NOCs
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT noc_number, project_name, timestamp, validity_end_date 
            FROM noc_extractions 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        recent = [dict(row) for row in cursor.fetchall()]
        
        # Expiring soon
        expiring = get_expiring_nocs(14)
        
        return {
            "total_nocs": total_count,
            "history_nocs": history_count,
            "by_type": by_type,
            "recent_nocs": recent,
            "expiring_soon": expiring[:5]  # Show top 5
        }
        
    finally:
        conn.close()


def migrate_from_excel(excel_path: str) -> int:
    """Migrate data from Excel file to database. Returns number of records migrated."""
    import pandas as pd
    
    if not os.path.exists(excel_path):
        logger.error(f"[DB] Excel file not found: {excel_path}")
        return 0
    
    try:
        # Read Excel file
        df = pd.read_excel(excel_path, sheet_name='NOC Extractions')
        
        # Map column names
        column_mapping = {
            'ID': 'id',
            'Timestamp': 'timestamp',
            'NOC Number': 'noc_number',
            'Project Name': 'project_name',
            'Issue Date': 'issue_date',
            'NOC Type': 'noc_type',
            'Validity End Date': 'validity_end_date',
            'Comments': 'comments',
            'Submitted By': 'submitted_by'
        }
        
        # Rename columns
        df.rename(columns=column_mapping, inplace=True)
        
        # Fill NaN values with empty strings
        df.fillna('', inplace=True)
        
        # Convert to records
        records = df.to_dict('records')
        
        # Insert into database
        migrated = 0
        for record in records:
            # Skip if already exists (based on NOC number)
            existing = get_noc_by_number(str(record.get('noc_number', '')))
            
            if not existing:
                save_noc_extraction(
                    noc_number=str(record.get('noc_number', '')),
                    project_name=str(record.get('project_name', '')),
                    issue_date=str(record.get('issue_date', '')),
                    noc_type=str(record.get('noc_type', '')),
                    validity_end_date=str(record.get('validity_end_date', '')),
                    comments=str(record.get('comments', '')),
                    submitted_by=str(record.get('submitted_by', '')),
                    timestamp=str(record.get('timestamp', ''))
                )
                migrated += 1
        
        logger.info(f"[DB] Migrated {migrated} records from Excel")
        return migrated
        
    except Exception as e:
        logger.error(f"[DB] Error migrating from Excel: {e}")
        return 0


# Initialize DB on import
init_db()