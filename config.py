import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Detect environment
IS_RENDER = os.environ.get('RENDER') == 'true' or os.environ.get('RENDER_SERVICE_NAME') is not None
IS_PRODUCTION = IS_RENDER or os.path.exists('/data/')

if IS_RENDER:
    logger.info("[ENV] Running on Render")
elif IS_PRODUCTION:
    logger.info("[ENV] Running in production mode (found /data/)")
else:
    logger.info("[ENV] Running in development mode")

# Base data directory
if IS_PRODUCTION:
    DATA_DIR = Path("/data")
else:
    DATA_DIR = Path(__file__).parent / "data"

# Ensure base directory exists
DATA_DIR.mkdir(exist_ok=True)

# Document directories
CURRENT_DOCS_DIR = DATA_DIR / "current_docs"
EXPIRED_DOCS_DIR = DATA_DIR / "expired_docs"

# Create document directories
CURRENT_DOCS_DIR.mkdir(exist_ok=True)
EXPIRED_DOCS_DIR.mkdir(exist_ok=True)

# Database paths
CURRENT_DB_PATH = DATA_DIR / "noc_current.db"
HISTORY_DB_PATH = DATA_DIR / "noc_history.db"

logger.info(f"[ENV] Data directory: {DATA_DIR}")
logger.info(f"[ENV] Current documents: {CURRENT_DOCS_DIR}")
logger.info(f"[ENV] Expired documents: {EXPIRED_DOCS_DIR}")
logger.info(f"[ENV] Current database: {CURRENT_DB_PATH}")
logger.info(f"[ENV] History database: {HISTORY_DB_PATH}")

# Configurable settings
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4o')
EXPIRY_NOTIFICATION_DAYS = int(os.getenv('EXPIRY_NOTIFICATION_DAYS', '14'))
MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '0'))  # 0 means no limit
API_PORT = int(os.getenv('API_PORT', '3000'))

# Additional settings
LOG_FILE = os.getenv('LOG_FILE', 'noc_reader_bot.log')
USER_HISTORY_LIMIT = int(os.getenv('USER_HISTORY_LIMIT', '20'))
LLM_HISTORY_LIMIT = int(os.getenv('LLM_HISTORY_LIMIT', '8'))
UPLOAD_DIR = os.getenv('UPLOAD_DIR', './uploads')