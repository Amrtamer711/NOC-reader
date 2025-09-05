import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, List
from slack_sdk.web.async_client import AsyncWebClient

logger = logging.getLogger(__name__)

# Admin config path
ADMIN_CONFIG_FILE = Path(__file__).parent / "admin_config.json"
_ADMIN_CONFIG: Dict[str, Dict[str, Dict[str, object]]] = {}


def load_admin_config() -> None:
    """Load admin configuration from JSON file."""
    global _ADMIN_CONFIG
    try:
        logger.info(f"[ADMIN] Looking for config at: {ADMIN_CONFIG_FILE}")
        if ADMIN_CONFIG_FILE.exists():
            logger.info(f"[ADMIN] Found config at: {ADMIN_CONFIG_FILE}")
            _ADMIN_CONFIG = json.loads(ADMIN_CONFIG_FILE.read_text(encoding="utf-8"))
            logger.info(f"[ADMIN] Loaded config: {list(_ADMIN_CONFIG.keys())}")
        else:
            logger.warning(f"[ADMIN] Config file not found at: {ADMIN_CONFIG_FILE}")
            _ADMIN_CONFIG = {}
    except Exception as e:
        logger.warning(f"Failed to load admin_config.json: {e}")
        _ADMIN_CONFIG = {}


def is_admin(slack_user_id: str) -> bool:
    """Check if user has admin privileges (can export database)."""
    if not _ADMIN_CONFIG:
        logger.info(f"[ADMIN_CHECK] Loading admin config")
        load_admin_config()
    
    # Admin users are those in the 'admins' group
    admin_members = _ADMIN_CONFIG.get("admins", {})
    logger.info(f"[ADMIN_CHECK] Checking if {slack_user_id} is admin")
    logger.info(f"[ADMIN_CHECK] Admin members: {list(admin_members.keys())}")
    
    for name, info in admin_members.items():
        user_id = info.get("slack_user_id")
        is_active = info.get("active")
        logger.info(f"[ADMIN_CHECK] Checking {name}: user_id={user_id}, active={is_active}")
        
        if is_active and user_id == slack_user_id:
            logger.info(f"[ADMIN_CHECK] User {slack_user_id} is admin!")
            return True
    
    logger.info(f"[ADMIN_CHECK] User {slack_user_id} is NOT admin")
    return False


async def get_user_display_name(slack_client: AsyncWebClient, user_id: str) -> str:
    """Get user's display name from Slack."""
    try:
        user_info = await slack_client.users_info(user=user_id)
        profile = user_info["user"]["profile"]
        return profile.get("display_name") or profile.get("real_name") or user_id
    except Exception as e:
        logger.error(f"[ADMIN] Error getting user info: {e}")
        return user_id


def get_admin_notification_channels() -> List[str]:
    """Get all admin notification channels."""
    if not _ADMIN_CONFIG:
        load_admin_config()
    
    channels = []
    
    # Get individual admin channels only
    admin_members = _ADMIN_CONFIG.get("admins", {})
    for name, info in admin_members.items():
        if info.get("active") and info.get("notification_channel"):
            channel = info["notification_channel"]
            if channel not in channels:
                channels.append(channel)
    
    if not channels:
        logger.warning("[ADMIN] No notification channels configured for admins")
    
    return channels


def get_admin_user_ids() -> List[str]:
    """Get all active admin user IDs."""
    if not _ADMIN_CONFIG:
        load_admin_config()
    
    user_ids = []
    admin_members = _ADMIN_CONFIG.get("admins", {})
    
    for name, info in admin_members.items():
        if info.get("active") and info.get("slack_user_id"):
            user_ids.append(info["slack_user_id"])
    
    return user_ids


def add_admin(username: str, slack_user_id: str, notification_channel: str, active: bool = True) -> bool:
    """Add a new admin to the configuration file."""
    try:
        # Load current config
        if not _ADMIN_CONFIG:
            load_admin_config()
        
        # Ensure admins section exists
        if "admins" not in _ADMIN_CONFIG:
            _ADMIN_CONFIG["admins"] = {}
        
        # Add new admin
        _ADMIN_CONFIG["admins"][username] = {
            "active": active,
            "slack_user_id": slack_user_id,
            "notification_channel": notification_channel
        }
        
        # Save to file
        with open(ADMIN_CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(_ADMIN_CONFIG, f, indent=2)
        
        logger.info(f"[ADMIN] Added new admin: {username}")
        return True
        
    except Exception as e:
        logger.error(f"[ADMIN] Failed to add admin {username}: {e}")
        return False


# Initialize admin config on import
load_admin_config()