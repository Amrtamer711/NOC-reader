import os
import json
import logging
import asyncio
import re
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.signature import SignatureVerifier
import requests

# Import database and permissions modules
import db
import permissions
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants and configuration
UAE_TZ = ZoneInfo('Asia/Dubai')

# Validate required environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logger.error('OPENAI_API_KEY not found in environment variables')
    raise ValueError('OPENAI_API_KEY is required')

# Slack configuration
SLACK_BOT_TOKEN = os.environ.get('SLACK_BOT_TOKEN')
SLACK_SIGNING_SECRET = os.environ.get('SLACK_SIGNING_SECRET')
if not SLACK_BOT_TOKEN:
    logger.error('SLACK_BOT_TOKEN not found in environment variables')
    raise ValueError('SLACK_BOT_TOKEN is required')
if not SLACK_SIGNING_SECRET:
    logger.error('SLACK_SIGNING_SECRET not found in environment variables')
    raise ValueError('SLACK_SIGNING_SECRET is required')

# Initialize FastAPI app and OpenAI client
api = FastAPI(title='NOC Document Reader API', version='2.0')
client = AsyncOpenAI(api_key=OPENAI_API_KEY)
slack_client = AsyncWebClient(token=SLACK_BOT_TOKEN)
signature_verifier = SignatureVerifier(SLACK_SIGNING_SECRET)

# In-memory state
pending_confirmations: Dict[str, Dict[str, Any]] = {}
pending_admin_confirmations: Dict[str, Dict[str, Any]] = {}
user_history: Dict[str, List[Dict[str, str]]] = {}

# ================ OpenAI Responses helpers ================
def _extract_responses_text(res: Any) -> str:
    """Safely extract textual content from a Responses API result across SDK variants."""
    try:
        # Prefer convenience
        text = getattr(res, 'output_text', None)
        if isinstance(text, str) and text.strip():
            return text
        # Older/newer shapes
        output = getattr(res, 'output', None)
        if isinstance(output, list) and len(output) > 0:
            first = output[0]
            content = getattr(first, 'content', None)
            if isinstance(content, list):
                # Scan from end to find last text-like part
                for part in reversed(content):
                    # SDK may expose objects or dicts
                    if hasattr(part, 'text') and getattr(part, 'text'):
                        return getattr(part, 'text')
                    if isinstance(part, dict) and part.get('text'):
                        return part['text']
        # Fallback: string repr
        return str(res)
    except Exception:
        return ""

# ================= Pydantic models =================
class NOCParseRequest(BaseModel):
    file_path: Optional[str] = None
    save_to_excel: bool = False
    submitted_by: str = 'API'

# ================= Database utilities =================
async def save_to_database(data: Dict[str, Any], document_path: Optional[str] = None) -> int:
    """Save parsed NOC data to database with optional document, return assigned ID."""
    try:
        document_data = None
        document_filename = None
        
        # Read document if provided
        if document_path and os.path.exists(document_path):
            try:
                with open(document_path, 'rb') as f:
                    document_data = f.read()
                document_filename = os.path.basename(document_path)
            except Exception as e:
                logger.error(f'Error reading document file: {e}')
        
        # Save to database
        record_id = await asyncio.to_thread(
            db.save_noc_extraction,
            noc_number=data.get('noc_number', ''),
            project_name=data.get('project_name', ''),
            issue_date=data.get('issue_date', ''),
            noc_type=data.get('noc_type', ''),
            validity_end_date=data.get('validity_end_date', ''),
            comments=data.get('comments', ''),
            submitted_by=data.get('submitted_by', ''),
            timestamp=datetime.now(UAE_TZ).isoformat(),
            raw_data=data,
            document_data=document_data,
            document_filename=document_filename
        )
        return record_id
    except Exception as e:
        logger.error(f'Error saving to database: {e}')
        return 0

# ================= Date parsing utilities =================
# ================= Parsing with OpenAI =================
async def parse_noc_with_ai(file_path: str) -> Dict[str, Any]:
    """Parse a NOC PDF using OpenAI Responses API and extract the key fields."""
    if not file_path:
        raise ValueError('file_path is required')
    if not file_path.lower().endswith('.pdf'):
        raise ValueError('Only PDF files are supported. Please provide a .pdf file path.')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')

    uploaded_file_id: Optional[str] = None

    # Build system prompt with current date in UAE timezone
    now = datetime.now(UAE_TZ)
    today_str = now.strftime('%B %d, %Y')
    system_prompt = (
        f'You are an expert, helpful assistant that parses NOC (No Objection Certificate) documents.'
        f" Today's date is {today_str}.\n\n"
        'Your goal is to help the user by extracting accurate, normalized fields while keeping responses concise.'
        '\nExtract ONLY the following fields from the attached NOC PDF and return a STRICT JSON object:'
        '\n- noc_number: The NOC reference number or ID (choose the most prominent/official if multiple are present)'
        '\n- project_name: The official project name/title associated with the NOC (if multiple, pick the one tied to the NOC reference)'
        '\n- issue_date: The date the NOC was issued (prefer issuance over expiry/renewal; format as DD-MM-YYYY)'
        '\n- noc_type: The NOC type (e.g., company formation, activity change, etc.; keep it short and specific)'
        '\n- validity_end_date: The validity end date (date to/expiry), if present (format as DD-MM-YYYY)'
        "\n- comments: Brief human remarks or notes present in the document. EXCLUDE procedural instructions (e.g., 'bring original', 'pay fee', 'submit within 7 days')."
        '\nIf any field is not present, return an empty string for it.'
        '\nDo not include any commentary outside of the JSON output.'
        '\nNormalize values: trim labels/punctuation, and standardize dates to DD-MM-YYYY when possible.'
    )

    try:
        # Upload the PDF file
        with open(file_path, 'rb') as f:
            upload = await client.files.create(file=f, purpose='user_data')
        uploaded_file_id = upload.id
        logger.info(f'Uploaded file to OpenAI: {uploaded_file_id}')

        content = [
            {"type": "input_file", "file_id": uploaded_file_id},
            {
                "type": "input_text",
                "text": (
                    'Please read the PDF and output a JSON object matching the provided schema.'
                ),
            },
        ]

        # Call Responses API with enforced JSON schema
        res = await client.responses.create(
            model=config.OPENAI_MODEL,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            text={
                'format': {
                    'type': 'json_schema',
                    'name': 'noc_extraction_schema',
                    'strict': True,
                    'schema': {
                        'type': 'object',
                        'properties': {
                            'noc_number': {'type': 'string'},
                            'project_name': {'type': 'string'},
                            'issue_date': {'type': 'string', 'description': 'DD-MM-YYYY when possible'},
                            'noc_type': {'type': 'string'},
                            'validity_end_date': {'type': 'string', 'description': 'DD-MM-YYYY when possible'},
                            'comments': {'type': 'string'},
                        },
                        'required': ['noc_number', 'project_name', 'issue_date', 'noc_type', 'validity_end_date', 'comments'],
                        'additionalProperties': False,
                    },
                },
            },
            store=False,
        )

        # Best-effort JSON extraction from Responses output
        parsed: Dict[str, Any] = {}
        try:
            text_value = _extract_responses_text(res)
            if text_value:
                parsed = json.loads(text_value)
        except Exception as parse_err:
            logger.warning(f'Fallback parsing of response failed: {parse_err}')
            raise

        # Attach timestamp for provenance
        parsed['timestamp'] = datetime.now(UAE_TZ).isoformat()
        return parsed

    except Exception as e:
        logger.error(f'Error parsing NOC with AI: {e}')
        return {}
    finally:
        # Cleanup uploaded file
        if uploaded_file_id:
            try:
                await client.files.delete(uploaded_file_id)
            except Exception as cleanup_err:
                logger.warning(f'Failed to delete uploaded file: {cleanup_err}')

# ================= Conversation helpers =================
def markdown_to_slack(text: str) -> str:
    """Convert markdown formatting to Slack formatting."""
    # Bold: **text** or __text__ -> *text*
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    text = re.sub(r'__(.+?)__', r'*\1*', text)
    
    # Italic: *text* or _text_ -> _text_
    # First protect already converted bold
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'_\1_', text)
    
    # Code blocks: ```code``` -> ```code```
    # Slack uses the same syntax, so no change needed
    
    # Inline code: `code` -> `code`
    # Slack uses the same syntax, so no change needed
    
    # Headers: # Header -> *Header*
    text = re.sub(r'^#{1,6}\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
    
    # Links: [text](url) -> <url|text>
    text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<\2|\1>', text)
    
    # Strikethrough: ~~text~~ -> ~text~
    text = re.sub(r'~~(.+?)~~', r'~\1~', text)
    
    # Blockquotes: > text -> >>> text (for multiline) or > text (for single line)
    lines = text.split('\n')
    in_blockquote = False
    result = []
    blockquote_lines = []
    
    for line in lines:
        if line.startswith('> '):
            blockquote_lines.append(line[2:])
            in_blockquote = True
        else:
            if in_blockquote:
                if len(blockquote_lines) == 1:
                    result.append('> ' + blockquote_lines[0])
                else:
                    result.append('>>> ' + '\n'.join(blockquote_lines))
                blockquote_lines = []
                in_blockquote = False
            result.append(line)
    
    # Handle any remaining blockquote
    if blockquote_lines:
        if len(blockquote_lines) == 1:
            result.append('> ' + blockquote_lines[0])
        else:
            result.append('>>> ' + '\n'.join(blockquote_lines))
    
    return '\n'.join(result)

def append_to_history(user_id: str, role: str, content: str) -> None:
    if user_id not in user_history:
        user_history[user_id] = []
    user_history[user_id].append({"role": role, "content": content})
    if len(user_history[user_id]) > config.USER_HISTORY_LIMIT:
        user_history[user_id] = user_history[user_id][-20:]

def render_summary(noc_data: Dict[str, Any]) -> str:
    return (
        f"• NOC Number: `{noc_data.get('noc_number','') or 'N/A'}`\n"
        f"• Project Name: `{noc_data.get('project_name','') or 'N/A'}`\n"
        f"• Issue Date: `{noc_data.get('issue_date','') or 'N/A'}`\n"
        f"• NOC Type: `{noc_data.get('noc_type','') or 'N/A'}`\n"
        f"• Validity End Date: `{noc_data.get('validity_end_date','') or 'N/A'}`\n" +
        (f"• Comments: {noc_data.get('comments','')[:300]}\n" if noc_data.get('comments') else '')
    )

def format_noc_results(noc_data: Dict[str, Any]) -> str:
    return (
        "**NOC Extraction Result**\n\n" + render_summary(noc_data) +
        "Reply 'yes' to confirm saving, or provide corrections (e.g., 'Change date to 01-05-2024' or 'NOC number is 123'), or say 'cancel'."
    )

def render_admin_summary(admin_data: Dict[str, Any]) -> str:
    """Format admin data for display."""
    return (
        f"• Username: `{admin_data.get('username', 'N/A')}`\n"
        f"• Slack User ID: `{admin_data.get('slack_user_id', 'N/A')}`\n"
        f"• Notification Channel: `{admin_data.get('notification_channel', 'N/A')}`\n"
        f"• Active: `{admin_data.get('active', True)}`"
    )

def format_admin_confirmation(admin_data: Dict[str, Any]) -> str:
    """Format admin data for confirmation."""
    return (
        "**New Admin User**\n\n" + 
        render_admin_summary(admin_data) + "\n\n" +
        "```json\n" +
        json.dumps(admin_data, indent=2) + "\n" +
        "```\n\n" +
        "Reply 'yes' to add this admin, provide corrections (e.g., 'change username to john_doe'), or say 'cancel'."
    )

async def llm_confirmation_decision(user_id: str, user_text: str, pending: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to decide whether the user is confirming, editing fields, or cancelling."""
    today_str = datetime.now(UAE_TZ).strftime('%B %d, %Y')
    system_prompt = (
        "You are a friendly, concise Slack assistant helping confirm or edit extracted NOC fields. "
        f"Today's date is {today_str}. "
        "Return STRICT JSON describing the action to take. "
        "Allowed actions: 'confirm', 'cancel', 'edit', 'clarify'. "
        "If the user confirms, set action to 'confirm'. "
        "If the user cancels, set action to 'cancel'. "
        "If the user provides corrections, set action to 'edit' and include only corrected fields in 'fields'. "
        "If you need more info, set action to 'clarify'. "
        "Always craft a short, friendly Slack-style 'message' (<= 2 sentences) that acknowledges the user's intent and guides the next step. "
        "When asking to confirm, include a brief Slack-formatted summary of the current fields inline in 'message' (use backticks for values). "
        "When action is 'edit', briefly summarize applied changes in 'message'. "
        "When action is 'clarify', ask exactly one specific question."
    )

    pending_text = (
        "Pending NOC fields to confirm:\n" + render_summary(pending) +
        "User message: " + (user_text or '')
    )

    # Build conversation with short recent history
    history = user_history.get(user_id, [])[-config.LLM_HISTORY_LIMIT:]
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": pending_text}
    ]

    res = await client.responses.create(
        model=config.OPENAI_MODEL,
        input=messages,
        text={
            'format': {
                'type': 'json_schema',
                'name': 'noc_confirmation_schema',
                'strict': False,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'action': {'type': 'string', 'enum': ['confirm', 'cancel', 'edit', 'clarify']},
                        'fields': {
                            'type': 'object',
                            'properties': {
                                'noc_number': {'type': 'string'},
                                'project_name': {'type': 'string'},
                                'issue_date': {'type': 'string'},
                                'noc_type': {'type': 'string'},
                                'validity_end_date': {'type': 'string'},
                                'comments': {'type': 'string'},
                            },
                            'additionalProperties': False
                        },
                        'message': {'type': 'string'}
                    },
                    'required': ['action'],
                    'additionalProperties': False
                },
            }
        },
        store=False
    )

    # Parse JSON result
    parsed: Dict[str, Any] = {}
    try:
        text_value = _extract_responses_text(res)
        if text_value:
            parsed = json.loads(text_value)
    except Exception as e:
        logger.error(f'Error parsing confirmation response: {e}')
        parsed = {'action': 'clarify', 'message': "I couldn't understand. Please reply 'yes' to save, or provide corrected fields, or say 'cancel'."}

    return parsed

async def llm_admin_confirmation_decision(user_id: str, user_text: str, pending_admin: Dict[str, Any]) -> Dict[str, Any]:
    """Use LLM to decide whether the user is confirming, editing admin fields, or cancelling."""
    system_prompt = (
        "You are a friendly, concise Slack assistant helping confirm or edit admin user details. "
        "Return STRICT JSON describing the action to take. "
        "Allowed actions: 'confirm', 'cancel', 'edit', 'clarify'. "
        "If the user confirms, set action to 'confirm'. "
        "If the user cancels, set action to 'cancel'. "
        "If the user provides corrections, set action to 'edit' and include only corrected fields in 'fields'. "
        "If you need more info, set action to 'clarify'. "
        "Always craft a short, friendly Slack-style 'message' (<= 2 sentences). "
        "When asking to confirm, include the current admin details in your message. "
        "When action is 'edit', briefly summarize applied changes in 'message'. "
    )
    
    pending_text = (
        "Pending admin user to confirm:\n" + render_admin_summary(pending_admin) +
        "\nUser message: " + (user_text or '')
    )
    
    # Build conversation
    history = user_history.get(user_id, [])[-4:]
    messages = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": pending_text}
    ]
    
    res = await client.responses.create(
        model=config.OPENAI_MODEL,
        input=messages,
        text={
            'format': {
                'type': 'json_schema',
                'name': 'admin_confirmation_schema',
                'strict': False,
                'schema': {
                    'type': 'object',
                    'properties': {
                        'action': {'type': 'string', 'enum': ['confirm', 'cancel', 'edit', 'clarify']},
                        'fields': {
                            'type': 'object',
                            'properties': {
                                'username': {'type': 'string'},
                                'slack_user_id': {'type': 'string'},
                                'notification_channel': {'type': 'string'},
                                'active': {'type': 'boolean'}
                            },
                        },
                        'message': {'type': 'string'}
                    },
                    'required': ['action'],
                    'additionalProperties': False
                },
            }
        },
        store=False
    )
    
    # Parse JSON result
    parsed: Dict[str, Any] = {}
    try:
        text_value = _extract_responses_text(res)
        if text_value:
            parsed = json.loads(text_value)
    except Exception as e:
        logger.error(f'Error parsing admin confirmation response: {e}')
        parsed = {'action': 'clarify', 'message': "I couldn't understand. Please reply 'yes' to add this admin, provide corrections, or say 'cancel'."}
    
    return parsed

# ================= Utility functions =================
def cleanup_file(file_path: Optional[str]) -> None:
    """Safely remove a file if it exists."""
    if file_path:
        try:
            os.remove(file_path)
        except Exception:
            pass

async def upload_file_to_slack(slack_client: AsyncWebClient, channel: str, file_data: bytes, 
                              filename: str, title: str, initial_comment: str = "") -> bool:
    """Upload a file to Slack channel."""
    import tempfile
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1])
        temp_file.write(file_data)
        temp_file.close()
        
        response = await slack_client.files_upload_v2(
            channel=channel,
            file=temp_file.name,
            filename=filename,
            title=title,
            initial_comment=initial_comment
        )
        return response.get('ok', False)
    except Exception as e:
        logger.error(f"Failed to upload file to Slack: {e}")
        return False
    finally:
        if temp_file:
            cleanup_file(temp_file.name)

# ================= Slack helpers =================
async def download_slack_file(file_info: Dict[str, Any]) -> str:
    """Download a Slack file locally and return the path."""
    try:
        file_id = file_info.get('id')
        file_name = file_info.get('name', 'document.pdf')
        file_response = await slack_client.files_info(file=file_id)
        if not file_response.get('ok'):
            raise RuntimeError(f'Failed to get file info: {file_response}')

        file_data = file_response['file']
        
        # Check file size (if limit is set)
        if config.MAX_FILE_SIZE_MB > 0:
            file_size = file_data.get('size', 0)
            max_size_bytes = config.MAX_FILE_SIZE_MB * 1024 * 1024
            if file_size > max_size_bytes:
                raise RuntimeError(f'File too large: {file_size / (1024*1024):.1f}MB (max {config.MAX_FILE_SIZE_MB}MB)')
        
        file_url = file_data.get('url_private_download') or file_data.get('url_private')
        if not file_url:
            raise RuntimeError('No downloadable URL for file')

        os.makedirs(config.UPLOAD_DIR, exist_ok=True)
        safe_name = ''.join(ch if ch.isalnum() or ch in ('-', '_', '.') else '_' for ch in file_name)
        timestamp = datetime.now(UAE_TZ).strftime('%Y%m%d_%H%M%S')
        local_path = os.path.join(config.UPLOAD_DIR, f'noc_{timestamp}_{safe_name}')

        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        resp = requests.get(file_url, headers=headers)
        resp.raise_for_status()
        with open(local_path, 'wb') as f_out:
            f_out.write(resp.content)
        logger.info(f'Downloaded Slack file to {local_path}')
        return local_path
    except Exception as e:
        logger.error(f'Error downloading Slack file: {e}')
        return ''

async def handle_slack_message(channel: str, user_id: str, text: str, files: Optional[List[Dict[str, Any]]]) -> None:
    """Handle Slack message: manage confirmation flow and file parsing via LLM."""
    # Resolve user display name
    try:
        user_info = await slack_client.users_info(user=user_id)
        profile = user_info["user"]["profile"]
        display_name = profile.get("display_name") or profile.get("real_name") or "Unknown"
    except Exception:
        display_name = "Unknown"

    # Append user message to history
    if text:
        append_to_history(user_id, 'user', text)

    # Confirmation loop via LLM
    # Check for pending admin confirmations first
    if user_id in pending_admin_confirmations:
        pending_admin = pending_admin_confirmations[user_id]
        decision = await llm_admin_confirmation_decision(user_id, text or '', pending_admin)
        action = decision.get('action')
        message = decision.get('message')
        
        if action == 'confirm':
            # Add the admin to config
            success = await asyncio.to_thread(permissions.add_admin, 
                pending_admin['username'],
                pending_admin['slack_user_id'],
                pending_admin['notification_channel'],
                pending_admin.get('active', True)
            )
            
            if success:
                reply = f"✅ Admin user `{pending_admin['username']}` has been added successfully!"
            else:
                reply = "❌ Failed to add admin user. Please check the logs."
            
            pending_admin_confirmations.pop(user_id, None)
            append_to_history(user_id, 'assistant', reply)
            await slack_client.chat_postMessage(channel=channel, text=reply)
            return
            
        elif action == 'cancel':
            reply = "Admin addition cancelled."
            pending_admin_confirmations.pop(user_id, None)
            append_to_history(user_id, 'assistant', reply)
            await slack_client.chat_postMessage(channel=channel, text=reply)
            return
            
        elif action == 'edit':
            fields = decision.get('fields', {})
            if fields:
                pending_admin.update(fields)
                pending_admin_confirmations[user_id] = pending_admin
            if message:
                append_to_history(user_id, 'assistant', message)
                await slack_client.chat_postMessage(channel=channel, text=message)
                # Show updated admin data
                updated_msg = format_admin_confirmation(pending_admin)
                await slack_client.chat_postMessage(channel=channel, text=updated_msg)
            return
            
        elif action == 'clarify':
            if message:
                append_to_history(user_id, 'assistant', message)
                await slack_client.chat_postMessage(channel=channel, text=message)
            return
    
    # Check for pending NOC confirmations
    if user_id in pending_confirmations:
        pending = pending_confirmations[user_id]
        decision = await llm_confirmation_decision(user_id, text or '', pending)
        action = decision.get('action')
        message = decision.get('message')

        if action == 'confirm':
            pending['submitted_by'] = display_name
            document_path = pending.pop('_document_path', None)
            record_id = await save_to_database(pending, document_path)
            
            # Clean up document file after saving
            cleanup_file(document_path)
            
            if record_id:
                reply = f"Saved. Record ID: #{record_id}. NOC: `{pending.get('noc_number','')}`"
            else:
                reply = 'Failed to save NOC information. Please try again.'
            pending_confirmations.pop(user_id, None)
            append_to_history(user_id, 'assistant', reply)
            await slack_client.chat_postMessage(channel=channel, text=reply)
            return
        elif action == 'cancel':
            # Clean up document file on cancel
            document_path = pending.pop('_document_path', None)
            cleanup_file(document_path)
            
            pending_confirmations.pop(user_id, None)
            reply = message or 'Cancelled. No data was saved.'
            append_to_history(user_id, 'assistant', reply)
            await slack_client.chat_postMessage(channel=channel, text=reply)
            return
        elif action == 'edit':
            fields = decision.get('fields', {}) or {}
            for k in ('noc_number', 'project_name', 'issue_date', 'noc_type', 'validity_end_date', 'comments'):
                if k in fields and isinstance(fields[k], str) and fields[k].strip() != '':
                    pending[k] = fields[k].strip()
            # Keep pending, ask for confirm again (let LLM craft the message)
            reply = message or "Please review the updated fields and confirm."
            append_to_history(user_id, 'assistant', reply)
            await slack_client.chat_postMessage(channel=channel, text=reply)
            return
        else:  # clarify or unknown
            reply = message or "Please reply 'yes' to save, provide corrections, or say 'cancel'."
            append_to_history(user_id, 'assistant', reply)
            await slack_client.chat_postMessage(channel=channel, text=reply)
            return

    # If PDF file is attached, process it
    if files:
        pdf_file = next((f for f in files if (f.get('mimetype') == 'application/pdf' or f.get('name','').lower().endswith('.pdf'))), None)
        if pdf_file:
            # Check if user is admin before processing PDFs
            if not permissions.is_admin(user_id):
                reply = "❌ Sorry, only admins can process NOC documents. Please contact an administrator."
                await slack_client.chat_postMessage(channel=channel, text=reply)
                return
                
            local_path = await download_slack_file(pdf_file)
            if local_path and local_path.lower().endswith('.pdf'):
                noc_data = await parse_noc_with_ai(local_path)
                
                if noc_data:
                    # Store file path for later saving
                    pending_confirmations[user_id] = noc_data
                    pending_confirmations[user_id]['_document_path'] = local_path
                    # Ask LLM to craft the initial confirmation ask, including a brief summary
                    decision = await llm_confirmation_decision(user_id, '', noc_data)
                    reply = decision.get('message') or "I've extracted the fields. Please review and confirm."
                    append_to_history(user_id, 'assistant', reply)
                    await slack_client.chat_postMessage(channel=channel, text=reply)
                else:
                    # Clean up file on failure
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
                    fail_msg = 'Failed to extract NOC information from the PDF.'
                    append_to_history(user_id, 'assistant', fail_msg)
                    await slack_client.chat_postMessage(channel=channel, text=fail_msg)
                return

    # Use LLM with tools to handle all text-based requests
    is_admin = permissions.is_admin(user_id)
    reply = await llm_general_response(user_id, text or '', channel, is_admin)
    append_to_history(user_id, 'assistant', reply)
    await slack_client.chat_postMessage(channel=channel, text=reply)

async def llm_general_response(user_id: str, user_text: str, channel: str, is_admin: bool = False) -> str:
    """Use LLM with tools to handle various user requests."""
    today_str = datetime.now(UAE_TZ).strftime('%B %d, %Y')
    
    admin_note = ""
    if is_admin:
        admin_note = """
        Admin features available:
        - Export database to Excel
        - Retrieve specific NOC documents by reference number
        - View detailed statistics
        """
    
    system_prompt = f"""
You are a friendly Slack assistant that helps with NOC (No Objection Certificate) document processing.
Today's date is {today_str}.

FEATURES:
- Process NOC PDFs: Extract details from uploaded documents
- Database Summary: Show statistics and recent NOCs
- Export Database: Export all records to Excel (admin only)
- Retrieve Documents: Get specific NOC documents by reference number (admin only)

{admin_note}

IMPORTANT:
- Keep responses brief and friendly
- Use Slack formatting (backticks, bullets)
- When user mentions a NOC reference number with intent to retrieve, use get_noc_document tool
- When user wants to export/download the database, use export_database tool
- When user wants statistics/summary, use get_summary tool
"""
    
    history = user_history.get(user_id, [])[-config.LLM_HISTORY_LIMIT:]
    messages = ([{"role": "system", "content": system_prompt}] + history +
                [{"role": "user", "content": user_text or ' '}])
    
    tools = []
    
    # Add admin tools if user is admin
    if is_admin:
        tools.extend([
            {
                "type": "function",
                "name": "get_noc_document",
                "description": "Retrieve a specific NOC document by reference number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "noc_number": {
                            "type": "string",
                            "description": "The NOC reference number to retrieve"
                        }
                    },
                    "required": ["noc_number"]
                }
            },
            {
                "type": "function",
                "name": "export_database",
                "description": "Export the entire NOC database to Excel file",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            },
            {
                "type": "function",
                "name": "add_admin",
                "description": "Add a new admin user to the system",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "username": {
                            "type": "string",
                            "description": "Username for the admin (lowercase, no spaces)"
                        },
                        "slack_user_id": {
                            "type": "string",
                            "description": "Slack user ID (starts with U)"
                        },
                        "notification_channel": {
                            "type": "string",
                            "description": "Slack channel ID for notifications (starts with C)"
                        },
                        "active": {
                            "type": "boolean",
                            "description": "Whether the admin is active",
                            "default": True
                        }
                    },
                    "required": ["username", "slack_user_id", "notification_channel"]
                }
            },
            {
                "type": "function",
                "name": "get_summary",
                "description": "Get database summary with statistics",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        ])
    
    try:
        res = await client.responses.create(
            model=config.OPENAI_MODEL,
            input=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            store=False
        )
        
        # Check if LLM wants to use a tool
        if res.output and len(res.output) > 0:
            msg = res.output[0]
            
            if hasattr(msg, 'type') and msg.type == "function_call":
                if msg.name == "get_noc_document" and is_admin:
                    args = json.loads(msg.arguments)
                    noc_number = args.get("noc_number", "").strip()
                    
                    # Get document
                    doc_result = await asyncio.to_thread(db.get_noc_document, noc_number)
                    
                    if doc_result:
                        document_data, filename = doc_result
                        
                        # Get NOC details
                        noc_details = await asyncio.to_thread(db.get_noc_by_number, noc_number)
                        
                        # Upload to Slack
                        initial_comment = f"**NOC Details:**\n• Project: {noc_details.get('project_name', 'N/A') if noc_details else 'N/A'}\n• Type: {noc_details.get('noc_type', 'N/A') if noc_details else 'N/A'}\n• Issue Date: {noc_details.get('issue_date', 'N/A') if noc_details else 'N/A'}\n• Validity End Date: {noc_details.get('validity_end_date', 'N/A') if noc_details else 'N/A'}"
                        
                        await upload_file_to_slack(
                            slack_client, channel, document_data, 
                            filename, f'NOC {noc_number}', initial_comment
                        )
                        return f"✅ Found and uploaded NOC `{noc_number}`"
                    else:
                        return f"❌ Could not find document for NOC `{noc_number}`. It may not have a stored document or doesn't exist."
                
                elif msg.name == "export_database" and is_admin:
                    # Export database
                    excel_path = await asyncio.to_thread(db.export_to_excel)
                    
                    # Upload to Slack
                    with open(excel_path, 'rb') as f:
                        excel_data = f.read()
                    
                    await upload_file_to_slack(
                        slack_client, channel, excel_data,
                        f'noc_export_{datetime.now(UAE_TZ).strftime("%Y%m%d_%H%M%S")}.xlsx',
                        'NOC Database Export',
                        'Here is the complete NOC database export.'
                    )
                    
                    cleanup_file(excel_path)
                    
                    total = len(await asyncio.to_thread(db.get_all_noc_extractions))
                    return f"✅ Database exported successfully. Total records: {total}"
                
                elif msg.name == "add_admin" and is_admin:
                    args = json.loads(msg.arguments)
                    
                    # Validate required fields
                    username = args.get("username", "").strip().lower().replace(" ", "_")
                    slack_user_id = args.get("slack_user_id", "").strip()
                    notification_channel = args.get("notification_channel", "").strip()
                    active = args.get("active", True)
                    
                    if not username:
                        return "❌ Username is required. Please provide a username."
                    if not slack_user_id:
                        return "❌ Slack user ID is required. Use /ops_my_ids to get the user ID."
                    if not notification_channel:
                        return "❌ Notification channel is required. Use /ops_my_ids to get the channel ID."
                    
                    # Prepare admin data for confirmation
                    admin_data = {
                        "username": username,
                        "slack_user_id": slack_user_id,
                        "notification_channel": notification_channel,
                        "active": active
                    }
                    
                    # Store in pending confirmations
                    pending_admin_confirmations[user_id] = admin_data
                    
                    # Send confirmation message
                    confirmation_msg = format_admin_confirmation(admin_data)
                    return confirmation_msg
                
                elif msg.name == "get_summary":
                    summary = await asyncio.to_thread(db.get_noc_summary)
                    
                    reply = f"📊 *NOC Database Summary*\n\n"
                    reply += f"Total NOCs: *{summary['total_nocs']}*\n"
                    reply += f"Archived NOCs: *{summary.get('history_nocs', 0)}*\n\n"
                    
                    if summary['by_type']:
                        reply += "*By Type:*\n"
                        for noc_type, count in summary['by_type'].items():
                            reply += f"• {noc_type}: {count}\n"
                        reply += "\n"
                    
                    if summary['expiring_soon']:
                        reply += f"*Expiring Soon ({len(summary['expiring_soon'])} NOCs):*\n"
                        for noc in summary['expiring_soon'][:3]:
                            reply += f"• `{noc['noc_number']}` - {noc['project_name'] or 'N/A'} (expires: {noc['validity_end_date']})\n"
                        if len(summary['expiring_soon']) > 3:
                            reply += f"_...and {len(summary['expiring_soon']) - 3} more_\n"
                        reply += "\n"
                    
                    if summary['recent_nocs']:
                        reply += "*Recent NOCs:*\n"
                        for noc in summary['recent_nocs'][:3]:
                            reply += f"• `{noc['noc_number']}` - {noc['project_name'] or 'N/A'}\n"
                    
                    return reply
            
            # Extract regular text response
            text_value = _extract_responses_text(res)
            if text_value:
                return text_value
        
    except Exception as e:
        logger.warning(f'LLM response failed: {e}')
    
    return 'Hi! You can upload a NOC PDF and I will extract the NOC number, issue date, and type.'

# ================= FastAPI endpoints =================
@api.post('/cron/check-expiry')
async def cron_check_expiry(request: Request):
    """Cron endpoint to check for expiring and expired NOCs."""
    try:
        # Verify cron secret if provided
        cron_secret = os.environ.get('CRON_SECRET')
        if cron_secret:
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer ') or auth_header[7:] != cron_secret:
                return JSONResponse({'error': 'Unauthorized'}, status_code=401)
        
        # Archive expired NOCs
        archived = await asyncio.to_thread(db.archive_expired_nocs)
        
        # Notify admins about archived NOCs
        if archived:
            admin_channels = permissions.get_admin_notification_channels()
            if not admin_channels:
                logger.warning("[CRON] No admin channels configured for notifications")
            else:
                for noc_number in archived:
                    # Get NOC details before it was archived
                    msg = f"📦 *NOC Archived*\nNOC `{noc_number}` has expired and been moved to history."
                    
                    # Try to get document and send it
                    doc_result = await asyncio.to_thread(db.get_noc_document, noc_number)
                    
                    notification_sent = False
                    for channel in admin_channels:
                        try:
                            if doc_result:
                                document_data, filename = doc_result
                                await upload_file_to_slack(
                                    slack_client, channel, document_data,
                                    filename, f'Archived NOC {noc_number}', msg
                                )
                            else:
                                await slack_client.chat_postMessage(channel=channel, text=msg)
                            notification_sent = True
                        except Exception as e:
                            logger.error(f'Failed to send archive notification to {channel}: {e}')
                    
                    if not notification_sent:
                        logger.error(f"Failed to send archive notification for NOC {noc_number} to any channel")
        
        # Check for NOCs expiring in configured days
        expiring = await asyncio.to_thread(db.get_expiring_nocs, config.EXPIRY_NOTIFICATION_DAYS)
        notified = []
        
        today_str = datetime.now().date().strftime('%d-%m-%Y')
        admin_channels = permissions.get_admin_notification_channels()
        
        if not admin_channels:
            logger.warning("[CRON] No admin channels configured for expiry notifications")
        
        for noc in expiring:
            noc_number = noc.get('noc_number', '')
            last_notified = noc.get('last_notified_date', '')
            
            # Only notify once per day
            if last_notified == today_str:
                continue
            
            project_name = noc.get('project_name', 'N/A')
            validity_end = noc.get('validity_end_date', '')
            days_left = (datetime.strptime(validity_end, '%d-%m-%Y').date() - datetime.now().date()).days
            
            # Create appropriate message based on days left
            if days_left == 0:
                msg = (
                    f"🚨 *NOC EXPIRES TODAY*\n"
                    f"NOC `{noc_number}` for project `{project_name}` expires *TODAY* ({validity_end}).\n"
                    f"This NOC will be archived tomorrow."
                )
            elif days_left == 1:
                msg = (
                    f"⚠️ *NOC Expiring Tomorrow*\n"
                    f"NOC `{noc_number}` for project `{project_name}` expires *tomorrow* ({validity_end})."
                )
            else:
                msg = (
                    f"⚠️ *NOC Expiring Soon*\n"
                    f"NOC `{noc_number}` for project `{project_name}` expires in *{days_left} days* "
                    f"(on {validity_end})."
                )
            
            # Get document
            doc_result = await asyncio.to_thread(db.get_noc_document, noc_number)
            
            # Send to all admin channels
            notification_sent = False
            for channel in admin_channels:
                try:
                    if doc_result:
                        document_data, filename = doc_result
                        await upload_file_to_slack(
                            slack_client, channel, document_data,
                            filename, f'NOC {noc_number} - Expiring Soon', msg
                        )
                    else:
                        await slack_client.chat_postMessage(channel=channel, text=msg)
                    notification_sent = True
                except Exception as e:
                    logger.error(f'Failed to send expiry notification to {channel}: {e}')
            
            # Update last notified date only if we sent at least one notification
            if notification_sent:
                await asyncio.to_thread(db.update_last_notified, noc_number, today_str)
                notified.append(noc_number)
            else:
                logger.error(f"Failed to send expiry notification for NOC {noc_number} to any channel")
        
        return JSONResponse({
            'success': True,
            'archived': len(archived),
            'notified_expiring': len(notified),
            'timestamp': datetime.now(UAE_TZ).isoformat()
        })
        
    except Exception as e:
        logger.error(f'Cron check-expiry error: {e}')
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)


@api.post('/api/parse_noc')
async def api_parse_noc(request: NOCParseRequest):
    """Parse a NOC PDF and optionally save results."""
    try:
        if not request.file_path:
            return JSONResponse({
                'success': False,
                'message': 'No file path provided',
            }, status_code=400)

        noc_data = await parse_noc_with_ai(request.file_path)
        if not noc_data:
            return JSONResponse({
                'success': False,
                'message': 'Failed to parse NOC document',
            }, status_code=500)

        if request.save_to_excel:
            noc_data['submitted_by'] = request.submitted_by
            record_id = await save_to_database(noc_data, request.file_path)
            noc_data['record_id'] = record_id

        return JSONResponse({'success': True, 'data': noc_data})
    except Exception as e:
        logger.exception('Unhandled error in api_parse_noc')
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@api.get('/api/get_nocs')
async def api_get_nocs():
    """Retrieve all stored NOC extractions."""
    try:
        records = await asyncio.to_thread(db.get_all_noc_extractions)
        return JSONResponse({'success': True, 'records': records, 'count': len(records)})
    except Exception as e:
        return JSONResponse({'success': False, 'error': str(e)}, status_code=500)

@api.get('/api/health')
async def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now(UAE_TZ).isoformat()}

# ================= Slack endpoints =================
@api.post('/slack/events')
async def slack_events(request: Request):
    """Slack events endpoint for messages and file uploads."""
    body_bytes = await request.body()
    timestamp = request.headers.get('X-Slack-Request-Timestamp', '')
    signature = request.headers.get('X-Slack-Signature', '')

    if not signature_verifier.is_valid(body_bytes.decode(), timestamp, signature):
        return JSONResponse({'error': 'Invalid signature'}, status_code=403)

    data = await request.json()
    if data.get('type') == 'url_verification':
        return JSONResponse({'challenge': data.get('challenge')})

    if data.get('type') == 'event_callback':
        event = data.get('event', {})
        if event.get('type') == 'message' and not event.get('bot_id'):
            user_id = event.get('user')
            channel = event.get('channel')
            text = event.get('text', '')
            files = event.get('files', [])
            asyncio.create_task(handle_slack_message(channel, user_id, text, files))
            return JSONResponse({'status': 'ok'})

    return JSONResponse({'status': 'ok'})

@api.post('/slack/commands')
async def slack_commands(request: Request):
    """Handle Slack slash commands."""
    # Verify the request signature first (like in VideoCritique)
    body = await request.body()
    timestamp = request.headers.get('X-Slack-Request-Timestamp', '')
    signature = request.headers.get('X-Slack-Signature', '')
    
    if not signature_verifier.is_valid(body.decode(), timestamp, signature):
        logger.error(f"[SLASH] Signature verification failed")
        return JSONResponse({'error': 'Invalid signature'}, status_code=403)
    
    # Parse form data after verification
    form_data = await request.form()
    
    # Extract data from form
    command = form_data.get('command', '')
    user_id = form_data.get('user_id', '')
    channel_id = form_data.get('channel_id', '')
    
    if command == "/ops_my_ids":
        try:
            # Get user info
            user_info = await slack_client.users_info(user=user_id)
            user_name = user_info["user"]["profile"].get("real_name", "Unknown")
            user_email = user_info["user"]["profile"].get("email", "Not available")
            
            # Get channel info
            channel_type = "Unknown"
            channel_name = "Unknown"
            try:
                channel_info = await slack_client.conversations_info(channel=channel_id)
                if channel_info["ok"]:
                    chan = channel_info["channel"]
                    channel_name = chan.get("name", "Direct Message")
                    if chan.get("is_channel"):
                        channel_type = "Public Channel"
                    elif chan.get("is_group"):
                        channel_type = "Private Channel"
                    elif chan.get("is_im"):
                        channel_type = "Direct Message"
                    elif chan.get("is_mpim"):
                        channel_type = "Group DM"
            except:
                pass
            
            id_message = (f"🆔 *Your Slack Information*\n\n"
                         f"*User Details:*\n"
                         f"• Name: {user_name}\n"
                         f"• Email: {user_email}\n"
                         f"• User ID: `{user_id}`\n\n"
                         f"*Channel Information:*\n"
                         f"• Channel: {channel_name}\n"
                         f"• Type: {channel_type}\n"
                         f"• Channel ID: `{channel_id}`\n\n"
                         f"📋 *Copyable Format for admin_config.json:*\n"
                         f"```\n"
                         f'"admins": {{\n'
                         f'  "{user_name.lower().replace(" ", "_")}": {{\n'
                         f'    "active": true,\n'
                         f'    "slack_user_id": "{user_id}",\n'
                         f'    "notification_channel": "{channel_id}"\n'
                         f'  }}\n'
                         f'}}\n'
                         f"```\n\n"
                         f"💡 *Next Steps:*\n"
                         f"1. Copy the above JSON snippet\n"
                         f"2. Add it to your admin_config.json file in /data/\n"
                         f"3. Save the file to enable admin access")
            
            return JSONResponse({
                "response_type": "ephemeral",
                "text": id_message
            })
            
        except Exception as e:
            logger.error(f"Error getting user IDs: {e}")
            return JSONResponse({
                "response_type": "ephemeral",
                "text": f"❌ Error retrieving your information: {str(e)}"
            })
    
    else:
        return JSONResponse({
            "response_type": "ephemeral",
            "text": f"Unknown command: {command}"
        })

# ================= Entrypoint =================
async def main():
    # Migrate existing Excel data to database if needed
    excel_path = 'noc_extractions.xlsx'
    if Path(excel_path).exists():
        try:
            count = await asyncio.to_thread(db.migrate_from_excel, excel_path)
            if count > 0:
                logger.info(f"Migrated {count} records from Excel to database")
        except Exception as e:
            logger.error(f"Error migrating Excel data: {e}")
    
    # Run FastAPI via uvicorn
    import uvicorn
    config_obj = uvicorn.Config(app=api, host='0.0.0.0', port=config.API_PORT, log_level='info')
    server = uvicorn.Server(config_obj)

    logger.info(f'NOC Reader API running on http://localhost:{config.API_PORT}')
    logger.info(f'API docs available at http://localhost:{config.API_PORT}/docs')
    logger.info(f'Slack events endpoint: http://localhost:{config.API_PORT}/slack/events')

    await server.serve()

if __name__ == '__main__':
    asyncio.run(main())