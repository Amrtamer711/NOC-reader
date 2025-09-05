# NOC Reader Bot

A Slack bot that automatically extracts information from NOC (No Objection Certificate) PDF documents, stores them in a database, and provides admin tools for management.

## Features

- **PDF Processing**: Automatically extracts NOC details from uploaded PDF documents using AI
- **Database Storage**: Stores NOC data and documents in SQLite with current/history separation
- **Slack Integration**: Interactive Slack bot for document processing and retrieval
- **Admin Tools**: Export database to Excel, retrieve specific documents
- **Expiry Notifications**: Automatic notifications for NOCs expiring within 14 days
- **Document Archiving**: Automatic archiving of expired NOCs to history database

## Setup

### Environment Variables

Create a `.env` file with the following variables:

```bash
# OpenAI API
OPENAI_API_KEY=your-openai-api-key

# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_SIGNING_SECRET=your-signing-secret

# Google Drive (optional)
GOOGLE_SERVICE_ACCOUNT_JSON=routes-key.json
SHARED_DRIVE_ID=your-shared-drive-id
DRIVE_PARENT_ID=your-drive-folder-id
DRIVE_FILE_ID=existing-file-id-to-update

# Notifications
NOC_ALERT_RECIPIENTS=U123,U456  # Comma-separated Slack user IDs

# Cron (auto-generated on Render)
CRON_SECRET=your-cron-secret
```

### Admin Configuration

Create an `admin_config.json` file:

```json
{
  "permissions": {
    "export_database": ["admins"]
  },
  "admins": {
    "username": {
      "active": true,
      "slack_user_id": "U12345678",
      "notification_channel": "C12345678"
    }
  },
  "default_notification_channel": "C12345678"
}
```

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python noc_reader.py
```

## Usage

### Regular Users

- **Upload NOC**: Upload a PDF file to any channel where the bot is present
- **Confirm/Edit**: Review extracted information and confirm or edit before saving
- **View Summary**: Type `summary`, `stats`, or `status` to see database statistics

### Admin Commands

- **Export Database**: Type `export`, `export excel`, or `download excel`
- **Retrieve Document**: Type `get NOC123` or `fetch NOC123` (replace NOC123 with actual NOC number)

### API Endpoints

- `POST /slack/events` - Slack events webhook
- `POST /api/parse_noc` - Parse NOC via API
- `GET /api/get_nocs` - Get all NOC records
- `GET /api/health` - Health check
- `POST /cron/check-expiry` - Cron endpoint for expiry checks

## Database Structure

The system uses two SQLite databases:

### Current Database (`noc_current.db`)
Stores active NOCs with:
- NOC details (number, project, dates, type)
- Document binary data
- Tracking information

### History Database (`noc_history.db`)
Archives expired NOCs with:
- All original data
- Archive timestamp
- Archive reason

## Deployment

This project is configured for deployment on Render using the included `render.yaml`:

1. Fork this repository
2. Connect to Render
3. Configure environment variables
4. Deploy

The cron job will automatically run daily to check for expiring NOCs.

## Security

- Admin-only commands require user authentication
- Cron endpoints protected with bearer token
- Database uses WAL mode for concurrent access
- Sensitive files excluded via `.gitignore`