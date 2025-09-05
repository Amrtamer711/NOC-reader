# NOC Reader Bot

A Slack bot that extracts and manages Notice of Commencement (NOC) information from PDF documents. The bot uses AI to parse construction permits, stores them in a database, and provides admin tools for data management.

## Features

- **PDF Processing**: Extracts NOC information from uploaded PDF documents using OpenAI Vision API
- **Database Storage**: SQLite backend with separate databases for current and historical NOCs
- **Automated Expiry Management**: Daily cron job checks for expiring NOCs and notifies admins
- **Admin Tools**: Export database to Excel, retrieve specific documents by reference number
- **Natural Language Commands**: Uses LLM tools to understand admin commands without specific syntax
- **Document Archiving**: Automatically moves expired NOC documents to archive storage

## Setup

### Prerequisites

- Python 3.8+
- Slack Bot Token
- OpenAI API Key

### Local Development

1. Clone the repository:
```bash
git clone <your-repo-url>
cd Ops
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

5. Configure environment variables in `.env`:
```
SLACK_BOT_TOKEN=xoxb-your-slack-bot-token
SLACK_APP_TOKEN=xapp-your-slack-app-token
OPENAI_API_KEY=your-openai-api-key
```

6. Create admin configuration:
```bash
cp admin_config.example.json admin_config.json
```

7. Update `admin_config.json` with your admin users:
```json
{
  "permissions": {
    "export_database": ["admins"]
  },
  "admins": {
    "your_username": {
      "active": true,
      "slack_user_id": "U12345678",
      "notification_channel": "C12345678"
    }
  }
}
```

8. Run the bot:
```bash
python noc_reader.py
```

## Deployment on Render

1. Fork/push this repository to GitHub
2. Connect your GitHub repository to Render
3. Render will automatically detect the `render.yaml` configuration
4. Set environment variables in Render dashboard:
   - `SLACK_BOT_TOKEN`
   - `SLACK_APP_TOKEN`
   - `OPENAI_API_KEY`
   - `CRON_SECRET` (for cron authentication)
5. Deploy the service

## Usage

### For Users

1. **Upload NOC**: Simply upload a PDF file to any channel where the bot is present
2. **View Summary**: The bot will automatically extract and display NOC information
3. **Get Summary**: Type "summary" or "show me the summary" to see all active NOCs

### For Admins

Admins can use natural language commands:

- **Export Database**: "export the database", "send me an excel file", etc.
- **Get Document**: "get NOC document ABC123", "show me document for permit XYZ", etc.
- **Add Admin**: "add admin john with user ID U123 and channel C456", etc.
- **View Summary**: Same as regular users

### Slack Commands

- `/my_ids` - Shows your Slack user ID and channel ID (useful for admin configuration)

## Architecture

### Database Structure

The bot uses two SQLite databases:
- `noc_current.db`: Active NOCs
- `noc_history.db`: Expired/archived NOCs

Documents are stored in the filesystem:
- `/data/current_docs/`: Active NOC documents (production)
- `/data/expired_docs/`: Archived NOC documents (production)
- `data/current_docs/`: Active NOC documents (local)
- `data/expired_docs/`: Archived NOC documents (local)

### Automated Tasks

- **Daily Expiry Check**: Runs at 9 AM EST daily
  - Checks for NOCs expiring within 14 days
  - Sends notifications to admin channels
  - Archives NOCs that have expired

### Security

- Admin permissions controlled via `admin_config.json`
- Cron endpoints protected by bearer token authentication
- No hardcoded credentials (all via environment variables)

## Development

### Project Structure

```
Ops/
├── noc_reader.py          # Main application
├── db.py                  # Database operations
├── config.py              # Environment configuration
├── permissions.py         # Admin permissions
├── requirements.txt       # Python dependencies
├── render.yaml           # Render deployment config
├── admin_config.json     # Admin configuration (not in git)
├── .env                  # Environment variables (not in git)
└── data/                 # Local data directory (not in git)
    ├── current_docs/
    └── expired_docs/
```

### Adding New Features

1. **New LLM Tools**: Add tool functions to the `tools` list in `llm_general_response()`
2. **New Permissions**: Add to `permissions` section in `admin_config.json`
3. **Database Changes**: Update schema in `db.py` `_create_tables()` method

## Troubleshooting

### Common Issues

1. **Bot not responding**: Check Slack tokens and bot permissions
2. **PDF extraction failing**: Verify OpenAI API key and credits
3. **Database locked**: The bot uses WAL mode to prevent locking issues
4. **Notifications not sending**: Check admin config and Slack channel IDs

### Logs

The bot logs to `noc_reader_bot.log` with rotation at 10MB.

## License

[Your License]

## Support

For issues or questions, please open a GitHub issue.