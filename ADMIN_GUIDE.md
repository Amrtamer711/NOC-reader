# NOC Reader Admin Guide

## Admin Configuration

The admin configuration is stored in `admin_config.json`. To grant admin privileges to a user:

1. Find the user's Slack User ID (you can get this from Slack)
2. Add them to the `admins` section in `admin_config.json`:

```json
{
  "permissions": {
    "export_database": ["admins"]
  },
  "admins": {
    "username": {
      "active": true,
      "slack_user_id": "U12345678"
    }
  }
}
```

## Admin Commands

Admin users can use these commands in Slack:

### Export Database
- **Commands**: `export`, `export excel`, `export database`, `download excel`
- **Description**: Exports the entire NOC database to an Excel file and uploads it to the Slack channel
- **Permission**: Admin only

### Database Summary
- **Commands**: `summary`, `stats`, `status`  
- **Description**: Shows a summary of the database including:
  - Total number of NOCs
  - Breakdown by NOC type
  - NOCs expiring soon
  - Recent NOCs
- **Permission**: All users

## Database Migration

When first running with the database:
1. The system will automatically migrate any existing Excel data to the SQLite database
2. The migration preserves all existing records
3. Duplicate records (same NOC number and timestamp) are skipped

## Database Structure

The database stores:
- ID (auto-incremented)
- Timestamp
- NOC Number
- Project Name
- Issue Date
- NOC Type
- Validity End Date
- Comments
- Submitted By
- Raw Data (JSON)

## Regular Users

Regular users can:
- Upload NOC PDFs for extraction
- View summaries and statistics
- Receive expiry notifications (if configured)

Regular users cannot:
- Export the full database to Excel
- Modify admin configurations