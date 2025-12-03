# League Participants Migration Guide

The `league_participants` table is required for the League feature to work. If you see the error "league_participants table does not exist", follow one of these methods:

## Method 1: API Endpoint (Recommended for Production)

Run the migration via API endpoint:

```bash
curl -X POST "https://caria-production.up.railway.app/api/league/migrate?secret_key=YOUR_SECRET_KEY" \
  -H "Content-Type: application/json"
```

**Note:** Set `MIGRATION_SECRET_KEY` environment variable in Railway to secure this endpoint. Default is `dev-migration-key-change-in-prod` (change in production!).

## Method 2: Python Script

If you have access to the Railway CLI or can SSH into the container:

```bash
cd backend/api
python run_league_migration.py
```

The script will use `DATABASE_URL` from environment variables automatically.

## Method 3: Direct SQL (via Railway Dashboard)

1. Go to Railway Dashboard → Your PostgreSQL Service → Query
2. Run this SQL:

```sql
-- Create league_participants table to track opt-in status and anonymity
CREATE TABLE IF NOT EXISTS league_participants (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    is_anonymous BOOLEAN DEFAULT FALSE,
    display_name TEXT,
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster joins
CREATE INDEX IF NOT EXISTS idx_league_participants_joined ON league_participants(joined_at);
```

## Verification

After running the migration, verify it worked:

```sql
SELECT * FROM league_participants LIMIT 1;
```

If no error, the table exists and is ready to use!

