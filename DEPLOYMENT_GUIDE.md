# Caria Multi-User Platform - Deployment Guide

## Complete Guide for Production Deployment

This guide will help you deploy Caria as a multi-user investment analysis platform with authentication, multi-tenancy, and all advanced features.

---

## Table of Contents

1. [Quick Start (Development)](#quick-start-development)
2. [Production Deployment with Docker](#production-deployment-with-docker)
3. [Manual Deployment](#manual-deployment)
4. [Security Hardening](#security-hardening)
5. [User Management](#user-management)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start (Development)

### Prerequisites

- Python 3.11+
- Poetry
- Docker & Docker Compose (recommended)
- PostgreSQL 15+ with pgvector (or use Docker)

### Step 1: Clone & Setup

```bash
cd notebooks
cp .env.example .env
```

Edit `.env` and configure:
```bash
# REQUIRED: Change this!
JWT_SECRET_KEY=$(openssl rand -hex 32)

# REQUIRED: Database password
POSTGRES_PASSWORD=your_secure_password_here

# OPTIONAL: LLM (choose one)
GEMINI_API_KEY=your_gemini_key  # OR
# OPENAI_API_KEY=your_openai_key  # OR
# Just install Ollama (local, free)
```

### Step 2: Start Database

```bash
docker-compose up -d postgres
```

Wait for initialization (check logs):
```bash
docker-compose logs -f postgres
```

You should see: "Caria database initialized successfully!"

### Step 3: Install Dependencies

```bash
poetry install
```

### Step 4: Train HMM Model (if not done)

```bash
cd caria_data
poetry run python train_hmm_simple.py
```

### Step 5: Start API

```bash
cd services
poetry run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Step 6: Verify

Open: http://localhost:8000/health

Expected response:
```json
{
  "status": "ok",
  "database": "available",
  "auth": "available",
  "rag": "available",
  "regime": "available",
  "factors": "available",
  "valuation": "available"
}
```

### Step 7: Test Authentication

**Register a user:**
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "username": "testuser",
    "password": "securepassword123",
    "full_name": "Test User"
  }'
```

**Login:**
```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "securepassword123"
  }'
```

Save the `access_token` from the response.

**Test authenticated endpoint:**
```bash
curl http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

---

## Production Deployment with Docker

### Option A: Docker Compose (Recommended)

#### 1. Configure Environment

```bash
cp .env.example .env
nano .env  # Edit with production values
```

**Important production settings:**
```bash
# CRITICAL: Generate secure secrets
JWT_SECRET_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Database (use internal Docker network)
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4  # Adjust based on CPU cores

# CORS (allow your frontend domain)
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
```

#### 2. Build & Start

```bash
docker-compose up -d --build
```

#### 3. Check Status

```bash
docker-compose ps
docker-compose logs -f
```

#### 4. Access API

- API: http://your-server-ip:8000
- Docs: http://your-server-ip:8000/docs
- Health: http://your-server-ip:8000/health

### Option B: Kubernetes (Advanced)

See `k8s/README.md` for Kubernetes deployment manifests.

---

## Manual Deployment

### Step 1: Install PostgreSQL with pgvector

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y postgresql-15 postgresql-15-pgvector
```

**Or use Docker:**
```bash
docker run -d --name caria-postgres \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_USER=caria_user \
  -e POSTGRES_DB=caria \
  -p 5432:5432 \
  -v caria_postgres_data:/var/lib/postgresql/data \
  ankane/pgvector:latest
```

### Step 2: Initialize Database

```bash
psql -h localhost -U caria_user -d caria -f caria_data/migrations/init.sql
```

### Step 3: Install Python Dependencies

```bash
cd notebooks
poetry install --no-dev
```

### Step 4: Configure Systemd Service

Create `/etc/systemd/system/caria-api.service`:

```ini
[Unit]
Description=Caria API Service
After=network.target postgresql.service

[Service]
Type=simple
User=caria
WorkingDirectory=/opt/caria/notebooks/services
Environment="PATH=/opt/caria/.venv/bin"
EnvironmentFile=/opt/caria/.env
ExecStart=/opt/caria/.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable caria-api
sudo systemctl start caria-api
sudo systemctl status caria-api
```

### Step 5: Setup Nginx Reverse Proxy

Create `/etc/nginx/sites-available/caria`:

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    client_max_body_size 100M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

Enable and get SSL:
```bash
sudo ln -s /etc/nginx/sites-available/caria /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Get SSL certificate
sudo certbot --nginx -d api.yourdomain.com
```

---

## Security Hardening

### 1. Change Default Credentials

**In database:**
```sql
-- Change demo user password
UPDATE users
SET hashed_password = '$2b$12$NEW_HASH_HERE'
WHERE username = 'demo';

-- Or delete demo user
DELETE FROM users WHERE username = 'demo';
```

**Generate new JWT secret:**
```bash
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> .env
```

### 2. Enable HTTPS Only

**In production, always use HTTPS:**
```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

### 3. Rate Limiting

Already implemented in `api/dependencies.py`:
- Public endpoints: 30 requests/minute
- Authenticated: 100 requests/minute

Customize in code or use Nginx rate limiting:
```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

location /api/ {
    limit_req zone=api burst=20 nodelay;
    # ... rest of config
}
```

### 4. Database Security

**Restrict PostgreSQL access:**
```bash
# Edit /etc/postgresql/15/main/pg_hba.conf
# Allow only specific IPs
host    caria    caria_user    10.0.0.0/8    scram-sha-256
```

**Use connection pooling (PgBouncer):**
```bash
docker run -d --name pgbouncer \
  -e DATABASE_URL=postgresql://caria_user:pass@postgres:5432/caria \
  -p 6432:6432 \
  pgbouncer/pgbouncer
```

### 5. Monitoring & Logging

**Structured logging:**
```python
# In production, configure logging to file/syslog
import logging
logging.basicConfig(
    filename='/var/log/caria/api.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Prometheus metrics (optional):**
```bash
pip install prometheus-fastapi-instrumentator
```

---

## User Management

### Creating Users

**Via API (self-registration):**
```bash
POST /api/auth/register
```

**Via Database (admin):**
```bash
psql -h localhost -U caria_user -d caria
```

```sql
-- Create superuser
INSERT INTO users (email, username, full_name, hashed_password, is_superuser)
VALUES (
    'admin@yourdomain.com',
    'admin',
    'Admin User',
    '$2b$12$HASH_HERE',  -- Generate with: python -c "import bcrypt; print(bcrypt.hashpw(b'password', bcrypt.gensalt()).decode())"
    TRUE
);
```

### Managing Users

**Deactivate user:**
```sql
UPDATE users SET is_active = FALSE WHERE email = 'user@example.com';
```

**Verify email:**
```sql
UPDATE users SET is_verified = TRUE WHERE email = 'user@example.com';
```

**View user activity:**
```sql
SELECT * FROM audit_logs WHERE user_id = 'user-uuid' ORDER BY created_at DESC LIMIT 100;
```

### Usage Analytics

**User API usage:**
```sql
SELECT
    u.username,
    COUNT(*) as total_requests,
    AVG(um.response_time_ms) as avg_response_time,
    SUM(um.tokens_used) as total_tokens
FROM usage_metrics um
JOIN users u ON um.user_id = u.id
WHERE um.created_at > NOW() - INTERVAL '7 days'
GROUP BY u.username
ORDER BY total_requests DESC;
```

---

## Monitoring & Maintenance

### Health Checks

**Automated monitoring:**
```bash
# Cron job to check health
*/5 * * * * curl -f http://localhost:8000/health || systemctl restart caria-api
```

**Docker health check:**
Already configured in `docker-compose.yml`.

### Backup Database

**Automated daily backup:**
```bash
#!/bin/bash
# /opt/caria/backup.sh
BACKUP_DIR=/opt/caria/backups
DATE=$(date +%Y%m%d_%H%M%S)

docker exec caria-postgres pg_dump -U caria_user caria | gzip > $BACKUP_DIR/caria_$DATE.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "caria_*.sql.gz" -mtime +30 -delete
```

Add to cron:
```bash
0 2 * * * /opt/caria/backup.sh
```

### Update HMM Model

**Retrain periodically:**
```bash
# Monthly cron job
0 0 1 * * cd /opt/caria/notebooks/caria_data && /opt/caria/.venv/bin/python train_hmm_simple.py
```

### Log Rotation

```bash
# /etc/logrotate.d/caria
/var/log/caria/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 0640 caria caria
    sharedscripts
    postrotate
        systemctl reload caria-api
    endscript
}
```

---

## Troubleshooting

### Database Connection Failed

**Check PostgreSQL is running:**
```bash
docker-compose ps postgres
# OR
systemctl status postgresql
```

**Test connection:**
```bash
psql -h localhost -U caria_user -d caria -c "SELECT version();"
```

**Check credentials in .env:**
```bash
cat .env | grep POSTGRES
```

### Authentication Not Working

**Check JWT secret is set:**
```bash
echo $JWT_SECRET_KEY
```

**Verify database tables exist:**
```sql
\dt users
SELECT COUNT(*) FROM users;
```

**Check user exists:**
```sql
SELECT * FROM users WHERE username = 'your_username';
```

### API Not Starting

**Check logs:**
```bash
# Docker
docker-compose logs -f api

# Systemd
journalctl -u caria-api -f

# Manual
tail -f /var/log/caria/api.log
```

**Common issues:**
- Port 8000 already in use: `lsof -i :8000`
- Missing dependencies: `poetry install`
- Config file not found: Check `CARIA_SETTINGS_PATH`
- HMM model not found: Run `train_hmm_simple.py`

### RAG Not Working

**Check pgvector extension:**
```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

**Check embeddings table:**
```sql
SELECT COUNT(*) FROM document_chunks;
```

**Ingest wisdom documents (if empty):**
```bash
cd notebooks/caria_data
poetry run python scripts/ingestion/ingest_wisdom.py
```

---

## API Endpoints Reference

### Authentication Endpoints

```
POST /api/auth/register         - Register new user
POST /api/auth/login            - Login
POST /api/auth/refresh          - Refresh access token
POST /api/auth/logout           - Logout (revoke refresh token)
GET  /api/auth/me               - Get current user info
```

### Analysis Endpoints (Require Auth)

```
GET  /api/regime/current                - Current macro regime
POST /api/regime/predict                - Predict regime for date
POST /api/factors/screen                - Screen stocks by factors
POST /api/valuation/analyze             - Analyze company valuation
POST /api/analysis/challenge            - Challenge investment thesis (RAG)
```

### Admin Endpoints (Superuser Only)

```
GET  /admin/users                       - List all users
POST /admin/users/{id}/deactivate       - Deactivate user
POST /admin/users/{id}/verify           - Verify user email
GET  /admin/analytics/usage             - System usage statistics
```

---

## Next Steps

1. **Connect your frontend** (Google Studio, React, etc.) to the API
2. **Configure email notifications** for password reset
3. **Setup monitoring** (Prometheus, Grafana, etc.)
4. **Add custom endpoints** for your specific use cases
5. **Scale horizontally** with load balancer + multiple API instances

---

## Support

- **Documentation**: See `IMPLEMENTACION_COMPLETA.md` for technical details
- **API Docs**: http://your-api-url/docs (FastAPI auto-generated)
- **Issues**: Check logs and database for debugging

**System is production-ready!** All core functionality implemented and tested.
