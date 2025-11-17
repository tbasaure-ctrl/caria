# Caria - Multi-User Investment Analysis Platform

**Version 2.0** - Enterprise-ready multi-tenant system with authentication, RAG, and advanced valuation

---

## What is Caria?

Caria is an AI-powered investment analysis platform that combines:

1. **Macro Regime Detection** - HMM-based market regime classification
2. **RAG-Powered Analysis** - Challenge investment theses with historical wisdom
3. **Factor Investing** - Screen stocks based on Value, Quality, Growth, Momentum
4. **Multi-Method Valuation** - DCF, Revenue Multiples, Scorecard for pre-revenue

**New in V2.0:**
- Multi-user authentication with JWT
- Per-user data isolation (multi-tenancy)
- API keys for programmatic access
- Usage tracking and audit logs
- Production-ready with Docker deployment

---

## Quick Start (5 Minutes)

### 1. Start the System

```bash
cd notebooks

# Copy environment template
cp .env.example .env

# Generate secure JWT secret
echo "JWT_SECRET_KEY=$(openssl rand -hex 32)" >> .env

# Start database + API
docker-compose up -d

# Wait for initialization (~30 seconds)
docker-compose logs -f postgres
```

### 2. Verify Health

```bash
curl http://localhost:8000/health
```

Expected:
```json
{
  "status": "ok",
  "database": "available",
  "auth": "available",
  "regime": "available",
  "factors": "available",
  "valuation": "available"
}
```

### 3. Register a User

```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "you@example.com",
    "username": "yourname",
    "password": "securepassword123"
  }'
```

Save the `access_token` from the response!

### 4. Try It Out

**Get current regime:**
```bash
curl http://localhost:8000/api/regime/current
```

**Screen top stocks:**
```bash
curl -X POST http://localhost:8000/api/factors/screen \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"top_n": 10, "regime": "expansion"}'
```

**Analyze company:**
```bash
curl -X POST http://localhost:8000/api/valuation/analyze \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "method": "auto"}'
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Google Studio UI                      │
│            (Your existing dashboard - already built)     │
└───────────────────────────┬─────────────────────────────┘
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Server                       │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │ Auth Router│  │ Analysis   │  │  Regime Router   │  │
│  │ (JWT/Users)│  │   Router   │  │  Factors Router  │  │
│  └────────────┘  └────────────┘  └──────────────────┘  │
└───────────────────────────┬─────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  PostgreSQL  │   │     HMM      │   │     LLM      │
│  + pgvector  │   │    Model     │   │   Service    │
│              │   │  (Regime)    │   │ (Llama/Gemini)│
│  - Users     │   │              │   │              │
│  - Audit     │   │              │   │              │
│  - RAG docs  │   │              │   │              │
└──────────────┘   └──────────────┘   └──────────────┘
```

---

## Key Features

### 1. Authentication & Multi-Tenancy

- **JWT-based authentication** - Secure token-based auth
- **User management** - Register, login, logout, refresh tokens
- **Per-user data isolation** - Each user has their own portfolios, analyses
- **API keys** - For programmatic access
- **Audit logging** - Track all user actions
- **Rate limiting** - 100 req/min for authenticated, 30 for public

### 2. Macro Regime Detection (System I)

- **Hidden Markov Model** with 4 states:
  - Expansion (growth, low risk)
  - Slowdown (cooling, moderate risk)
  - Recession (contraction, high risk)
  - Stress (crisis, extreme risk)
- **12,753 historical predictions** (1990-2024)
- **Real-time regime probabilities**
- **WACC adjustment by regime** for DCF

### 3. RAG-Powered Thesis Challenge (System II)

- **Vector search** with pgvector (768-dim embeddings)
- **Multi-LLM support**:
  - Llama 3.2 (local, free) - Recommended
  - Google Gemini (API)
  - OpenAI GPT (fallback)
- **Bias detection** - Identifies confirmation bias, overconfidence, etc.
- **Historical wisdom** - Learn from past market lessons

### 4. Factor Investing (System III)

Screen stocks by:
- **Value** - P/E, P/B, EV/EBITDA
- **Profitability** - ROIC, ROE, margins
- **Growth** - Revenue growth, earnings growth
- **Momentum** - Price momentum, earnings surprises
- **Solvency** - Debt ratios, interest coverage

**Regime-adaptive weights** - Automatically adjust for market conditions

### 5. Multi-Method Valuation (System IV)

#### DCF (Discounted Cash Flow)
- For established companies with positive FCF
- Regime-adjusted WACC
- Incorporates net debt

#### Revenue Multiples
- For companies with revenue but negative FCF
- Sector-specific multiples (SaaS 8x, Tech 4x, etc.)
- EV/Revenue and P/S ratios

#### Scorecard/Berkus
- For pre-revenue startups
- Stage-based valuations (pre-seed $1-8M, seed $3-20M, etc.)
- Team, market, technology, traction factors

---

## API Documentation

### Authentication

```http
POST /api/auth/register
POST /api/auth/login
POST /api/auth/refresh
POST /api/auth/logout
GET  /api/auth/me
```

### Analysis (Require Authentication)

```http
GET  /api/regime/current
POST /api/regime/predict
POST /api/factors/screen
POST /api/valuation/analyze
POST /api/analysis/challenge
```

**Interactive docs:** http://localhost:8000/docs

---

## Configuration

### Environment Variables

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PASSWORD=your_secure_password

# Security
JWT_SECRET_KEY=your_secret_key_here

# LLM (choose one)
GEMINI_API_KEY=your_key      # OR
OPENAI_API_KEY=your_key      # OR
# Install Ollama (local, free)
```

See `.env.example` for complete list.

### LLM Setup

**Option A: Llama (Local, Free) - RECOMMENDED**
```bash
# 1. Install Ollama: https://ollama.ai/download
# 2. Pull model:
ollama pull llama3.2

# 3. Install Python client:
poetry add ollama
```

**Option B: Gemini (API, Free tier available)**
```bash
# 1. Get API key: https://makersuite.google.com/app/apikey
# 2. Set environment:
export GEMINI_API_KEY="your_key"

# 3. Install client:
poetry add google-generativeai
```

---

## Data Requirements

### Input Data (Already Included)

- **Macro data**: FRED indicators (1990-2024)
- **Stock fundamentals**: 476 tickers from silver/
- **Price data**: 2.8M+ observations
- **Technical indicators**: RSI, MACD, ATR, SMA, EMA

### Model Artifacts

- **HMM model**: `caria_data/models/regime_hmm_model.pkl`
- **Predictions**: `caria_data/data/silver/regime/hmm_regime_predictions.parquet`

To retrain:
```bash
cd caria_data
poetry run python train_hmm_simple.py
```

---

## Deployment

### Development

```bash
docker-compose up -d
```

### Production

See `DEPLOYMENT_GUIDE.md` for:
- Docker deployment
- Kubernetes manifests
- Manual deployment with systemd
- Nginx reverse proxy
- SSL certificates
- Backup procedures
- Monitoring setup

---

## User Workflows

### For Analysts

1. **Register** → Create account
2. **Login** → Get access token
3. **Check regime** → Understand market environment
4. **Screen stocks** → Find opportunities aligned with regime
5. **Analyze valuations** → Assess fair value
6. **Challenge thesis** → Validate with RAG

### For Developers

1. **Get API key** → Programmatic access
2. **Integrate** → Connect your app to Caria API
3. **Build dashboards** → Visualize insights
4. **Automate** → Schedule periodic analyses

### For Admins

1. **Manage users** → via SQL or admin endpoints
2. **Monitor usage** → Track API calls, tokens used
3. **Audit logs** → Review user actions
4. **Backup database** → Daily automated backups

---

## Performance

- **API response time**: < 500ms (avg)
- **Concurrent users**: 100+ (with 4 workers)
- **Database**: Optimized indexes, connection pooling
- **Caching**: In-memory for regime predictions
- **Rate limiting**: Prevents abuse

---

## Security

- **Password hashing**: bcrypt with salt
- **JWT tokens**: HS256 signing
- **Token expiration**: 1 hour access, 30 days refresh
- **HTTPS only**: In production
- **CORS**: Configurable allowed origins
- **SQL injection**: Protected by parameterized queries
- **Rate limiting**: 30-100 req/min depending on auth

---

## Testing

Run tests:
```bash
poetry run pytest tests/
```

Manual testing:
```bash
# See DEPLOYMENT_GUIDE.md for curl examples
```

---

## Roadmap

**Completed:**
- [x] Multi-user authentication
- [x] Multi-tenancy
- [x] 4 analysis systems (HMM, RAG, Factors, Valuation)
- [x] Multi-LLM support
- [x] Docker deployment
- [x] Production-ready API

**Planned:**
- [ ] Email verification (password reset)
- [ ] Webhooks for alerts
- [ ] Real-time websocket updates
- [ ] Mobile app
- [ ] Advanced backtesting
- [ ] Ensemble ML models (XGBoost, LSTM, Transformer)

---

## Troubleshooting

**Database connection failed?**
```bash
docker-compose logs postgres
docker-compose restart postgres
```

**Auth not working?**
```bash
# Check JWT secret is set
echo $JWT_SECRET_KEY

# Verify database initialized
docker exec -it caria-postgres psql -U caria_user -d caria -c "\dt"
```

**API errors?**
```bash
# Check API logs
docker-compose logs -f api

# Restart API
docker-compose restart api
```

See `DEPLOYMENT_GUIDE.md` for detailed troubleshooting.

---

## Documentation

- `DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `IMPLEMENTACION_COMPLETA.md` - Technical implementation details
- `QUICK_START.md` - Simple 5-minute setup
- `RESUMEN_FINAL.md` - Executive summary
- http://localhost:8000/docs - Interactive API docs

---

## Support & Contact

- **GitHub Issues**: [Your repo]
- **Documentation**: See `/docs` folder
- **API Docs**: http://localhost:8000/docs

---

## License

[Your license here]

---

## Credits

Built with:
- FastAPI - Web framework
- PostgreSQL + pgvector - Database
- scikit-learn - Machine learning
- Ollama / Gemini - LLM providers
- bcrypt - Password hashing
- PyJWT - JWT tokens

**Project Status:** Production-ready, actively maintained

---

**Ready to use!** Start with `docker-compose up -d` and you're live in 5 minutes.
