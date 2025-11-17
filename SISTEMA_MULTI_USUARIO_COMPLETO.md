# SISTEMA MULTI-USUARIO COMPLETO - CARIA V2.0

## IMPLEMENTACION COMPLETADA

Tu sistema Caria ahora es una plataforma empresarial multi-usuario lista para produccion.

---

## LO QUE SE IMPLEMENTO HOY

### 1. POSTGRESQL + PGVECTOR

**Archivos:**
- `docker-compose.yml` - Orquestacion completa
- `caria_data/migrations/init.sql` - Schema completo de base de datos
- `Dockerfile` - Container de la API

**Base de datos incluye:**
- Tabla `users` - Usuarios con autenticacion
- Tabla `refresh_tokens` - Tokens de sesion
- Tabla `api_keys` - Keys para acceso programatico
- Tabla `document_chunks` - Embeddings para RAG con pgvector
- Tabla `user_portfolios` - Watchlists por usuario
- Tabla `user_analyses` - Historial de analisis
- Tabla `audit_logs` - Auditoria completa
- Tabla `usage_metrics` - Tracking de uso

**Usuario demo incluido:**
- Email: `demo@caria.com`
- Password: `demo123`

### 2. SISTEMA DE AUTENTICACION

**Archivos creados:**
- `caria_data/src/caria/models/auth.py` - Modelos Pydantic para auth
- `caria_data/src/caria/services/auth_service.py` - Servicio JWT completo
- `services/api/dependencies.py` - FastAPI dependencies
- `services/api/routes/auth.py` - Endpoints de auth

**Features implementadas:**
- Registro de usuarios con validacion
- Login con username o email
- JWT access tokens (1 hora de vida)
- Refresh tokens (30 dias de vida)
- Logout con revocacion de tokens
- Password hashing con bcrypt
- Rate limiting (30 req/min publico, 100 auth)
- Audit logging de todas las acciones

**Endpoints:**
```
POST /api/auth/register         - Registrar usuario
POST /api/auth/login            - Login
POST /api/auth/refresh          - Refrescar token
POST /api/auth/logout           - Logout
GET  /api/auth/me               - Info del usuario actual
```

### 3. MULTI-TENANCY

**Cada usuario tiene:**
- Sus propios portfolios/watchlists
- Historial de analisis privado
- Preferencias individuales
- Metricas de uso separadas

**Aislamiento de datos:**
- Queries con `WHERE user_id = current_user.id`
- API keys por usuario
- Documentos RAG pueden ser publicos o por usuario

### 4. SEGURIDAD

**Implementado:**
- Password hashing con bcrypt + salt
- JWT con firma HS256
- HTTPS ready (configurar en produccion)
- Rate limiting por IP/usuario
- SQL injection protection (parametrized queries)
- CORS configurable
- Token expiration y refresh mechanism
- Audit logging completo

### 5. DOCUMENTACION

**Archivos:**
- `DEPLOYMENT_GUIDE.md` - Guia completa paso a paso
- `README_MULTI_USER.md` - README principal
- `.env.example` - Template de variables de entorno
- `tests/test_auth_flow.py` - Tests end-to-end

**API docs auto-generadas:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## COMO USAR (PASO A PASO)

### PASO 1: Configurar Variables de Entorno

```bash
cd notebooks
cp .env.example .env
nano .env
```

**Configuracion minima:**
```bash
# Generar secret key seguro
JWT_SECRET_KEY=$(openssl rand -hex 32)

# Password de base de datos
POSTGRES_PASSWORD=TuPasswordSeguro123

# LLM (elegir uno)
GEMINI_API_KEY=tu_key_aqui  # O instalar Ollama
```

### PASO 2: Levantar el Sistema

```bash
# Iniciar PostgreSQL + API
docker-compose up -d

# Ver logs
docker-compose logs -f
```

Esperar mensaje: "Caria database initialized successfully!"

### PASO 3: Verificar Estado

```bash
curl http://localhost:8000/health
```

Deberia retornar:
```json
{
  "status": "ok",
  "database": "available",
  "auth": "available",
  "regime": "available",
  "factors": "available",
  "valuation": "available",
  "rag": "available"
}
```

### PASO 4: Registrar Primer Usuario

**Via curl:**
```bash
curl -X POST http://localhost:8000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "tu@email.com",
    "username": "tunombre",
    "password": "TuPassword123!",
    "full_name": "Tu Nombre Completo"
  }'
```

**Guardar el `access_token` de la respuesta!**

### PASO 5: Probar Endpoints

**Regimen actual (publico):**
```bash
curl http://localhost:8000/api/regime/current
```

**Screening con autenticacion:**
```bash
curl -X POST http://localhost:8000/api/factors/screen \
  -H "Authorization: Bearer TU_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "top_n": 10,
    "regime": "expansion"
  }'
```

**Valuacion:**
```bash
curl -X POST http://localhost:8000/api/valuation/analyze \
  -H "Authorization: Bearer TU_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "method": "auto"
  }'
```

### PASO 6: Conectar con Google Studio

Tu UI ya esta conectada a los endpoints. Ahora solo necesitas:

1. **Agregar autenticacion** - Incluir token en headers
2. **Endpoints disponibles** - Mismos de antes + `/api/auth/*`
3. **Flujo:**
   - Usuario hace login en tu UI
   - Guardas el token en localStorage/cookies
   - Incluyes token en todos los requests: `Authorization: Bearer TOKEN`

**Ejemplo de fetch en JavaScript:**
```javascript
// Login
const loginResponse = await fetch('http://your-api:8000/api/auth/login', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    username: 'user',
    password: 'pass'
  })
});
const {token, user} = await loginResponse.json();
localStorage.setItem('token', token.access_token);

// Usar endpoints protegidos
const analysisResponse = await fetch('http://your-api:8000/api/valuation/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${localStorage.getItem('token')}`
  },
  body: JSON.stringify({ticker: 'AAPL', method: 'auto'})
});
```

---

## ARQUITECTURA COMPLETA

```
┌─────────────────────────────────────────┐
│        FRONTEND (Google Studio)         │
│  - Dashboard con visualizaciones        │
│  - Login/Register UI                    │
└───────────────┬─────────────────────────┘
                │ HTTPS/REST + JWT
                ▼
┌─────────────────────────────────────────┐
│         FASTAPI SERVER (Port 8000)      │
│  ┌─────────────────────────────────┐   │
│  │  Auth Router                    │   │
│  │  - /api/auth/register           │   │
│  │  - /api/auth/login              │   │
│  │  - /api/auth/refresh            │   │
│  │  - /api/auth/me                 │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Analysis Routers (Protected)   │   │
│  │  - /api/regime/*                │   │
│  │  - /api/factors/*               │   │
│  │  - /api/valuation/*             │   │
│  │  - /api/analysis/*              │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │  Middleware                     │   │
│  │  - JWT verification             │   │
│  │  - Rate limiting                │   │
│  │  - Audit logging                │   │
│  └─────────────────────────────────┘   │
└───────────────┬─────────────────────────┘
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐
│PostgreSQL│ │   HMM   │ │   LLM   │
│+pgvector │ │  Model  │ │ Service │
│         │ │         │ │         │
│ Tables: │ │ Regime  │ │ Llama/  │
│ -users  │ │Detection│ │ Gemini  │
│ -tokens │ │         │ │         │
│ -docs   │ │         │ │         │
│ -audit  │ │         │ │         │
└─────────┘ └─────────┘ └─────────┘
```

---

## GESTION DE USUARIOS

### Crear Usuario Admin

```sql
psql -h localhost -U caria_user -d caria
```

```sql
-- Generar password hash
-- En Python: import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())

INSERT INTO users (email, username, full_name, hashed_password, is_superuser, is_verified)
VALUES (
    'admin@yourcompany.com',
    'admin',
    'Admin User',
    '$2b$12$YOUR_BCRYPT_HASH_HERE',
    TRUE,  -- Superuser
    TRUE   -- Verified
);
```

### Ver Usuarios

```sql
SELECT id, username, email, is_active, is_verified, created_at, last_login
FROM users
ORDER BY created_at DESC;
```

### Desactivar Usuario

```sql
UPDATE users SET is_active = FALSE WHERE email = 'user@example.com';
```

### Ver Actividad de Usuario

```sql
SELECT
    action,
    resource_type,
    details,
    ip_address,
    created_at
FROM audit_logs
WHERE user_id = 'user-uuid-here'
ORDER BY created_at DESC
LIMIT 50;
```

### Metricas de Uso

```sql
SELECT
    u.username,
    COUNT(*) as total_requests,
    AVG(um.response_time_ms) as avg_response_ms,
    SUM(um.tokens_used) as total_llm_tokens
FROM usage_metrics um
JOIN users u ON um.user_id = u.id
WHERE um.created_at > NOW() - INTERVAL '7 days'
GROUP BY u.username
ORDER BY total_requests DESC;
```

---

## PRUEBAS (TESTING)

### Ejecutar Tests

```bash
# Levantar API primero
docker-compose up -d

# Ejecutar tests
poetry run pytest tests/test_auth_flow.py -v
```

### Tests Incluidos

- Registro de usuario
- Login con username/email
- Validacion de password
- JWT tokens (access + refresh)
- Logout y revocacion
- Endpoints protegidos
- Validacion de inputs
- Rate limiting

---

## MONITOREO

### Healthcheck Automatico

```bash
# Cron job cada 5 minutos
*/5 * * * * curl -f http://localhost:8000/health || systemctl restart caria-api
```

### Logs

```bash
# Docker
docker-compose logs -f api

# Ver logs de auth
docker-compose logs api | grep "auth"

# Ver errores
docker-compose logs api | grep "ERROR"
```

### Metricas Clave

**Usuarios activos:**
```sql
SELECT COUNT(*) FROM users WHERE is_active = TRUE;
```

**Logins hoy:**
```sql
SELECT COUNT(*) FROM audit_logs
WHERE action = 'user.login.success'
AND created_at > CURRENT_DATE;
```

**Requests por endpoint:**
```sql
SELECT endpoint, COUNT(*) as count
FROM usage_metrics
WHERE created_at > NOW() - INTERVAL '1 day'
GROUP BY endpoint
ORDER BY count DESC;
```

---

## BACKUP Y RESTAURACION

### Backup Database

```bash
# Backup completo
docker exec caria-postgres pg_dump -U caria_user caria | gzip > backup_$(date +%Y%m%d).sql.gz

# Backup solo datos de usuarios
docker exec caria-postgres pg_dump -U caria_user -t users -t audit_logs caria > users_backup.sql
```

### Restaurar

```bash
# Restaurar completo
gunzip < backup_20250101.sql.gz | docker exec -i caria-postgres psql -U caria_user caria

# Restaurar tabla especifica
docker exec -i caria-postgres psql -U caria_user caria < users_backup.sql
```

---

## ESCALABILIDAD

### Multiple Workers

```bash
# En docker-compose.yml o comando directo
uvicorn api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Load Balancer (Nginx)

```nginx
upstream caria_backend {
    server caria-api-1:8000;
    server caria-api-2:8000;
    server caria-api-3:8000;
}

server {
    listen 443 ssl;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://caria_backend;
        # ... headers
    }
}
```

### Database Connection Pooling

Ya implementado en dependencies.py. Para produccion, considerar PgBouncer.

---

## SEGURIDAD EN PRODUCCION

### CRITICO - Cambiar antes de deployment:

1. **JWT Secret Key**
   ```bash
   # Generar uno nuevo SIEMPRE
   JWT_SECRET_KEY=$(openssl rand -hex 32)
   ```

2. **Database Password**
   ```bash
   POSTGRES_PASSWORD=$(openssl rand -base64 32)
   ```

3. **HTTPS Obligatorio**
   - Configurar SSL/TLS en Nginx
   - Usar Let's Encrypt (certbot)

4. **Firewall**
   ```bash
   # Solo permitir puerto 443 (HTTPS) y 22 (SSH)
   ufw allow 443/tcp
   ufw allow 22/tcp
   ufw enable
   ```

5. **Rate Limiting**
   - Ya implementado en codigo
   - Reforzar en Nginx si necesitas mas control

---

## RESUMEN FINAL

### LO QUE TIENES AHORA:

- Sistema multi-usuario completo
- Autenticacion JWT segura
- Base de datos PostgreSQL + pgvector
- Multi-tenancy (datos aislados por usuario)
- API REST documentada
- Rate limiting y seguridad
- Audit logging
- Docker deployment
- Tests end-to-end
- Documentacion completa

### LO QUE FALTA (OPCIONAL):

- Email verification (password reset)
- Webhooks para notificaciones
- Panel de admin en UI
- Metricas avanzadas (Prometheus/Grafana)
- Kubernetes deployment
- CI/CD pipeline

### PROXIMOS PASOS SUGERIDOS:

1. **Probar localmente** - `docker-compose up -d`
2. **Registrar usuario admin** - via API o SQL
3. **Conectar tu UI** - Agregar login flow
4. **Deploy a produccion** - Seguir DEPLOYMENT_GUIDE.md
5. **Monitorear** - Configurar healthchecks y backups

---

## TODO LISTO PARA PRODUCCION

Tu sistema esta completamente funcional y listo para usuarios reales.

**Comandos rapidos:**
```bash
# Iniciar todo
docker-compose up -d

# Ver estado
curl http://localhost:8000/health

# Ver logs
docker-compose logs -f

# Detener todo
docker-compose down

# Backup
docker exec caria-postgres pg_dump -U caria_user caria > backup.sql
```

**Documentacion:**
- `DEPLOYMENT_GUIDE.md` - Guia completa de deployment
- `README_MULTI_USER.md` - README principal
- http://localhost:8000/docs - API docs interactiva

**LISTO PARA USAR!**
