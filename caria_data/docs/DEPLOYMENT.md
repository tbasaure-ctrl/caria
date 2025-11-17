# Guía de Deployment - Caria

Esta guía describe cómo desplegar Caria en producción, incluyendo backend (FastAPI) y frontend (React).

## Requisitos Previos

- Docker y Docker Compose instalados
- PostgreSQL con pgvector (o usar Docker)
- Variables de entorno configuradas

## Configuración de Variables de Entorno

### Backend (.env en `services/`)

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=caria_user
POSTGRES_PASSWORD=your-secure-password
POSTGRES_DB=caria

# JWT
JWT_SECRET_KEY=your-secret-key-min-32-chars

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,https://yourdomain.com

# API Configuration
CARIA_SETTINGS_PATH=../caria_data/configs/base.yaml
CARIA_MODEL_CHECKPOINT=

# Optional
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Frontend (.env.local en `caria_data/caria-app/`)

```bash
VITE_API_URL=http://localhost:8000
VITE_GEMINI_API_KEY=your-key  # Optional
```

## Deployment con Docker Compose

### 1. Preparar Archivos

Asegúrate de que todos los archivos necesarios estén en su lugar:
- `services/Dockerfile`
- `services/docker-compose.yml`
- `caria_data/caria-app/Dockerfile`
- `caria_data/caria-app/nginx.conf`

### 2. Configurar Variables de Entorno

Crea un archivo `.env` en `services/` con todas las variables necesarias (ver arriba).

### 3. Ejecutar Migraciones

Antes de levantar los servicios, ejecuta las migraciones de base de datos:

```bash
cd caria_data
python scripts/migrations/run_migrations.py
```

O si usas Docker:

```bash
docker-compose exec api python -m caria.scripts.migrations.run_migrations
```

### 4. Levantar Servicios

```bash
cd services
docker-compose up -d
```

Esto levantará:
- `postgres`: Base de datos PostgreSQL con pgvector
- `api`: FastAPI backend (puerto 8000)
- `frontend`: React frontend con nginx (puerto 3000)

### 5. Verificar Deployment

```bash
# Healthcheck del backend
curl http://localhost:8000/health

# Healthcheck del frontend
curl http://localhost:3000
```

## Deployment Manual (Sin Docker)

### Backend

1. **Instalar dependencias:**
   ```bash
   cd services
   pip install -r requirements.txt
   pip install -r ../caria_data/requirements.txt
   ```

2. **Configurar variables de entorno:**
   ```bash
   export POSTGRES_PASSWORD=your-password
   export JWT_SECRET_KEY=your-secret-key
   # ... etc
   ```

3. **Ejecutar migraciones:**
   ```bash
   cd ../caria_data
   python scripts/migrations/run_migrations.py
   ```

4. **Levantar API:**
   ```bash
   cd ../services
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```

### Frontend

1. **Instalar dependencias:**
   ```bash
   cd caria_data/caria-app
   npm install
   ```

2. **Configurar .env.local:**
   ```bash
   cp .env.example .env.local
   # Editar .env.local con VITE_API_URL
   ```

3. **Build:**
   ```bash
   npm run build
   ```

4. **Servir con nginx o servidor estático:**
   ```bash
   # Con nginx (ver nginx.conf)
   # O con serve:
   npx serve -s dist -p 3000
   ```

## Healthchecks

### Backend

- **Liveness**: `GET /health/live` - Verifica que la app responde
- **Readiness**: `GET /health/ready` - Verifica dependencias (DB, modelos)
- **General**: `GET /health` - Estado completo de servicios

### Frontend

- Nginx responde en puerto 80 (o 3000 si mapeado)

## Migraciones de Base de Datos

Las migraciones se ejecutan automáticamente al inicializar la base de datos (si usas `init_db.sql`), o manualmente:

```bash
cd caria_data
python scripts/migrations/run_migrations.py
```

Las migraciones se trackean en la tabla `schema_migrations` y solo se aplican una vez.

## Verificación Post-Deployment

1. **Backend:**
   ```bash
   curl http://localhost:8000/health
   # Debe retornar estado de todos los servicios
   ```

2. **Frontend:**
   - Abrir `http://localhost:3000` en navegador
   - Debe mostrar landing page
   - Probar login/registro

3. **Endpoints protegidos:**
   ```bash
   # Sin auth (debe fallar con 401)
   curl -X POST http://localhost:8000/api/factors/screen
   
   # Con auth (debe funcionar)
   curl -X POST http://localhost:8000/api/factors/screen \
     -H "Authorization: Bearer YOUR_TOKEN"
   ```

## Troubleshooting

### Backend no conecta a PostgreSQL

- Verificar que PostgreSQL está corriendo
- Verificar variables de entorno `POSTGRES_*`
- Verificar que pgvector está instalado: `CREATE EXTENSION vector;`

### Frontend no conecta a API

- Verificar `VITE_API_URL` en `.env.local`
- Verificar CORS en backend (`CORS_ORIGINS`)
- Verificar que backend está corriendo

### Migraciones fallan

- Verificar que PostgreSQL tiene permisos
- Verificar que usuario tiene acceso a base de datos
- Ver logs: `docker-compose logs postgres`

### Modelos no cargan

- Verificar que `models/regime_hmm_model.pkl` existe
- Verificar path en `CARIA_SETTINGS_PATH`
- Verificar permisos de lectura en volúmenes Docker

## Producción

### Recomendaciones de Seguridad

1. **Secrets Management:**
   - Usar secrets manager (AWS Secrets Manager, HashiCorp Vault)
   - NO commitear `.env` files
   - Rotar `JWT_SECRET_KEY` regularmente

2. **HTTPS:**
   - Configurar reverse proxy (nginx/traefik) con SSL
   - Usar Let's Encrypt para certificados

3. **Rate Limiting:**
   - Configurar rate limits apropiados por endpoint
   - Considerar usar Redis para rate limiting distribuido

4. **Monitoring:**
   - Configurar logging estructurado
   - Exponer métricas Prometheus
   - Configurar alertas

5. **Backups:**
   - Backup regular de PostgreSQL
   - Backup de modelos entrenados
   - Backup de embeddings en pgvector

## Escalabilidad

- **Backend**: Usar múltiples instancias con load balancer
- **Frontend**: Servir desde CDN (Cloudflare, AWS CloudFront)
- **Database**: Considerar read replicas para PostgreSQL
- **Cache**: Usar Redis para caching de respuestas frecuentes

