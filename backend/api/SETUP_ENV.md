# Configuración de Variables de Entorno

## Configuración Rápida

### Opción 1: Archivo .env (Recomendado)

Crea un archivo `.env` en el directorio `services/api/` con:

```bash
# Base de datos PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=caria_user
POSTGRES_PASSWORD=tu_password_aqui
POSTGRES_DB=caria

# FMP API Key (para precios en tiempo real)
FMP_API_KEY=79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq

# JWT Secret Key
JWT_SECRET_KEY=tu_secret_key_seguro_aqui

# CORS Origins
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

Luego carga las variables antes de iniciar la API:

```bash
# Linux/Mac
export $(cat .env | xargs)

# Windows PowerShell
Get-Content .env | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') {
        [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
    }
}
```

### Opción 2: Variables de Entorno del Sistema

Configura las variables directamente en tu sistema:

**Linux/Mac:**
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=caria_user
export POSTGRES_PASSWORD=tu_password
export POSTGRES_DB=caria
export FMP_API_KEY=79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq
export JWT_SECRET_KEY=tu_secret_key_seguro
```

**Windows (PowerShell):**
```powershell
$env:POSTGRES_HOST="localhost"
$env:POSTGRES_PORT="5432"
$env:POSTGRES_USER="caria_user"
$env:POSTGRES_PASSWORD="tu_password"
$env:POSTGRES_DB="caria"
$env:FMP_API_KEY="79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq"
$env:JWT_SECRET_KEY="tu_secret_key_seguro"
```

**Windows (CMD):**
```cmd
set POSTGRES_HOST=localhost
set POSTGRES_PORT=5432
set POSTGRES_USER=caria_user
set POSTGRES_PASSWORD=tu_password
set POSTGRES_DB=caria
set FMP_API_KEY=79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq
set JWT_SECRET_KEY=tu_secret_key_seguro
```

### Opción 3: Docker Compose

Si usas Docker, agrega las variables en `docker-compose.yml`:

```yaml
services:
  api:
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_USER=caria_user
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=caria
      - FMP_API_KEY=79fY9wvC9qtCJHcn6Yelf4ilE9TkRMoq
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
```

## Verificación

Para verificar que las variables están configuradas:

```bash
# Linux/Mac
echo $FMP_API_KEY

# Windows PowerShell
echo $env:FMP_API_KEY

# Windows CMD
echo %FMP_API_KEY%
```

## Notas Importantes

1. **FMP_API_KEY**: Esta es tu API key específica para precios en tiempo real. No la compartas públicamente.

2. **JWT_SECRET_KEY**: Genera una clave segura para producción:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. **Seguridad**: Nunca commitees el archivo `.env` al repositorio. Ya está en `.gitignore`.

4. **Producción**: En producción, usa un gestor de secretos (AWS Secrets Manager, HashiCorp Vault, etc.)

