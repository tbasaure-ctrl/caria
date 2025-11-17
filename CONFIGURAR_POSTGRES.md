# Guía Rápida: Configurar PostgreSQL

## Paso 1: Verificar que PostgreSQL está corriendo

```powershell
# Verificar servicio de PostgreSQL (Windows)
Get-Service -Name postgresql*

# O verificar si el puerto 5432 está en uso
netstat -an | findstr 5432
```

## Paso 2: Configurar Variables de Entorno

```powershell
# Configurar contraseña de PostgreSQL
$env:POSTGRES_PASSWORD='tu_password_aqui'

# Otras variables (opcionales, tienen defaults)
$env:POSTGRES_HOST='localhost'
$env:POSTGRES_PORT='5432'
$env:POSTGRES_USER='caria_user'
$env:POSTGRES_DB='caria'
```

## Paso 3: Verificar Conexión

```powershell
cd services/api
python check_postgres.py
```

Este script te dirá:
- ✅ Si la conexión funciona
- ✅ Qué tablas existen
- ✅ Si falta la tabla `holdings`

## Paso 4: Ejecutar Migración de Holdings

```powershell
# Si check_postgres.py dice que falta la tabla holdings:
python run_migration.py
```

## Paso 5: Verificar que Todo Funciona

```powershell
# Reiniciar la API
python start_api.py

# En otra terminal, verificar health:
curl http://localhost:8000/health
```

Deberías ver:
```json
{
  "status": "ok",
  "database": "available",  // ✅ Ahora disponible
  "auth": "available",       // ✅ Ahora disponible
  ...
}
```

## Si PostgreSQL no está instalado

### Opción A: Instalar PostgreSQL localmente

1. Descargar desde: https://www.postgresql.org/download/windows/
2. Durante instalación, recordar la contraseña del usuario `postgres`
3. Crear base de datos y usuario:

```sql
-- Conectar como postgres
CREATE DATABASE caria;
CREATE USER caria_user WITH PASSWORD 'tu_password';
GRANT ALL PRIVILEGES ON DATABASE caria TO caria_user;
```

### Opción B: Usar Docker

```powershell
docker run --name caria-postgres `
  -e POSTGRES_USER=caria_user `
  -e POSTGRES_PASSWORD=tu_password `
  -e POSTGRES_DB=caria `
  -p 5432:5432 `
  -d postgres:15
```

Luego configurar variables:
```powershell
$env:POSTGRES_HOST='localhost'
$env:POSTGRES_PORT='5432'
$env:POSTGRES_USER='caria_user'
$env:POSTGRES_PASSWORD='tu_password'
$env:POSTGRES_DB='caria'
```

## Troubleshooting

### Error: "connection to server failed"
- Verifica que PostgreSQL está corriendo
- Verifica que el puerto 5432 no está bloqueado por firewall
- Verifica las credenciales

### Error: "database caria does not exist"
```sql
-- Conectar a PostgreSQL y ejecutar:
CREATE DATABASE caria;
```

### Error: "role caria_user does not exist"
```sql
-- Conectar como postgres y ejecutar:
CREATE USER caria_user WITH PASSWORD 'tu_password';
GRANT ALL PRIVILEGES ON DATABASE caria TO caria_user;
```

### Error: "permission denied"
```sql
-- Conectar como postgres y ejecutar:
GRANT ALL PRIVILEGES ON DATABASE caria TO caria_user;
\c caria
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO caria_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO caria_user;
```

## Verificación Final

Una vez configurado, deberías poder:

1. ✅ Ver `database: "available"` en `/health`
2. ✅ Registrar usuarios en `/api/auth/register`
3. ✅ Hacer login en `/api/auth/login`
4. ✅ Crear holdings en `/api/holdings`
5. ✅ Ver holdings con precios en `/api/holdings/with-prices`

