# Troubleshooting - Migraciones de Base de Datos

## Error: "permiso denegado al esquema public"

Este error indica que el usuario de PostgreSQL no tiene permisos para crear tablas.

### Solución 1: Ejecutar init_db.sql primero (Recomendado)

El archivo `init_db.sql` crea todas las tablas necesarias, incluyendo `schema_migrations`, y otorga los permisos correctos.

```bash
# Como superusuario de PostgreSQL (postgres)
psql -U postgres -f infrastructure/init_db.sql
```

### Solución 2: Conceder permisos manualmente

Conecta como superusuario y concede permisos:

```sql
-- Conectar como superusuario
psql -U postgres -d caria

-- Conceder permisos al usuario
GRANT ALL ON SCHEMA public TO caria_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO caria_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO caria_user;

-- Crear la tabla schema_migrations si no existe
CREATE TABLE IF NOT EXISTS schema_migrations (
    id SERIAL PRIMARY KEY,
    migration_name VARCHAR(255) UNIQUE NOT NULL,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Solución 3: Ejecutar migraciones como superusuario

Temporalmente, puedes ejecutar las migraciones como superusuario:

```bash
# Configurar variables de entorno para superusuario
$env:POSTGRES_USER="postgres"
$env:POSTGRES_PASSWORD="password_de_postgres"
python scripts/migrations/run_migrations.py
```

## Error: "no existe la relación schema_migrations"

La tabla `schema_migrations` no existe. Esto se resuelve ejecutando `init_db.sql` primero (ver Solución 1 arriba).

## Error: "connection refused" o "no password supplied"

Verifica que:
1. PostgreSQL esté corriendo: `pg_isready` o `Get-Service postgresql*`
2. Las variables de entorno estén configuradas (ver `README.md`)
3. El archivo `.env` exista y tenga los valores correctos

## Verificar estado de la base de datos

```sql
-- Conectar a la base de datos
psql -U caria_user -d caria

-- Verificar que schema_migrations existe
\dt schema_migrations

-- Ver migraciones aplicadas
SELECT * FROM schema_migrations ORDER BY applied_at;
```

## Flujo recomendado de inicialización

1. **Crear base de datos y usuario** (como superusuario):
   ```sql
   CREATE DATABASE caria;
   CREATE USER caria_user WITH PASSWORD 'tu_contraseña';
   ```

2. **Ejecutar init_db.sql** (como superusuario):
   ```bash
   psql -U postgres -f infrastructure/init_db.sql
   ```

3. **Ejecutar migraciones** (como usuario normal):
   ```bash
   # Configurar .env con credenciales de caria_user
   python scripts/migrations/run_migrations.py
   ```

