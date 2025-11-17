# Ejecutar Migraciones de Base de Datos

Este script ejecuta las migraciones de base de datos en orden, asegurándose de que solo se apliquen una vez.

## Configuración

Antes de ejecutar las migraciones, configura las variables de entorno necesarias:

### Opción 1: Variables de entorno del sistema

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=caria_user
export POSTGRES_PASSWORD=tu_contraseña
export POSTGRES_DB=caria
```

### Opción 2: Archivo .env (recomendado)

Crea un archivo `.env` en el directorio `caria_data/`:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=caria_user
POSTGRES_PASSWORD=tu_contraseña
POSTGRES_DB=caria
```

El script intentará cargar automáticamente este archivo si `python-dotenv` está instalado.

## Ejecución

```bash
cd caria_data
python scripts/migrations/run_migrations.py
```

## Verificación

El script mostrará qué migraciones se aplicaron y cuáles ya estaban aplicadas:

```
INFO:caria.migrations:Aplicando migración: 001_add_auth_tables.sql
INFO:caria.migrations:Migración 001_add_auth_tables.sql aplicada exitosamente
INFO:caria.migrations:Aplicadas 1 migraciones nuevas
```

## Troubleshooting

### Error: "no password supplied"

Asegúrate de que `POSTGRES_PASSWORD` esté configurado. Puedes verificar con:

```bash
echo $POSTGRES_PASSWORD  # Linux/Mac
echo %POSTGRES_PASSWORD%  # Windows CMD
$env:POSTGRES_PASSWORD   # Windows PowerShell
```

### Error: "connection refused"

- Verifica que PostgreSQL esté corriendo: `pg_isready`
- Verifica que el puerto sea correcto: `netstat -an | grep 5432`
- Verifica que el usuario tenga permisos en la base de datos

### Error: "database does not exist"

Crea la base de datos primero:

```bash
psql -U postgres -c "CREATE DATABASE caria;"
```

O ejecuta `init_db.sql` primero:

```bash
psql -U postgres -f infrastructure/init_db.sql
```

