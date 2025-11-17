# Quick Start - Caria Database Setup

## Opción 1: Con pgvector instalado (Recomendado para producción)

Si tienes pgvector instalado, ejecuta:

```sql
-- En pgAdmin o psql
\i infrastructure/init_db.sql
```

## Opción 2: Sin pgvector (Para desarrollo rápido)

Si no tienes pgvector instalado, ejecuta:

```sql
-- En pgAdmin o psql
\i infrastructure/init_db_without_vector.sql
```

**Nota:** Las funcionalidades de RAG (Sistema II) requerirán pgvector más adelante.

## Opción 3: Usar Docker (Más fácil)

Si usas Docker, pgvector viene pre-instalado:

```bash
cd services
docker-compose up -d postgres
```

Luego ejecuta `init_db.sql` normalmente.

## Después de init_db.sql

Una vez que `init_db.sql` se ejecute exitosamente, ejecuta las migraciones:

```bash
python scripts/migrations/run_migrations.py
```

## Instalar pgvector en Windows

Ver `docs/INSTALL_PGVECTOR.md` para instrucciones detalladas.

### Resumen rápido:

1. Descarga pgvector para PostgreSQL 17 desde: https://github.com/pgvector/pgvector/releases
2. Copia archivos a PostgreSQL:
   - `vector.dll` → `C:\Program Files\PostgreSQL\17\lib\`
   - `vector.control` → `C:\Program Files\PostgreSQL\17\share\extension\`
   - `vector--*.sql` → `C:\Program Files\PostgreSQL\17\share\extension\`
3. Reinicia PostgreSQL: `Restart-Service postgresql-x64-17`
4. Verifica: `CREATE EXTENSION vector;`

## Verificación

Después de ejecutar `init_db.sql`, verifica que las tablas se crearon:

```sql
SELECT table_name 
FROM information_schema.tables 
WHERE table_schema = 'public' 
ORDER BY table_name;
```

Deberías ver: `users`, `schema_migrations`, `wisdom_chunks`, `prices`, `fundamentals`, etc.
