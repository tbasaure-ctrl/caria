# Instalación de pgvector en PostgreSQL

La extensión `pgvector` es necesaria para almacenar y buscar embeddings vectoriales en PostgreSQL. Si ves el error:

```
ERROR: la extensión «vector» no está disponible
```

Necesitas instalar pgvector primero.

## Opción 1: Usar Docker (Recomendado)

Si usas Docker, usa una imagen que ya incluye pgvector:

```yaml
# En docker-compose.yml
postgres:
  image: ankane/pgvector:latest  # Ya incluye pgvector
  # ... resto de configuración
```

## Opción 2: Instalación Manual en Windows

### Paso 1: Descargar pgvector

1. Ve a https://github.com/pgvector/pgvector/releases
2. Descarga la versión compatible con PostgreSQL 17 (Windows)
3. Extrae los archivos

### Paso 2: Instalar en PostgreSQL

1. Copia los archivos a la carpeta de extensiones de PostgreSQL:
   - `vector.dll` → `C:\Program Files\PostgreSQL\17\lib\`
   - `vector.control` → `C:\Program Files\PostgreSQL\17\share\extension\`
   - `vector--*.sql` → `C:\Program Files\PostgreSQL\17\share\extension\`

2. Reinicia el servicio de PostgreSQL:
   ```powershell
   Restart-Service postgresql-x64-17
   ```

### Paso 3: Verificar Instalación

En pgAdmin o psql:

```sql
CREATE EXTENSION vector;
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## Opción 3: Compilar desde Código Fuente

Si prefieres compilar desde el código fuente:

1. Instala Visual Studio Build Tools
2. Instala Rust (requerido para pgvector)
3. Clona el repositorio:
   ```bash
   git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
   cd pgvector
   ```
4. Compila e instala:
   ```bash
   # Ver instrucciones en README.md del repositorio
   ```

## Opción 4: Usar PostgreSQL con pgvector Pre-instalado

Considera usar una distribución de PostgreSQL que incluya pgvector:
- **Postgres.app** (Mac) con extensión pgvector
- **PostgreSQL con Docker** usando `ankane/pgvector`

## Verificación Post-Instalación

Después de instalar, ejecuta:

```sql
-- En psql o pgAdmin
CREATE EXTENSION IF NOT EXISTS vector;
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

Deberías ver:
```
 extname | extversion 
---------+------------
 vector  | 0.5.1
```

## Continuar con init_db.sql

Una vez que pgvector esté instalado, puedes ejecutar `init_db.sql` sin errores.

Si no puedes instalar pgvector ahora, puedes comentar las líneas relacionadas con `vector` en `init_db.sql` y continuar. Las funcionalidades de RAG (Sistema II) requerirán pgvector más adelante.

## Troubleshooting

### Error: "No se pudo abrir el archivo de control"

- Verifica que `vector.control` esté en `share/extension/`
- Verifica permisos de lectura
- Reinicia PostgreSQL

### Error: "No se encontró la biblioteca"

- Verifica que `vector.dll` esté en `lib/`
- Verifica que la versión sea compatible con tu PostgreSQL
- Reinicia PostgreSQL

### Error: "Extensión no disponible"

- Verifica que PostgreSQL esté corriendo
- Verifica que los archivos estén en las rutas correctas
- Consulta los logs de PostgreSQL para más detalles

