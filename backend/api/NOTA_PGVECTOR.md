# Nota sobre pgvector (Opcional)

## Warning que ves

```
No se pudo crear tabla en PostgreSQL (esto es opcional): 
(psycopg2.errors.UndefinedObject) no existe el tipo «vector»
```

## ¿Qué significa?

Este warning indica que la extensión `pgvector` no está instalada en tu PostgreSQL. Esta extensión es necesaria para:
- **RAG (Retrieval Augmented Generation)**: Funcionalidad de búsqueda semántica y chat inteligente
- **Embeddings**: Almacenamiento de vectores para búsqueda por similitud

## ¿Es un problema?

**NO**. Es completamente opcional. La API funciona perfectamente sin ella para:
- ✅ Autenticación de usuarios
- ✅ Registro y login
- ✅ Holdings de usuarios
- ✅ Precios en tiempo real
- ✅ Valuaciones (DCF, Multiples, Monte Carlo)
- ✅ Factores y régimen

## Si quieres habilitar RAG más adelante

### Opción 1: Instalar pgvector en PostgreSQL local

1. Descargar pgvector desde: https://github.com/pgvector/pgvector
2. Compilar e instalar según las instrucciones
3. O usar una versión de PostgreSQL que ya lo incluya

### Opción 2: Usar PostgreSQL con pgvector preinstalado

- **Docker**: `docker run -d -p 5432:5432 pgvector/pgvector:pg15`
- **Cloud**: Usar servicios como Supabase, Neon, o AWS RDS con pgvector

### Opción 3: Deshabilitar RAG completamente

Si no necesitas RAG, puedes ignorar este warning. La funcionalidad RAG simplemente no estará disponible, pero todo lo demás funcionará perfectamente.

## Verificación

Para verificar qué funcionalidades están disponibles:

```bash
curl http://localhost:8000/health
```

Deberías ver:
```json
{
  "status": "ok",
  "database": "available",
  "auth": "available",
  "rag": "disabled",  // ← Esto es normal sin pgvector
  "regime": "available",
  "factors": "available",
  "valuation": "available"
}
```

## Conclusión

**Puedes ignorar este warning**. La API funciona perfectamente sin pgvector para todas las funcionalidades principales.

