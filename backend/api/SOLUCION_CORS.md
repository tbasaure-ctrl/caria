# Solución: Error de CORS

## Problema

El navegador muestra:
```
Access to fetch at 'http://localhost:8000/api/auth/login' from origin 'http://localhost:3000' 
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present 
on the requested resource.
```

## Causa

Según la [documentación de CORS de web.dev](https://web.dev/articles/cross-origin-resource-sharing), cuando haces una solicitud compleja (POST con `Content-Type: application/json`), el navegador primero envía una **solicitud de preflight (OPTIONS)** y el servidor debe responder con los headers CORS apropiados.

El problema puede ser:
1. El servidor no está respondiendo correctamente a las solicitudes OPTIONS
2. Un error 500 está impidiendo que se envíen los headers CORS
3. El middleware CORS no está en el orden correcto

## Solución Implementada

### 1. CORS Middleware Mejorado

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,  # Cache preflight requests for 10 minutes
)
```

### 2. Orden Correcto

El middleware CORS debe estar **ANTES** de `_init_state(app)` y los routers para que funcione correctamente.

### 3. Manejo de Errores

Los errores 500 ahora no bloquean los headers CORS porque el middleware está configurado correctamente.

## Verificación

Para verificar que CORS funciona:

```bash
# Probar solicitud OPTIONS (preflight)
curl -X OPTIONS http://localhost:8000/api/auth/login \
  -H "Origin: http://localhost:3000" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: content-type" \
  -v
```

Deberías ver:
```
< HTTP/1.1 200 OK
< Access-Control-Allow-Origin: http://localhost:3000
< Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS, PATCH, HEAD
< Access-Control-Allow-Credentials: true
< Access-Control-Max-Age: 600
```

## Pasos para Resolver

1. **Reinicia la API** para aplicar los cambios:
   ```bash
   # Detén con CTRL+C y reinicia
   python start_api.py
   ```

2. **Limpia la caché del navegador** o usa modo incógnito

3. **Recarga el frontend** (F5)

4. **Intenta hacer login de nuevo**

## Si Aún No Funciona

1. Verifica que la API esté corriendo:
   ```bash
   curl http://localhost:8000/health
   ```

2. Verifica los logs de la API para ver si hay errores

3. Abre la pestaña Network en DevTools y verifica:
   - La solicitud OPTIONS (preflight) debe responder con 200
   - La solicitud POST debe tener los headers CORS en la respuesta

## Referencias

- [CORS en web.dev](https://web.dev/articles/cross-origin-resource-sharing)
- [FastAPI CORS Middleware](https://fastapi.tiangolo.com/tutorial/cors/)

