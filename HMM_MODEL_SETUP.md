# 🔧 Configuración del Modelo HMM - CRÍTICO

## ⚠️ Importancia del Modelo HMM

El modelo HMM (Hidden Markov Model) es **CRÍTICO** para el **Sistema I: Detección de Régimen Macroeconómico**. 

### Endpoints que lo requieren:
- ✅ `/api/regime/current` - Retorna régimen actual (tiene fallback pero valores por defecto)
- ⚠️ `/api/portfolio/regime-test` - Testing de portfolios por régimen (puede fallar sin modelo)
- ⚠️ `/api/tactical/allocation` - Asignación táctica basada en régimen (puede fallar sin modelo)

**Sin el modelo HMM, estas funcionalidades usarán valores por defecto o pueden fallar.**

## 📍 Ubicación del Modelo

### En desarrollo local:
```
caria_data/models/regime_hmm_model.pkl
```

### En Docker/Render:
```
/app/models/regime_hmm_model.pkl
```

El Dockerfile intenta copiar el modelo desde `caria_data/models/` a `/app/models/` durante el build.

## ✅ Opción 1: Verificar si el Modelo Existe Localmente

1. **Verifica si existe:**
```bash
ls -la caria_data/models/regime_hmm_model.pkl
```

2. **Si existe, el Dockerfile lo copiará automáticamente** durante el build de Render.

3. **Si NO existe**, necesitas entrenarlo (ver Opción 2).

## ✅ Opción 2: Entrenar el Modelo HMM

### Requisitos:
- Datos macro procesados: `caria_data/data/silver/macro/fred_data.parquet`
- Python con dependencias instaladas

### Método Simple (Recomendado):

```bash
cd caria_data
python train_hmm_simple.py
```

**Este script:**
1. Carga datos macro desde `data/silver/macro/fred_data.parquet`
2. Filtra período 1990-2024
3. Entrena modelo HMM con 4 estados
4. Guarda en `models/regime_hmm_model.pkl`
5. Genera predicciones históricas

### Método Completo (Con Prefect):

```bash
cd caria_data
python scripts/orchestration/run_regime_hmm.py
```

### Verificar que se Entrenó Correctamente:

```bash
python caria_data/validate_hmm.py
```

**Deberías ver:**
```
=== MODELO HMM CARGADO EXITOSAMENTE ===
=== PARÁMETROS HMM ===
...
```

## ✅ Opción 3: Copiar Modelo Manualmente al Dockerfile

Si el modelo existe localmente pero no se copia correctamente:

### Verificar Dockerfile:

El Dockerfile ya tiene esta lógica (líneas 38-43):
```dockerfile
RUN if [ -f /app/caria_data/models/regime_hmm_model.pkl ]; then \
        cp /app/caria_data/models/regime_hmm_model.pkl /app/models/regime_hmm_model.pkl && \
        echo "✓ Regime HMM model copied to /app/models/"; \
    else \
        echo "⚠ Warning: regime_hmm_model.pkl not found - regime detection may use defaults"; \
    fi
```

### Si el modelo NO está en git:

1. **Agregar al repositorio:**
```bash
git add caria_data/models/regime_hmm_model.pkl
git commit -m "Add HMM model"
git push
```

2. **O usar Git LFS si es muy grande:**
```bash
git lfs track "*.pkl"
git add caria_data/models/regime_hmm_model.pkl
git commit -m "Add HMM model (LFS)"
git push
```

## ✅ Opción 4: Entrenar el Modelo en Render (Después del Deploy)

Si el modelo no está disponible durante el build:

1. **Conecta a Render Shell:**
   - Render Dashboard → tu servicio → **Shell** tab

2. **Entrena el modelo:**
```bash
cd /app/caria_data
python train_hmm_simple.py
```

3. **Copia al directorio correcto:**
```bash
cp models/regime_hmm_model.pkl /app/models/regime_hmm_model.pkl
```

4. **Reinicia el servicio:**
   - Render Dashboard → **Manual Deploy** → **Deploy latest commit**

**⚠️ NOTA:** Este método requiere que los datos macro (`fred_data.parquet`) estén disponibles en el contenedor.

## 🔍 Verificar que el Modelo Está Disponible en Render

### Test 1: Health Check
```bash
curl https://caria-api.onrender.com/health
```

**Debería mostrar:**
```json
{
  "status": "ok",
  "regime": "available"  // ✅ Disponible
}
```

Si muestra `"regime": "unavailable"`, el modelo no está cargado.

### Test 2: Regime Endpoint
```bash
curl https://caria-api.onrender.com/api/regime/current
```

**Con modelo:**
```json
{
  "regime": "expansion",
  "probabilities": {
    "expansion": 0.65,
    "slowdown": 0.20,
    "recession": 0.10,
    "stress": 0.05
  },
  "confidence": 0.85,
  "features_used": {...}
}
```

**Sin modelo (fallback):**
```json
{
  "regime": "slowdown",
  "probabilities": {
    "expansion": 0.2,
    "slowdown": 0.5,
    "recession": 0.2,
    "stress": 0.1
  },
  "confidence": 0.5,
  "features_used": {}
}
```

## 📋 Checklist para Render

- [ ] Modelo existe en `caria_data/models/regime_hmm_model.pkl` localmente
- [ ] Modelo está en git (o configurado Git LFS)
- [ ] Dockerfile copia el modelo correctamente (verificar logs de build)
- [ ] Health check muestra `"regime": "available"`
- [ ] Endpoint `/api/regime/current` retorna datos reales (no fallback)

## 🆘 Troubleshooting

### Problema: Modelo no se copia durante build

**Solución:**
1. Verifica que el archivo existe: `ls -la caria_data/models/regime_hmm_model.pkl`
2. Verifica que está en git: `git ls-files caria_data/models/regime_hmm_model.pkl`
3. Si no está en git, agrégalo y haz push
4. Revisa logs de build en Render para ver el mensaje de copia

### Problema: Modelo existe pero RegimeService no lo encuentra

**Solución:**
1. Verifica la ruta en logs: busca "Modelo HMM no encontrado en"
2. El modelo debe estar en `/app/models/regime_hmm_model.pkl` en el contenedor
3. Verifica que el Dockerfile lo copió correctamente

### Problema: No tengo datos macro para entrenar

**Solución:**
1. Los datos macro deben estar en `caria_data/data/silver/macro/fred_data.parquet`
2. Si no existen, ejecuta primero el pipeline de datos macro
3. O usa el modelo pre-entrenado si está disponible en el repositorio

## 📝 Notas Importantes

1. **El modelo es grande (~1-5MB)** - considera usar Git LFS si tienes problemas con git
2. **El entrenamiento toma tiempo** - puede tardar varios minutos
3. **El modelo necesita datos históricos** - mínimo 1990-2024 para buen rendimiento
4. **Sin el modelo, algunas funcionalidades usarán valores por defecto** - pero es mejor tener el modelo real


