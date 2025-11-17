# 💳 Configurar Facturación en Google Cloud Platform

## ⚠️ Error Común

Si ves este error:
```
ERROR: Billing account for project 'XXXXX' is not found. 
Billing must be enabled for activation of service(s)
```

**Significa que necesitas habilitar la facturación antes de usar servicios de GCP.**

## 🚀 Solución Rápida

### Paso 1: Habilitar Facturación (5 minutos)

#### Opción A: Desde la Consola Web (Recomendado)

1. **Ve a la consola de facturación:**
   ```
   https://console.cloud.google.com/billing
   ```

2. **Si no tienes cuenta de facturación:**
   - Click en "Create Billing Account"
   - Completa el formulario con tu información
   - Agrega una tarjeta de crédito (requerida, pero no se cobrará automáticamente)
   - **Nota**: GCP ofrece $300 USD de crédito gratis para nuevos usuarios

3. **Vincular cuenta al proyecto:**
   - Ve a: https://console.cloud.google.com/billing/projects
   - Selecciona tu proyecto `caria-backend` (o el que estés usando)
   - Click en "Link Billing Account"
   - Selecciona tu cuenta de facturación

#### Opción B: Desde la Línea de Comandos

```bash
# 1. Listar cuentas de facturación disponibles
gcloud billing accounts list

# 2. Vincular cuenta de facturación al proyecto
# Reemplaza BILLING_ACCOUNT_ID con el ID de tu cuenta
gcloud billing projects link TU_PROYECTO_ID --billing-account=BILLING_ACCOUNT_ID

# Ejemplo:
# gcloud billing projects link caria-backend --billing-account=01ABCD-2EFGH3-4IJKL5
```

### Paso 2: Verificar que la Facturación Está Habilitada

```bash
# Verificar estado de facturación del proyecto
gcloud billing projects describe TU_PROYECTO_ID

# Deberías ver algo como:
# billingAccountName: billingAccounts/01ABCD-2EFGH3-4IJKL5
# billingEnabled: true
```

### Paso 3: Habilitar APIs (Ahora Sí Funcionará)

```bash
# Habilitar APIs necesarias
gcloud services enable \
    run.googleapis.com \
    sqladmin.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com \
    artifactregistry.googleapis.com \
    containerregistry.googleapis.com
```

## 💰 Información sobre Costos

### Crédito Gratis de GCP

- **$300 USD** de crédito gratis para nuevos usuarios
- Válido por **90 días**
- Perfecto para probar y desarrollar

### Costos Estimados para Caria

- **Cloud Run**: 
  - Primeros 2 millones de requests/mes: **GRATIS**
  - Después: ~$0.40 por millón de requests
  - **Para desarrollo/pruebas: Probablemente GRATIS**

- **Cloud SQL (db-f1-micro)**:
  - ~$7.50/mes (instancia pequeña)
  - Puedes usar Cloud SQL con pgvector

- **Cloud Build**:
  - Primeros 120 minutos/día: **GRATIS**
  - Después: ~$0.003/minuto

- **Secret Manager**:
  - Primeros 6 secretos: **GRATIS**
  - Después: ~$0.06/secret/mes

### Total Estimado

- **Primeros 3 meses**: $0 (usando crédito gratis)
- **Después (con poco tráfico)**: ~$7-10/mes
- **Con tráfico moderado**: ~$15-25/mes

### Protecciones contra Cobros Inesperados

1. **Límites de presupuesto:**
   ```bash
   # Crear alerta de presupuesto
   # Ve a: https://console.cloud.google.com/billing/budgets
   ```

2. **Desactivar facturación automática:**
   - GCP no cobra automáticamente más allá del crédito gratis
   - Puedes configurar alertas cuando el uso alcance cierto umbral

3. **Eliminar recursos cuando no los uses:**
   - Cloud Run escala a 0 cuando no hay tráfico (no cobra)
   - Puedes pausar Cloud SQL cuando no lo uses

## ✅ Checklist

- [ ] Crear cuenta de facturación en GCP Console
- [ ] Vincular cuenta de facturación al proyecto
- [ ] Verificar que `billingEnabled: true`
- [ ] Habilitar APIs necesarias
- [ ] Configurar alertas de presupuesto (opcional pero recomendado)

## 🔗 Enlaces Útiles

- **Consola de Facturación**: https://console.cloud.google.com/billing
- **Proyectos y Facturación**: https://console.cloud.google.com/billing/projects
- **Precios de Cloud Run**: https://cloud.google.com/run/pricing
- **Precios de Cloud SQL**: https://cloud.google.com/sql/pricing
- **Crédito Gratis**: https://cloud.google.com/free

## 🆘 Troubleshooting

### Error: "Billing account not found"
- Verifica que creaste una cuenta de facturación
- Verifica que la vinculaste al proyecto correcto

### Error: "Permission denied"
- Necesitas permisos de "Billing Account User" o "Owner"
- Verifica tus permisos en: https://console.cloud.google.com/iam-admin/iam

### ¿Puedo usar GCP sin tarjeta de crédito?
- No, GCP requiere tarjeta de crédito para habilitar facturación
- Pero puedes usar el crédito gratis de $300 sin que se cobre nada
- Puedes configurar límites de presupuesto para evitar cobros

## 🎯 Próximo Paso

Una vez habilitada la facturación, continúa con:
```bash
./setup-gcp.sh
```

O sigue las instrucciones en `GCP_MIGRATION_GUIDE.md`

