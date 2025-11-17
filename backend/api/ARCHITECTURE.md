# Arquitectura Modular Monolítica

## Per Audit Document (4.1): Modular Monolith Architecture

Este documento describe la arquitectura modular monolítica implementada según el documento de auditoría técnica.

## Estructura de Dominios

La aplicación está organizada en **5 dominios principales** con límites estrictos:

### 1. Identity Domain (`api/domains/identity/`)
**Responsabilidad**: Autenticación, gestión de usuarios, sesiones

**Endpoints**:
- `/api/auth/register` - Registro de usuarios
- `/api/auth/login` - Inicio de sesión
- `/api/auth/refresh` - Renovación de tokens
- `/api/auth/me` - Información del usuario actual

**Límites**:
- Otros dominios NO deben acceder directamente a tablas de usuarios
- Usar `get_current_user` dependency para obtener usuario autenticado
- Sesiones manejadas dentro del dominio

### 2. Portfolio Domain (`api/domains/portfolio/`)
**Responsabilidad**: Gestión de portafolios, análisis, asignación táctica

**Endpoints**:
- `/api/holdings` - Gestión de posiciones
- `/api/portfolio/analytics` - Análisis con quantstats
- `/api/portfolio/tactical/allocation` - Asignación macro-condicional (TAA)
- `/api/montecarlo/*` - Simulaciones Monte Carlo

**Límites**:
- Acceso a holdings solo a través de este dominio
- Análisis de portafolio encapsulado aquí
- No debe acceder directamente a datos de otros usuarios

### 3. Social Domain (`api/domains/social/`)
**Responsabilidad**: Comunidad, chat, interacciones sociales

**Endpoints**:
- `/api/community/posts` - Posts de comunidad
- `/api/community/posts/{id}/vote` - Votación
- `/api/chat/history` - Historial de chat
- WebSocket: `/socket.io` - Chat en tiempo real

**Límites**:
- Datos sociales aislados de otros dominios
- Chat requiere autenticación pero es independiente de identity domain internamente

### 4. Analysis Domain (`api/domains/analysis/`)
**Responsabilidad**: Análisis cuantitativo, modelos, validación

**Endpoints**:
- `/api/regime/current` - Detección de régimen
- `/api/factors/screen` - Screening de factores
- `/api/valuation/{ticker}` - Valuación de acciones
- `/api/analysis/challenge` - Análisis de tesis
- `/api/model/validation/*` - Validación del modelo

**Límites**:
- Modelos cuantitativos encapsulados
- No debe modificar datos de usuarios directamente
- Solo lectura de datos de mercado

### 5. Market Data Domain (`api/domains/market_data/`)
**Responsabilidad**: Datos de mercado en tiempo real

**Endpoints**:
- `/api/prices/realtime` - Precios en tiempo real

**Límites**:
- Solo lectura de datos de mercado
- No debe modificar datos de usuarios
- Fuente única de verdad para datos de mercado

## Principios de Diseño

### 1. Límites Estrictos
- Cada dominio es independiente y autocontenido
- Comunicación entre dominios solo a través de APIs públicas
- No hay dependencias circulares entre dominios

### 2. Persistencia Poliglota
- **PostgreSQL**: Datos transaccionales (usuarios, holdings, posts)
- **Redis** (futuro): Datos volátiles (sesiones, caché, WebSocket state)

### 3. APIs Idempotentes
Endpoints críticos deben ser idempotentes (pueden llamarse múltiples veces sin efectos secundarios):

- ✅ `/api/holdings` (POST) - Usa `ticker` como clave única
- ✅ `/api/community/posts/{id}/vote` - Usa `(post_id, user_id)` como clave única
- ✅ `/api/portfolio/tactical/allocation` - Solo lectura, idempotente por naturaleza

### 4. Separación de Responsabilidades
- **Routes**: Solo manejan HTTP, validación, y delegación
- **Services**: Lógica de negocio y acceso a datos
- **Models**: Estructuras de datos y validación

## Migración Gradual

Los routers legacy (`api/routes/*`) se mantienen por compatibilidad pero están marcados como deprecated. 
La migración completa a dominios se completará en fases:

1. ✅ Fase 1: Estructura de dominios creada
2. ⏳ Fase 2: Migrar servicios a dominios
3. ⏳ Fase 3: Deprecar routers legacy
4. ⏳ Fase 4: Implementar Redis para datos volátiles

## Beneficios

1. **Escalabilidad**: Cada dominio puede escalarse independientemente
2. **Mantenibilidad**: Código organizado por responsabilidad
3. **Testabilidad**: Dominios pueden testearse de forma aislada
4. **Migración futura**: Fácil migrar a microservicios si es necesario

