# Cómo Crear un Personal Access Token (PAT) en GitHub

## Método 1: Desde GitHub.com (Recomendado)

### Pasos:

1. **Inicia sesión en GitHub**
   - Ve a https://github.com
   - Inicia sesión con tu cuenta

2. **Ve a Settings**
   - Haz clic en tu foto de perfil (arriba a la derecha)
   - Selecciona **Settings**

3. **Accede a Developer settings**
   - En el menú lateral izquierdo, baja hasta el final
   - Haz clic en **Developer settings**

4. **Personal access tokens**
   - En el menú lateral, selecciona **Personal access tokens**
   - Luego haz clic en **Tokens (classic)** o **Fine-grained tokens**

   **Opciones:**
   - **Tokens (classic)**: Más simple, permisos amplios
   - **Fine-grained tokens**: Más seguro, permisos específicos (requiere repositorio/org específico)

5. **Generar nuevo token**
   - Haz clic en **Generate new token** o **Generate new token (classic)**

6. **Configurar token**
   
   **Para Tokens (classic):**
   - **Note**: Nombre descriptivo (ej: "Caria Project - Cloud Run Deployment")
   - **Expiration**: Selecciona duración (90 días, 1 año, o sin expiración)
   - **Select scopes**: Marca los permisos necesarios:
     - ✅ `repo` - Control total de repositorios privados
     - ✅ `workflow` - Actualizar GitHub Actions workflows
     - ✅ `write:packages` - Subir paquetes
     - ✅ `read:org` - Leer información de organización (si aplica)
   
   **Para Fine-grained tokens:**
   - Selecciona repositorio(s) específicos
   - Configura permisos granulares
   - Permisos mínimos necesarios:
     - Repository permissions → Contents: Read and write
     - Repository permissions → Actions: Read and write
     - Repository permissions → Metadata: Read-only

7. **Generar y copiar**
   - Haz clic en **Generate token** (abajo)
   - **⚠️ IMPORTANTE:** Copia el token inmediatamente
   - **El token SOLO se muestra UNA VEZ**
   - Si lo pierdes, tendrás que crear uno nuevo

8. **Guardar de forma segura**
   - Guarda el token en un gestor de contraseñas
   - O en un archivo seguro (no subir a GitHub)
   - Ejemplo: `GITHUB_TOKEN.txt` (agregar a `.gitignore`)

## Método 2: URL Directa

Puedes ir directamente a:
- **Classic tokens:** https://github.com/settings/tokens
- **Fine-grained tokens:** https://github.com/settings/tokens?type=beta

## Uso del Token

### En la línea de comandos (Git)

```bash
# Usar token como contraseña cuando Git pida credenciales
git push origin main
# Username: tu-usuario-github
# Password: pega-tu-token-aqui
```

### Configurar Git para usar token automáticamente

**Opción A: Configurar globalmente**
```bash
git config --global credential.helper store
# La primera vez que uses git push, ingresarás usuario y token
# Después se guardará automáticamente
```

**Opción B: Usar token en URL (temporal)**
```bash
git remote set-url origin https://TU_TOKEN@github.com/usuario/repo.git
```

**Opción C: Variable de entorno (recomendado para CI/CD)**
```bash
# En Windows PowerShell
$env:GITHUB_TOKEN = "tu-token-aqui"
git push origin main

# O agregar permanentemente al perfil de PowerShell
[System.Environment]::SetEnvironmentVariable('GITHUB_TOKEN', 'tu-token-aqui', 'User')
```

### En GitHub Actions

```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    steps:
      - uses: actions/checkout@v3
        with:
          token: ${{ secrets.GITHUB_TOKEN }}  # Token automático de GitHub Actions
```

Para usar un PAT personal:
1. Ve a Settings → Secrets → Actions
2. Crea nuevo secret llamado `PERSONAL_ACCESS_TOKEN`
3. Pega tu token
4. Usa en workflow: `${{ secrets.PERSONAL_ACCESS_TOKEN }}`

## Permisos Recomendados para Este Proyecto

Para el proyecto Caria (Cloud Run deployment), necesitas:

**Scopes mínimos (Classic token):**
- ✅ `repo` - Para push/pull
- ✅ `workflow` - Para actualizar workflows de GitHub Actions

**Si usas Artifact Registry:**
- ✅ `write:packages` - Para subir imágenes Docker

## Seguridad

### ⚠️ NUNCA:
- ❌ Compartir tu token públicamente
- ❌ Subirlo a GitHub en commits
- ❌ Dejarlo en código fuente
- ❌ Enviarlo por email o chat no seguro

### ✅ SÍ:
- ✅ Usar `.gitignore` para archivos con tokens
- ✅ Usar GitHub Secrets para CI/CD
- ✅ Rotar tokens periódicamente
- ✅ Usar tokens con expiración
- ✅ Usar Fine-grained tokens cuando sea posible
- ✅ Dar permisos mínimos necesarios

## Revocar Token

Si pierdes o comprometes tu token:

1. Ve a https://github.com/settings/tokens
2. Encuentra el token en la lista
3. Haz clic en el botón de **revoke** (revocar)
4. Crea un nuevo token

## Verificar Token

Puedes probar tu token con:

```bash
# Probar autenticación
curl -H "Authorization: token TU_TOKEN" https://api.github.com/user

# Debería devolver tu información de usuario
```

## Troubleshooting

**Error: "Authentication failed"**
- Verifica que el token esté correcto
- Verifica que tenga los permisos necesarios
- Verifica que no haya expirado

**Error: "Permission denied"**
- Asegúrate de tener permisos en el repositorio
- Verifica que el scope `repo` esté habilitado

**Error: "Token revoked"**
- El token fue revocado o expiró
- Crea un nuevo token

## Recursos

- Documentación oficial: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token
- Fine-grained tokens: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens






