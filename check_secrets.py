"""
Script para verificar que los secrets estén configurados en Google Cloud Secret Manager.
Requiere gcloud CLI instalado y autenticado.
"""

import subprocess
import json
import sys

SECRETS_TO_CHECK = [
    "gemini-api-key",
    "fmp-api-key",
    "reddit-client-id",
    "reddit-client-secret",
    "postgres-password",
    "jwt-secret-key"
]

def check_secret_exists(secret_name: str, project_id: str = "caria-backend") -> dict:
    """Verifica si un secret existe en Secret Manager."""
    try:
        result = subprocess.run(
            ["gcloud", "secrets", "describe", secret_name, 
             "--project", project_id, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            secret_info = json.loads(result.stdout)
            return {
                "exists": True,
                "name": secret_info.get("name", secret_name),
                "create_time": secret_info.get("createTime", "Unknown"),
                "replication": secret_info.get("replication", {})
            }
        else:
            return {
                "exists": False,
                "error": result.stderr.strip()
            }
    except FileNotFoundError:
        return {
            "exists": False,
            "error": "gcloud CLI not found. Install it from: https://cloud.google.com/sdk/docs/install"
        }
    except subprocess.TimeoutExpired:
        return {
            "exists": False,
            "error": "Timeout checking secret"
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }

def check_secret_version(secret_name: str, project_id: str = "caria-backend") -> dict:
    """Verifica las versiones disponibles de un secret."""
    try:
        result = subprocess.run(
            ["gcloud", "secrets", "versions", "list", secret_name,
             "--project", project_id, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            versions = json.loads(result.stdout)
            return {
                "has_versions": len(versions) > 0,
                "versions": versions,
                "latest": versions[0] if versions else None
            }
        else:
            return {
                "has_versions": False,
                "error": result.stderr.strip()
            }
    except Exception as e:
        return {
            "has_versions": False,
            "error": str(e)
        }

def main():
    print("=" * 80)
    print("VERIFICACIÓN DE SECRETS EN GOOGLE CLOUD SECRET MANAGER")
    print("=" * 80)
    print(f"\nProyecto: caria-backend\n")
    
    results = {}
    
    for secret_name in SECRETS_TO_CHECK:
        print(f"🔍 Verificando: {secret_name}")
        result = check_secret_exists(secret_name)
        results[secret_name] = result
        
        if result["exists"]:
            print(f"   ✅ Secret existe")
            print(f"   📅 Creado: {result.get('create_time', 'Unknown')}")
            
            # Verificar versiones
            version_info = check_secret_version(secret_name)
            if version_info.get("has_versions"):
                latest = version_info.get("latest", {})
                state = latest.get("state", "Unknown")
                print(f"   📌 Versión más reciente: {latest.get('name', 'Unknown')} - Estado: {state}")
            else:
                print(f"   ⚠️  No hay versiones disponibles")
        else:
            print(f"   ❌ Secret NO existe")
            if "error" in result:
                print(f"   Error: {result['error']}")
        print()
    
    # Resumen
    print("=" * 80)
    print("RESUMEN")
    print("=" * 80)
    
    existing = sum(1 for r in results.values() if r.get("exists"))
    missing = len(SECRETS_TO_CHECK) - existing
    
    print(f"\n✅ Secrets existentes: {existing}/{len(SECRETS_TO_CHECK)}")
    print(f"❌ Secrets faltantes: {missing}/{len(SECRETS_TO_CHECK)}\n")
    
    if missing > 0:
        print("🔴 SECRETS FALTANTES:")
        for secret_name, result in results.items():
            if not result.get("exists"):
                print(f"   - {secret_name}")
        
        print("\n📝 Para crear un secret faltante:")
        print("   gcloud secrets create SECRET_NAME --project=caria-backend")
        print("   echo 'SECRET_VALUE' | gcloud secrets versions add SECRET_NAME --data-file=- --project=caria-backend")
        print("\n   Ejemplo para Reddit:")
        print("   gcloud secrets create reddit-client-id --project=caria-backend")
        print("   echo '1eIYr0z6slzt62EXy1KQ6Q' | gcloud secrets versions add reddit-client-id --data-file=- --project=caria-backend")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

