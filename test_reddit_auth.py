"""
Test directo de autenticación de Reddit para diagnosticar el problema.
"""

import os
import praw

# Credenciales actuales
CLIENT_ID = "1eIYr0z6slzt62EXy1KQ6Q"
CLIENT_SECRET = "p53Yud4snfuadHAvgva_6vWkj0eXcw"

print("="*60)
print("TEST DE AUTENTICACIÓN DE REDDIT")
print("="*60)

# Test 1: User Agent simple (formato básico)
print("\n1. Probando con User Agent simple...")
try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent="Caria-Investment-App-v1.0",
        check_for_async=False
    )
    # Intentar acceder a un subreddit público
    subreddit = reddit.subreddit("test")
    print(f"   Subreddit name: {subreddit.display_name}")
    print("   ✅ Autenticación exitosa!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: User Agent con formato recomendado por Reddit
print("\n2. Probando con User Agent formato Reddit recomendado...")
try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent="web:CariaInvestmentApp:v1.0 (by /u/caria_app)",
        check_for_async=False
    )
    subreddit = reddit.subreddit("test")
    print(f"   Subreddit name: {subreddit.display_name}")
    print("   ✅ Autenticación exitosa!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: User Agent con formato alternativo
print("\n3. Probando con User Agent formato alternativo...")
try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent="CariaInvestmentApp/1.0",
        check_for_async=False
    )
    subreddit = reddit.subreddit("test")
    print(f"   Subreddit name: {subreddit.display_name}")
    print("   ✅ Autenticación exitosa!")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Verificar credenciales directamente
print("\n4. Verificando credenciales...")
print(f"   Client ID: {CLIENT_ID[:10]}...")
print(f"   Client Secret: {CLIENT_SECRET[:10]}...")
print("   ⚠️ Si todos fallan, las credenciales pueden ser incorrectas")
print("   ⚠️ O Reddit puede requerir OAuth flow completo para algunas operaciones")

