"""
Run migration to add arena columns to community_posts table.
This script can be run locally or on Railway to apply the migration.

Usage:
    python run_arena_migration.py
"""
import os
import psycopg2
from pathlib import Path

def run_migration():
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        return False
    
    # Read migration SQL
    migration_file = Path(__file__).parent / 'migrations' / 'add_arena_columns_to_community_posts.sql'
    
    if not migration_file.exists():
        print(f"ERROR: Migration file not found: {migration_file}")
        return False
    
    with open(migration_file, 'r') as f:
        migration_sql = f.read()
    
    # Connect and run migration
    try:
        print("Connecting to database...")
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        print("Running migration...")
        cursor.execute(migration_sql)
        
        # Get results from verification query
        results = cursor.fetchall()
        
        conn.commit()
        
        print("\n✅ Migration completed successfully!")
        print("\nAdded columns:")
        for row in results:
            column_name, data_type, nullable, default = row
            print(f"  - {column_name}: {data_type} (nullable={nullable}, default={default})")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    success = run_migration()
    exit(0 if success else 1)
