import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

# Cargar variables si estamos en local
load_dotenv()

# ==========================================
# CONFIGURACIÓN DE LA BASE DE DATOS
# ==========================================
DATABASE_URL = os.getenv("DATABASE_URL")

# CORRECCIÓN PARA RAILWAY/SQLALCHEMY:
# SQLAlchemy requiere 'postgresql://', pero Railway a veces da 'postgres://'
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    # Esto es solo para que no explote si corres el script sin variables, 
    # pero en producción debe existir.
    print("⚠️ ADVERTENCIA: No se encontró DATABASE_URL.")
    engine = None
    SessionLocal = None
else:
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# ==========================================
# MODELO DE LA TABLA
# ==========================================
class ScreeningResult(Base):
    __tablename__ = "screenings_semanales"

    id = Column(Integer, primary_key=True, index=True)
    fecha_reporte = Column(DateTime, default=datetime.utcnow)
    ticker = Column(String, index=True)
    c_score = Column(Float)
    quality_score = Column(Float)
    valuation_score = Column(Float)
    momentum_score = Column(Float)
    catalysts_score = Column(Float)
    risk_penalty = Column(Float)

# ==========================================
# FUNCIONES
# ==========================================
def init_db():
    """Crea la tabla en Neon si no existe"""
    if engine:
        Base.metadata.create_all(bind=engine)
        print("✅ Base de datos inicializada/verificada en Neon.")

def guardar_resultados(df_resultados):
    """Guarda el DataFrame de pandas en Neon"""
    if not SessionLocal:
        print("❌ No hay conexión a base de datos.")
        return

    session = SessionLocal()
    try:
        count = 0
        for _, row in df_resultados.iterrows():
            desglose = row['Desglose']
            
            nuevo = ScreeningResult(
                ticker=row['Ticker'],
                c_score=row['C_Score'],
                quality_score=desglose['Quality'],
                valuation_score=desglose['Valuation'],
                momentum_score=desglose['Momentum'],
                catalysts_score=desglose['Catalysts'],
                risk_penalty=desglose['Risk_Safety']
            )
            session.add(nuevo)
            count += 1
        
        session.commit()
        print(f"💾 Guardados {count} registros en Neon.")
        return {"status": "success", "saved": count}
    except Exception as e:
        session.rollback()
        print(f"❌ Error guardando en DB: {e}")
        return {"status": "error", "detail": str(e)}
    finally:
        session.close()
