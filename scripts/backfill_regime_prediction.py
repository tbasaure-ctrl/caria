     # scripts/backfill_regime_predictions.py
import os
from datetime import datetime, timedelta

import psycopg2
from caria.services.regime_service import RegimeService
        from api.services.model_validation import get_model_validation_service

     conn = psycopg2.connect(
         host=os.getenv("POSTGRES_HOST", "localhost"),
         port=int(os.getenv("POSTGRES_PORT", "5432")),
         user=os.getenv("POSTGRES_USER", "caria_user"),
         password=os.getenv("POSTGRES_PASSWORD"),
         database=os.getenv("POSTGRES_DB", "caria"),
     )

     service = RegimeService(conn)
     cursor = conn.cursor()

     start = datetime(2023, 1, 1)
     end = datetime(2024, 1, 1)
     current = start

     while current <= end:
         regime = service.get_current_regime(date=current)
         cursor.execute(
             """
             INSERT INTO regime_predictions (date, predicted_regime, probabilities)
             VALUES (%s, %s, %s)
             ON CONFLICT (date) DO UPDATE SET
                predicted_regime = EXCLUDED.predicted_regime,
                probabilities = EXCLUDED.probabilities
             """,
             (current.date(), regime["regime"], regime["probabilities"])
         )
         current += timedelta(days=1)

     conn.commit()
     cursor.close()
     conn.close()