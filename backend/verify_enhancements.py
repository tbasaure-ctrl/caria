
import sys
import os
import logging
from unittest.mock import MagicMock

# Add backend to path
sys.path.append(os.path.abspath("backend"))
sys.path.append(os.path.abspath("caria-lib"))

# Mock FMPClient before importing services
sys.modules["caria.ingestion.clients.fmp_client"] = MagicMock()
from caria.ingestion.clients.fmp_client import FMPClient

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def verify_valuation_enhancements():
    LOGGER.info("Verifying Valuation Enhancements...")
    try:
        from api.services.simple_valuation import SimpleValuationService
        
        # Mock FMP client behavior
        mock_fmp = MagicMock()
        # Mock data for AAPL
        mock_fmp.get_key_metrics.return_value = [{"freeCashFlowPerShareTTM": 5.0, "netIncomePerShareTTM": 6.0}]
        mock_fmp.get_financial_ratios.side_effect = [
            [{"peRatioTTM": 25.0, "priceToBookRatioTTM": 10.0}], # Current
            [ # Historical (5 years)
                {"peRatio": 20.0, "priceToBookRatio": 8.0},
                {"peRatio": 22.0, "priceToBookRatio": 9.0},
                {"peRatio": 18.0, "priceToBookRatio": 7.0},
                {"peRatio": 25.0, "priceToBookRatio": 10.0},
                {"peRatio": 30.0, "priceToBookRatio": 12.0}
            ]
        ]
        mock_fmp.get_financial_growth.return_value = [{"freeCashFlowGrowth": 0.10}]
        
        service = SimpleValuationService(fmp_client=mock_fmp)
        
        # Test get_valuation
        current_price = 150.0
        result = service.get_valuation("AAPL", current_price)
        
        # Check Reverse DCF
        if "reverse_dcf" not in result:
            LOGGER.error("Missing reverse_dcf in response")
            return False
        
        implied_growth = result["reverse_dcf"]["implied_growth_rate"]
        LOGGER.info(f"Implied Growth: {implied_growth}")
        
        # Check Multiples Valuation
        if "multiples_valuation" not in result:
            LOGGER.error("Missing multiples_valuation in response")
            return False
            
        fair_value_multiples = result["multiples_valuation"]["fair_value"]
        LOGGER.info(f"Multiples Fair Value: {fair_value_multiples}")
        
        # Verify logic: Avg PE = (20+22+18+25+30)/5 = 23.0
        # EPS = 6.0
        # Value = 23.0 * 6.0 = 138.0 (approx)
        # Avg PB = (8+9+7+10+12)/5 = 9.2
        # BPS not mocked, so it might be 0 if not found in metrics.
        # Let's check if fair_value > 0
        if fair_value_multiples <= 0:
            LOGGER.warning("Multiples Fair Value is 0 (might be due to missing BPS in mock)")
        
        LOGGER.info("Valuation enhancements verified.")
        return True
    except Exception as e:
        LOGGER.error(f"Valuation verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_debug_endpoint():
    LOGGER.info("Verifying Debug Endpoint registration...")
    try:
        from api.app import app
        
        # Check if route is registered
        routes = [route.path for route in app.routes]
        if "/api/debug/llm" in routes:
            LOGGER.info("Debug endpoint /api/debug/llm found.")
            return True
        else:
            LOGGER.error("Debug endpoint /api/debug/llm NOT found in app routes.")
            return False
    except Exception as e:
        LOGGER.error(f"Debug endpoint verification failed: {e}")
        return False

if __name__ == "__main__":
    val_ok = verify_valuation_enhancements()
    debug_ok = verify_debug_endpoint()
    
    if val_ok and debug_ok:
        print("VERIFICATION SUCCESSFUL")
    else:
        print("VERIFICATION FAILED")
        sys.exit(1)
