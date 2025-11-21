
import sys
import os
import logging
from unittest.mock import MagicMock

# Add backend to path
sys.path.append(os.path.abspath("backend"))

# Mock FMPClient before importing services
sys.modules["caria.ingestion.clients.fmp_client"] = MagicMock()
from caria.ingestion.clients.fmp_client import FMPClient

# Setup logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def verify_valuation_service():
    LOGGER.info("Verifying SimpleValuationService...")
    try:
        from api.services.simple_valuation import SimpleValuationService
        
        # Mock FMP client behavior
        mock_fmp = MagicMock()
        mock_fmp.get_key_metrics.return_value = [{"freeCashFlowPerShareTTM": 5.0, "currency": "USD"}]
        mock_fmp.get_financial_ratios.return_value = [{"peRatioTTM": 20.0}]
        mock_fmp.get_financial_growth.return_value = [{"freeCashFlowGrowth": 0.10}]
        
        service = SimpleValuationService(fmp_client=mock_fmp)
        
        # Test get_valuation
        result = service.get_valuation("AAPL", 150.0)
        
        if not result:
            LOGGER.error("Valuation service returned None")
            return False
            
        if result.get("dcf", {}).get("method") == "Error":
            LOGGER.error(f"Valuation service returned Error: {result['dcf']['explanation']}")
            return False

        # Verify correct methods were called
        mock_fmp.get_key_metrics.assert_called_with("AAPL", period="quarter")
        mock_fmp.get_financial_ratios.assert_called_with("AAPL", period="quarter")
        
        LOGGER.info("SimpleValuationService verified successfully.")
        return True
    except Exception as e:
        LOGGER.error(f"SimpleValuationService verification failed: {e}")
        return False

def verify_llm_service():
    LOGGER.info("Verifying LLMService logging...")
    try:
        from api.services.llm_service import LLMService
        
        # Check if the code contains the logging line
        with open("backend/api/services/llm_service.py", "r") as f:
            content = f.read()
            if 'LOGGER.warning(f"Llama call failed: {e}")' in content:
                LOGGER.info("LLMService logging verified.")
                return True
            else:
                LOGGER.error("LLMService logging line not found.")
                return False
    except Exception as e:
        LOGGER.error(f"LLMService verification failed: {e}")
        return False

if __name__ == "__main__":
    val_ok = verify_valuation_service()
    llm_ok = verify_llm_service()
    
    if val_ok and llm_ok:
        print("VERIFICATION SUCCESSFUL")
    else:
        print("VERIFICATION FAILED")
        sys.exit(1)
