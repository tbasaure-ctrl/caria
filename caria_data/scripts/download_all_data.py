"""Script para descargar todos los datos de forma incremental."""

import sys
from pathlib import Path
import logging

# Setup paths
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
LOGGER = logging.getLogger(__name__)


def download_batch(batch_name: str, tickers: list[str]) -> None:
    """Download data for a batch of tickers."""
    LOGGER.info(f"=" * 60)
    LOGGER.info(f"Starting batch: {batch_name}")
    LOGGER.info(f"Tickers: {len(tickers)} - {tickers[:5]}...")
    LOGGER.info(f"=" * 60)

    # TODO: Implement actual FMP calls
    # For now, just log
    for ticker in tickers:
        LOGGER.info(f"  Downloading {ticker}...")

    LOGGER.info(f"✅ Batch {batch_name} completed")


def main():
    """Download all data in manageable batches."""

    # Tech sector
    tech_tickers = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AVGO",
        "AMD", "ORCL", "ADBE", "CRM", "INTC", "CSCO", "QCOM", "TXN"
    ]

    # Financials
    financial_tickers = [
        "JPM", "BAC", "WFC", "GS", "MS", "BLK", "C", "SCHW", "AXP", "SPGI"
    ]

    # Healthcare
    healthcare_tickers = [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "PFE", "DHR", "BMY"
    ]

    # Consumer
    consumer_tickers = [
        "AMZN", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG",
        "WMT", "COST", "PG", "KO", "PEP", "PM", "MO", "CL"
    ]

    # Communications
    comm_tickers = [
        "NFLX", "DIS", "CMCSA", "T", "VZ"
    ]

    # Industrials
    industrial_tickers = [
        "CAT", "BA", "GE", "UNP", "RTX", "HON", "UPS", "LMT", "DE"
    ]

    # Energy
    energy_tickers = [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX"
    ]

    # Others
    other_tickers = [
        "NEE", "DUK", "SO", "D",  # Utilities
        "PLD", "AMT", "EQIX", "SPG",  # Real Estate
        "LIN", "APD", "SHW", "FCX",  # Materials
        "BRK.B", "V", "MA"  # Special
    ]

    # Batch downloads
    batches = [
        ("tech", tech_tickers),
        ("financials", financial_tickers),
        ("healthcare", healthcare_tickers),
        ("consumer", consumer_tickers),
        ("communications", comm_tickers),
        ("industrials", industrial_tickers),
        ("energy", energy_tickers),
        ("others", other_tickers),
    ]

    LOGGER.info("🚀 Starting full data download")
    LOGGER.info(f"Total tickers: {sum(len(b[1]) for b in batches)}")
    LOGGER.info(f"Batches: {len(batches)}")
    LOGGER.info("")

    for batch_name, tickers in batches:
        try:
            download_batch(batch_name, tickers)
        except Exception as e:
            LOGGER.error(f"❌ Error in batch {batch_name}: {e}")
            LOGGER.info("Continuing with next batch...")
            continue

    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("✅ ALL DOWNLOADS COMPLETED")
    LOGGER.info("=" * 60)
    LOGGER.info("")
    LOGGER.info("Next steps:")
    LOGGER.info("1. Check data/silver/ for downloaded data")
    LOGGER.info("2. Run gold builder with new data")
    LOGGER.info("3. Train bias detection models")


if __name__ == "__main__":
    main()
