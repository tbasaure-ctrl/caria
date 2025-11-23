"""
FMP-based scoring service.

This is currently an alias to ScoringService which uses OpenBB/FMP under the hood.
In the future, this can be refactored to use FMP directly if needed.
"""

from api.services.scoring_service import ScoringService

# Alias FMPScoringService to ScoringService
# The underlying ScoringService already uses FMP via OpenBB client
FMPScoringService = ScoringService
