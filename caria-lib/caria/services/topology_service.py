import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
try:
    import gudhi
except ImportError:
    gudhi = None

LOGGER = logging.getLogger(__name__)

class TopologyService:
    """
    Caria Topological MRI (TDA Engine).
    
    Uses Persistent Homology to scan the "shape" of the market.
    - Metric: Distance = sqrt(2 * (1 - Correlation))
    - Filtration: Rips Complex
    - Diagnosis: Betti Numbers (H0, H1) & Persistence Landscapes
    """
    
    def __init__(self):
        self.window_size = 60  # days for correlation
        if not gudhi:
            LOGGER.warning("GUDHI library not found. Topological MRI will be disabled.")

    def compute_distance_matrix(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Converts returns to correlation distance matrix.
        D_ij = sqrt(2 * (1 - rho_ij))
        """
        # Correlation matrix
        corr = returns.corr()
        
        # Distance matrix (Gower's distance for correlation)
        # Range: 0 (perfectly correlated) to 2 (perfectly anti-correlated)
        dist = np.sqrt(2 * (1 - corr))
        
        # Fill diagonal with 0 and handle potential floating point errors
        np.fill_diagonal(dist.values, 0)
        dist = dist.fillna(2) # Assume uncorrelated if NaN
        
        return corr, dist.values

    def scan_market_topology(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs the Topological MRI scan.
        """
        if not gudhi or returns.empty:
            return {"status": "OFFLINE", "reason": "No GUDHI or Data"}

        try:
            # 1. Prepare the "Tissue" (Point Cloud from Distance Matrix)
            corr_df, dist_matrix = self.compute_distance_matrix(returns)
            
            # 2. The "Filtration" (Rips Complex)
            # max_dimension=2 means we compute H0 (clusters) and H1 (loops)
            rips_complex = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=2.0)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            
            # 3. The "Barcode" (Persistence)
            # persistence() returns list of (dimension, (birth, death))
            diag = simplex_tree.persistence()
            
            # 4. Analyze Betti Numbers & Features
            betti_0 = 0 # Connected components (at threshold epsilon)
            betti_1 = 0 # Loops/Holes (at threshold epsilon)
            
            # We look at persistence intervals.
            # "Significant" loops are those that persist for a long range of filtration values.
            # A "Collapse" is when H1 loops disappear or become very short-lived.
            
            total_h1_persistence = 0.0
            max_h1_life = 0.0
            
            for dim, (birth, death) in diag:
                if death == float('inf'):
                    life = 2.0 - birth # Max distance is 2
                else:
                    life = death - birth
                
                if dim == 1:
                    total_h1_persistence += life
                    max_h1_life = max(max_h1_life, life)
                    if life > 0.1: # Filter noise
                        betti_1 += 1
                elif dim == 0:
                    if life > 0.1:
                        betti_0 += 1

            # 5. Diagnosis
            # Healthy Market: High H1 persistence (sponge-like, complex structure)
            # Collapsing Market: Low H1 persistence (brick-like, simple structure)
            
            complexity_score = min(100, (total_h1_persistence * 10)) # Normalize roughly
            
            if complexity_score < 20:
                diagnosis = "CRITICAL: TOPOLOGY COLLAPSE"
                description = "Market manifold has rigidified. Loops closed. Systemic fragility HIGH."
                status_color = "RED"
            elif complexity_score < 50:
                diagnosis = "WARNING: STRUCTURAL DECAY"
                description = "Market complexity decreasing. Correlations tightening."
                status_color = "YELLOW"
            else:
                diagnosis = "HEALTHY: COMPLEX MANIFOLD"
                description = "High dimensional structure intact. Sponge-like topology."
                status_color = "GREEN"

            # 6. Find "Topological Aliens" (Outliers)
            # Assets with high average distance to all others
            avg_dist = pd.Series(dist_matrix.mean(axis=1), index=corr_df.index)
            # Outliers are those furthest away (least correlated)
            outliers = avg_dist.nlargest(3)
            
            aliens = []
            for ticker, dist_val in outliers.items():
                aliens.append({
                    "ticker": ticker,
                    "isolation_score": float(dist_val), # Higher = more alien
                    "type": "Topological Outlier"
                })

            return {
                "status": "ONLINE",
                "diagnosis": diagnosis,
                "description": description,
                "status_color": status_color,
                "metrics": {
                    "betti_1_loops": betti_1,
                    "total_persistence": float(total_h1_persistence),
                    "max_loop_life": float(max_h1_life),
                    "complexity_score": float(complexity_score)
                },
                "aliens": aliens
            }

        except Exception as e:
            LOGGER.error(f"Topology Scan Failed: {e}")
            return {"status": "ERROR", "reason": str(e)}
