import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import logging
from typing import Dict, Any, List

try:
    import gudhi
except ImportError:
    gudhi = None

LOGGER = logging.getLogger(__name__)

class TopologicalMRI:
    """
    The Topological MRI Engine.
    Uses Minimum Spanning Tree (MST) and Persistent Homology to diagnose market structure.
    """
    
    def scan(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Performs the Topological MRI Scan.
        Identifies 'Aliens' by analyzing the Market's Minimum Spanning Tree.
        """
        assets = returns_df.columns.tolist()
        
        # 1. Construct the Metric Space
        # Distance metric: d(x,y) = sqrt(2 * (1 - correlation))
        corr_matrix = returns_df.corr().values
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(dist_matrix, 0) # Self-distance is 0
        
        # 2. Extract Topology (Minimum Spanning Tree)
        # The MST represents the 'skeleton' of the market.
        # Assets connected by 'Long Edges' are topologically isolated.
        mst = minimum_spanning_tree(dist_matrix)
        mst_array = mst.toarray()
        
        # 3. Calculate "Isolation Score" (The Alien Metric)
        # For each node, what is the connection strength to its nearest neighbor?
        # A high value means the asset is 'floating' away from the cluster.
        isolation_scores = []
        
        for i, asset in enumerate(assets):
            # Find all edges connected to this asset in the MST
            # MST is undirected, check both rows and cols
            edges = np.concatenate((mst_array[i, :], mst_array[:, i]))
            edges = edges[edges > 0] # Filter zero entries
            
            if len(edges) > 0:
                # The 'Primary Tether' is the strongest link (shortest distance) to the cluster
                # The isolation score is the length of this tether.
                score = np.min(edges)
            else:
                score = 0 # Should not happen in connected graph
                
            isolation_scores.append({
                "ticker": asset,
                "isolation_score": float(score),
                "type": "Topological Outlier" if score > 0.65 else "Core Asset"
            })

        # 4. Sort by Alien Status
        aliens = sorted(isolation_scores, key=lambda x: x['isolation_score'], reverse=True)
        
        # 5. Compute Betti Numbers (Market Complexity) using Gudhi if available
        betti_1 = 0
        complexity_score = 50 # Default
        
        if gudhi:
            try:
                rips_complex = gudhi.RipsComplex(distance_matrix=dist_matrix, max_edge_length=2.0)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
                diag = simplex_tree.persistence()
                
                total_persistence = 0.0
                for dim, (birth, death) in diag:
                    if death == float('inf'): life = 2.0 - birth
                    else: life = death - birth
                    
                    if dim == 1 and life > 0.1:
                        betti_1 += 1
                        total_persistence += life
                
                complexity_score = min(100, total_persistence * 10)
            except Exception as e:
                LOGGER.warning(f"Gudhi calculation failed: {e}")

        # 6. Diagnosis
        if complexity_score < 20:
            diagnosis = "CRITICAL: TOPOLOGY COLLAPSE"
            description = "Market manifold has rigidified. Loops closed. Systemic fragility HIGH."
            status_color = "RED"
        elif complexity_score < 40:
            diagnosis = "WARNING: STRUCTURAL DECAY"
            description = "Market complexity decreasing. Correlations tightening."
            status_color = "YELLOW"
        else:
            diagnosis = "HEALTHY: COMPLEX MANIFOLD"
            description = "High dimensional structure intact. Sponge-like topology."
            status_color = "GREEN"

        return {
            "status": "ONLINE",
            "diagnosis": diagnosis,
            "description": description,
            "status_color": status_color,
            "metrics": {
                "betti_1_loops": betti_1,
                "complexity_score": float(complexity_score)
            },
            "aliens": aliens[:3] # Return top 3 aliens
        }
