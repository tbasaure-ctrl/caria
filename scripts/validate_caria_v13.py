
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- MODEL DEFINITION (From V13 Notebook) ---
class EconomicGraphDiscoverer(nn.Module):
    def __init__(self, num_nodes, in_feats, d_model=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        
        self.input_proj = nn.Sequential(
            nn.Linear(in_feats, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.node_embed = nn.Parameter(torch.randn(num_nodes, 32) * 0.1)
        self.adj_temperature = 0.3

        self.graph_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])

        self.temporal = nn.GRU(d_model * num_nodes, d_model * 2, batch_first=True, num_layers=2, dropout=dropout)

        self.output = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_nodes)
        )

    def get_adjacency(self):
        sim = torch.mm(self.node_embed, self.node_embed.t())
        return F.softmax(sim / self.adj_temperature, dim=-1)

    def forward(self, x):
        b, s, n, f = x.shape
        x = self.input_proj(x)
        adj = self.get_adjacency()

        x_flat = x.view(b * s, n, -1)
        
        for layer in self.graph_layers:
            x_agg = torch.bmm(adj.unsqueeze(0).expand(b*s, n, n), x_flat)
            x_flat = layer(x_flat + x_agg)
        
        x = x_flat.view(b, s, n, -1)
        x_temporal = x.reshape(b, s, -1)
        
        _, h = self.temporal(x_temporal)
        h = h[-1]
        
        return self.output(h)

# --- VALIDATION SCRIPT ---
def validate_v13_architecture():
    print("🚀 Validating Caria V13 Model Architecture...")
    
    # 1. Setup Mock Data
    NODES = 27
    FEATS = 15 # Market(3) + Macro(6) + Global(5) + FX(1)
    SEQ_LEN = 45
    BATCH_SIZE = 8
    
    print(f"  Nodes: {NODES}")
    print(f"  Features: {FEATS}")
    print(f"  Sequence Length: {SEQ_LEN}")
    
    # Random Tensor: [Batch, Seq, Nodes, Feats]
    x = torch.randn(BATCH_SIZE, SEQ_LEN, NODES, FEATS)
    # Random Labels: [Batch, Nodes] (Binary)
    y = torch.randint(0, 2, (BATCH_SIZE, NODES)).float()
    
    # 2. Initialize Model
    try:
        model = EconomicGraphDiscoverer(num_nodes=NODES, in_feats=FEATS)
        print("  ✅ Model initialized successfully.")
    except Exception as e:
        print(f"  ❌ Model initialization failed: {e}")
        return

    # 3. Forward Pass Check
    try:
        out = model(x)
        print(f"  ✅ Forward pass successful. Output shape: {out.shape} (Expected: [{BATCH_SIZE}, {NODES}])")
        assert out.shape == (BATCH_SIZE, NODES)
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return

    # 4. Adjacency Matrix Check
    try:
        adj = model.get_adjacency()
        print(f"  ✅ Adjacency matrix shape: {adj.shape}")
        
        # Check properties
        row_sums = adj.sum(dim=1)
        if torch.allclose(row_sums, torch.ones(NODES), atol=1e-5):
            print("  ✅ Adjacency matrix is valid (Rows sum to 1).")
        else:
            print(f"  ⚠️ Adjacency matrix rows do not sum to 1: {row_sums[:3]}")
            
    except Exception as e:
        print(f"  ❌ Adjacency check failed: {e}")

    # 5. Backward Pass (Training Step) Check
    try:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        print(f"  ✅ Backward pass (training step) successful. Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"  ❌ Backward pass failed: {e}")
        return

    print("\n🎉 CARIA V13 ARCHITECTURE IS VALID!")
    print("You can confidently run this code in Colab with the manual data.")

if __name__ == "__main__":
    validate_v13_architecture()
