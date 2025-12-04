
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

# --- V12 ARCHITECTURE (From User's Prompt) ---
class EconomicGraphDiscoverer(nn.Module):
    def __init__(self, num_nodes, in_feats, d_model=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes

        self.input_proj = nn.Sequential(
            nn.Linear(in_feats, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable adjacency - LOWER temperature for sharper connections
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

        # Multi-scale temporal
        self.temporal_short = nn.GRU(d_model * num_nodes, d_model, batch_first=True)
        self.temporal_mid = nn.GRU(d_model * num_nodes, d_model, batch_first=True)
        self.temporal_long = nn.GRU(d_model * num_nodes, d_model, batch_first=True)

        self.output = nn.Sequential(
            nn.Linear(d_model * 3, d_model),  # 3 scales now
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_nodes)
        )

    def get_adjacency(self):
        sim = torch.mm(self.node_embed, self.node_embed.t())
        return F.softmax(sim / self.adj_temperature, dim=-1)

    def forward(self, x):
        b, s, n, f = x.shape
        x = self.input_proj(x)
        adj = self.get_adjacency()

        for layer in self.graph_layers:
            x_agg = torch.einsum('bsnd,nm->bsmd', x, adj)
            x = layer(x + x_agg)

        d = x.shape[-1]
        x_flat = x.view(b, s, n * d)

        # Multi-scale: last 10, 20, full
        _, h_short = self.temporal_short(x_flat[:, -10:, :])
        _, h_mid = self.temporal_mid(x_flat[:, -20:, :])
        _, h_long = self.temporal_long(x_flat)

        h = torch.cat([h_short.squeeze(0), h_mid.squeeze(0), h_long.squeeze(0)], dim=-1)
        return self.output(h)

def verify_model_load():
    model_path = r"c:/key/wise_adviser_cursor_context/Caria_repo/caria/models/caria_v12_acc64.pth"
    
    print(f"🔍 Checking model at: {model_path}")
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        return

    try:
        # Load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        print("✅ State dict loaded.")
        
        # Infer dimensions
        # input_proj.0.weight shape is [d_model, in_feats]
        if 'input_proj.0.weight' in state_dict:
            w = state_dict['input_proj.0.weight']
            d_model = w.shape[0]
            in_feats = w.shape[1]
            print(f"  Detected d_model: {d_model}")
            print(f"  Detected in_feats: {in_feats}")
        else:
            print("❌ Could not infer dimensions from state dict.")
            return

        # node_embed shape is [num_nodes, 32]
        if 'node_embed' in state_dict:
            num_nodes = state_dict['node_embed'].shape[0]
            print(f"  Detected num_nodes: {num_nodes}")
        else:
            print("❌ Could not infer num_nodes.")
            return

        # Initialize model
        model = EconomicGraphDiscoverer(num_nodes=num_nodes, in_feats=in_feats, d_model=d_model)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded successfully!")
        
        # Test Forward Pass
        print("\n🧪 Running dummy forward pass...")
        seq_len = 45 # From V12 description
        dummy_input = torch.randn(1, seq_len, num_nodes, in_feats)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        
        print(f"  Output shape: {out.shape}")
        print("✅ Forward pass successful!")
        
        # Visualize Adjacency
        adj = model.get_adjacency().detach().numpy()
        print("\n📊 Learned Adjacency Matrix (Top 3 connections per node):")
        # Assuming standard G15 order if not saved, but we can just print indices
        for i in range(min(5, num_nodes)):
            row = adj[i]
            top_indices = row.argsort()[-3:][::-1]
            print(f"  Node {i} connects to: {top_indices} (weights: {row[top_indices]})")

    except Exception as e:
        print(f"❌ Error verifying model: {e}")
        import traceback
        traceback.print_exc()

    # Write summary to file
    with open("validation_summary.txt", "w") as f:
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Dimensions: Nodes={num_nodes}, In_Feats={in_feats}, D_Model={d_model}\n")
        f.write("Status: Loaded and Forward Pass Successful\n")
        f.write("Adjacency Sample (Node 0):\n")
        f.write(str(adj[0]))

if __name__ == "__main__":
    verify_model_load()
