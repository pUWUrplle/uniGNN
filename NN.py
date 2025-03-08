# ====== SECTION 1: Imports ======
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np


# ====== SECTION 2: Load Data ======
def load_data(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    samples = []
    
    for i in range(0, len(lines), 3):
        # --- Parse Polygon 1 ---
        polygon1 = lines[i].split()
        n1 = int(polygon1[0])
        coords1 = list(map(float, polygon1[1:]))
        node_features1 = np.array(coords1).reshape(n1, 2)  # Shape: [n1, 2]
        
        # --- Parse Polygon 2 ---
        polygon2 = lines[i+1].split()
        n2 = int(polygon2[0])
        coords2 = list(map(float, polygon2[1:]))
        node_features2 = np.array(coords2).reshape(n2, 2)  # Shape: [n2, 2]
        
        # --- Output Parameters ---
        output = list(map(float, lines[i+2].split()))
        
        # ==== Build the Graph ====
        # Combine nodes from both polygons
        x = torch.tensor(np.vstack([node_features1, node_features2]), dtype=torch.float)
        
        # Build edges for Polygon 1 (cyclic connections)
        edge_index1 = []
        for j in range(n1):
            # Forward edge: j -> (j+1) % n1
            edge_index1.append([j, (j+1) % n1])
            # Backward edge: (j+1) % n1 -> j
            edge_index1.append([(j+1) % n1, j])
        
        # Build edges for Polygon 2 (offset by n1)
        edge_index2 = []
        for j in range(n2):
            src = n1 + j
            dst = n1 + (j+1) % n2
            edge_index2.append([src, dst])
            edge_index2.append([dst, src])
        
        edge_index = torch.tensor(edge_index1 + edge_index2, dtype=torch.long).t().contiguous()
        
        # Create graph identifiers
        graph_id = torch.cat([
            torch.zeros(n1, dtype=torch.long),  # 0 for Polygon 1
            torch.ones(n2, dtype=torch.long)    # 1 for Polygon 2
        ])
        
        y = torch.tensor(output, dtype=torch.float).view(1, -1)  # Shape: [3, 7]
        
        data = Data(x=x, edge_index=edge_index, y=y, graph_id=graph_id)
        samples.append(data)
    
    return samples

# ====== SECTION 3: Define Model ======
class PolyGNN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(2, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc = torch.nn.Linear(2 * hidden_dim, 7)
    
    def forward(self, data):
        x, edge_index, graph_id = data.x, data.edge_index, data.graph_id
        
        # GNN layers
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        x = self.conv2(x, edge_index).relu()
        
        # Separate polygons using graph_id
        poly1_mask = (graph_id == 0)
        poly2_mask = (graph_id == 1)
        
        # Pool each polygon into a vector
        poly1_embed = global_mean_pool(x[poly1_mask], batch=data.batch[poly1_mask])
        poly2_embed = global_mean_pool(x[poly2_mask], batch=data.batch[poly2_mask])
        
        # Combine embeddings
        combined = torch.cat([poly1_embed, poly2_embed], dim=1)
        
        # Final prediction
        out = self.fc(combined)
        return out

# ====== SECTION 4: Training Setup ======
def main():
    # Load data
    dataset = load_data("neural_network_dataset.txt")
    print(f"Loaded {len(dataset)} samples")
    
    # Split into train/validation
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolyGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # ====== OPTIMAL SETTINGS FOR RTX 3060 ======
    batch_size = 128  # Reduced from 256 to prevent VRAM overload
    num_workers = 8  # For DataLoader parallelism
    torch.backends.cudnn.benchmark = True  # Optimizes CUDA operations
    
    # Data loaders with pinned memory
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=True)

    # ====== TRAINING LOOP WITH PROGRESS ======
def main():
    # Load data
    dataset = load_data("neural_network_dataset.txt")
    print(f"Loaded {len(dataset)} samples")
    
    # Split into train/validation
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolyGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Data loaders
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    
    # Training loop
    for epoch in range(50):  # <-- Epoch loop starts here
        # ==== TRAINING PHASE ====
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        # Print progress

        # ==== VALIDATION PHASE ====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred, batch.y)
                val_loss += loss.item()

        # Print training progress
        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss/len(val_loader):.4f}")

    #save model    
    torch.save(model.state_dict(), "polygon_gnn.pth")
    print("Model saved as polygon_gnn.pth")  

# ====== RUN THE CODE ======
if __name__ == "__main__":
    main()