# ====== SECTION 1: Imports ======
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import time

# ====== SECTION 2: Load Data ======
def load_data(file_path, max_samples=None):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    samples = []
    
    # Limit samples if specified
    if max_samples:
        lines = lines[:max_samples * 3]
    
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
        
        y = torch.tensor(output, dtype=torch.float).view(1, -1)  # Shape: [1, 7]
        
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
        
        # Create batch indices for pooling
        # For each polygon, we need to assign nodes to their respective graphs in the batch
        batch_size = data.y.shape[0]
        poly1_batch = []
        poly2_batch = []
        
        # This is a bit complex because we need to handle the batched data correctly
        # We'll iterate through each graph in the batch
        for i in range(batch_size):
            # Get the number of nodes for each polygon in this graph
            n1 = torch.sum(data.graph_id[data.ptr[i]:data.ptr[i+1]] == 0).item()
            n2 = torch.sum(data.graph_id[data.ptr[i]:data.ptr[i+1]] == 1).item()
            
            # Add batch indices for poly1 nodes
            poly1_batch.extend([i] * n1)
            # Add batch indices for poly2 nodes
            poly2_batch.extend([i] * n2)
        
        poly1_batch = torch.tensor(poly1_batch, dtype=torch.long, device=x.device)
        poly2_batch = torch.tensor(poly2_batch, dtype=torch.long, device=x.device)
        
        # Pool each polygon into a vector
        poly1_embed = global_mean_pool(x[poly1_mask], poly1_batch)
        poly2_embed = global_mean_pool(x[poly2_mask], poly2_batch)
        
        # Combine embeddings
        combined = torch.cat([poly1_embed, poly2_embed], dim=1)
        
        # Final prediction
        out = self.fc(combined)
        return out

# ====== SECTION 4: Training Setup ======
def main():
    # Load a subset of data for testing (use full dataset later)
    print("Loading data...")
    dataset = load_data("gnn_50_dataset.txt", max_samples=1000000)
    print(f"Loaded {len(dataset)} samples")
    
    # Split into train/validation
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = PolyGNN(hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # ====== WEIGHTED LOSS FUNCTION ======
    # Create weights for different output parameters
    # Higher weights for containment tasks to address class imbalance
    loss_weights = torch.tensor([1.0,  # Area Ratio
                                 1.0,  # Perimeter Ratio
                                 1.0,  # Intersection
                                 50.0, # Contains1In2 (high weight for rare class)
                                 50.0, # Contains2In1 (high weight for rare class)
                                 1.0,  # Horizontal Position
                                 1.0   # Vertical Position
                                ]).to(device)
    
    def weighted_mse_loss(pred, target):
        # Calculate MSE for each sample and output parameter
        mse_per_param = (pred - target) ** 2
        # Apply weights to each parameter
        weighted_mse = mse_per_param * loss_weights
        # Return average over all samples and parameters
        return weighted_mse.mean()
    
    criterion = weighted_mse_loss
    
    # ====== DATA LOADERS ======
    batch_size = 32  # Reduced for testing
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    # Training history tracking
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    # Training loop
    num_epochs = 50  # Reduced for testing
    print("Starting training...")
    
    for epoch in range(num_epochs):
        # ==== TRAINING PHASE ====
        model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_count % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_count}: Loss = {loss.item():.6f}")
        
        avg_train_loss = total_loss / batch_count
        train_loss_history.append(avg_train_loss)
        
        # ==== VALIDATION PHASE ====
        model.eval()
        val_loss = 0
        val_batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred, batch.y)
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count
        val_loss_history.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "polygon_gnn_best.pth")
            print(f"Epoch {epoch+1}: New best model saved with val loss {best_val_loss:.6f}")
        
        # Print training progress
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
    
    # Save final model
    torch.save(model.state_dict(), "polygon_gnn_final.pth")
    print("Training completed. Final model saved as polygon_gnn_final.pth")
    
    # Plot training history (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history, label='Training Loss')
        plt.plot(val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Weighted MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.savefig('training_history.png')
        plt.close()
        print("Training history plot saved as training_history.png")
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")

# ====== RUN THE CODE ======
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total training time: {(end_time - start_time)/60:.2f} minutes")