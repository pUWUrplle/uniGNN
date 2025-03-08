# predict.py
import torch
from torch_geometric.data import Data
from NN import PolyGNN  # Replace with your actual script name import PolyGNN  # Replace with your actual script name

# ====== Helper Function: Read Polygon from Console ======
def read_polygon_from_console(polygon_number):
    print(f"\n=== Polygon {polygon_number} ===")
    n = int(input("Number of vertices (3-6): "))
    coords = []
    for i in range(n):
        x = float(input(f"Vertex {i+1} X: "))
        y = float(input(f"Vertex {i+1} Y: "))
        coords.append([x, y])
    return coords

# ====== Load Model ======
def load_model(model_path="polygon_gnn.pth"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PolyGNN(hidden_dim=64).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, device

def forward(self, data):
    x, edge_index, graph_id = data.x, data.edge_index, data.graph_id
    
    # GNN layers
    x = self.conv1(x, edge_index).relu()
    x = self.dropout(x)
    x = self.conv2(x, edge_index).relu()
    
    # Separate polygons using graph_id
    poly1_mask = (graph_id == 0)
    poly2_mask = (graph_id == 1)
    
    # Pool each polygon (use batch for proper grouping)
    poly1_embed = global_mean_pool(x[poly1_mask], batch=data.batch[poly1_mask])
    poly2_embed = global_mean_pool(x[poly2_mask], batch=data.batch[poly2_mask])
    
    # Combine embeddings
    combined = torch.cat([poly1_embed, poly2_embed], dim=1)
    
    # Final prediction
    out = self.fc(combined)
    return out

# ====== Prediction Function ======
def predict_polygons(model, device, poly1_coords, poly2_coords):
    n1 = len(poly1_coords)
    n2 = len(poly2_coords)
    
    # Node features
    x = torch.tensor(poly1_coords + poly2_coords, dtype=torch.float).to(device)
    
    # Build edges
    edge_index1 = []
    for j in range(n1):
        edge_index1.extend([[j, (j+1) % n1], [(j+1) % n1, j]])
    
    edge_index2 = []
    for j in range(n2):
        src = n1 + j
        dst = n1 + (j+1) % n2
        edge_index2.extend([[src, dst], [dst, src]])
    
    edge_index = torch.tensor(edge_index1 + edge_index2, dtype=torch.long).t().contiguous().to(device)
    
    # Create graph_id (0 for poly1, 1 for poly2)
    graph_id = torch.cat([
        torch.zeros(n1, dtype=torch.long),
        torch.ones(n2, dtype=torch.long)
    ]).to(device)
    
    # Create Data object with ALL required attributes
    data = Data(x=x, edge_index=edge_index, graph_id=graph_id, batch=torch.zeros(x.size(0), dtype=torch.long)).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(data)
    return pred.cpu().numpy()

# ====== Main Execution ======
if __name__ == "__main__":
    # Load model
    model, device = load_model()
    
    # Get input from console
    print("Enter coordinates for two polygons:")
    poly1 = read_polygon_from_console(1)
    poly2 = read_polygon_from_console(2)
    
    # Predict
    prediction = predict_polygons(model, device, poly1, poly2)
    print("\nPredicted parameters:", prediction.round(4))