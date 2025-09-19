# predict.py
import torch
import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, r2_score
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

# ====== Performance Measurement Functions ======
def load_test_data(file_path, max_samples=None):
    """Load test data from dataset file"""
    samples = []
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    sample_count = min(len(lines) // 3, max_samples) if max_samples else len(lines) // 3
    
    for i in range(0, sample_count * 3, 3):
        # Parse Polygon 1
        poly1 = lines[i].split()
        n1 = int(poly1[0])
        coords1 = list(map(float, poly1[1:]))
        poly1_coords = np.array(coords1).reshape(n1, 2).tolist()
        
        # Parse Polygon 2
        poly2 = lines[i+1].split()
        n2 = int(poly2[0])
        coords2 = list(map(float, poly2[1:]))
        poly2_coords = np.array(coords2).reshape(n2, 2).tolist()
        
        # Parse expected output
        expected = list(map(float, lines[i+2].split()))
        
        samples.append({
            'poly1': poly1_coords,
            'poly2': poly2_coords,
            'expected': expected
        })
    
    return samples

def evaluate_model_performance(model, device, test_samples, output_file="performance_report.txt"):
    """Evaluate model performance on test samples and write results to file"""
    results = {
        'total_time': 0.0,
        'per_sample_time': [],
        'metrics': {
            'area_ratio': {'true': [], 'pred': [], 'abs_error': []},
            'perim_ratio': {'true': [], 'pred': [], 'abs_error': []},
            'intersection': {'true': [], 'pred': [], 'thresholded': []},
            'contains1in2': {'true': [], 'pred': [], 'thresholded': []},
            'contains2in1': {'true': [], 'pred': [], 'thresholded': []},
            'horiz_pos': {'true': [], 'pred': [], 'abs_error': []},
            'vert_pos': {'true': [], 'pred': [], 'abs_error': []}
        }
    }
    
    print(f"Evaluating on {len(test_samples)} samples...")
    
    for i, sample in enumerate(test_samples):
        start_time = time.perf_counter()
        
        # Get prediction
        pred = predict_polygons(model, device, sample['poly1'], sample['poly2'])[0]
        
        # Record time
        inference_time = time.perf_counter() - start_time
        results['total_time'] += inference_time
        results['per_sample_time'].append(inference_time)
        
        # Store results
        true_vals = sample['expected']
        
        # Continuous outputs
        for idx, key in enumerate(['area_ratio', 'perim_ratio']):
            results['metrics'][key]['true'].append(true_vals[idx])
            results['metrics'][key]['pred'].append(pred[idx])
            results['metrics'][key]['abs_error'].append(abs(true_vals[idx] - pred[idx]))
        
        # Boolean outputs
        for j, key in enumerate(['intersection', 'contains1in2', 'contains2in1'], start=2):
            results['metrics'][key]['true'].append(true_vals[j])
            results['metrics'][key]['pred'].append(pred[j])
            results['metrics'][key]['thresholded'].append(1 if pred[j] > 0.5 else 0)
        
        # Positional outputs
        for idx, key in enumerate(['horiz_pos', 'vert_pos'], start=5):
            results['metrics'][key]['true'].append(true_vals[idx])
            results['metrics'][key]['pred'].append(pred[idx])
            results['metrics'][key]['abs_error'].append(abs(true_vals[idx] - pred[idx]))
    
    # Calculate metrics
    metrics_report = calculate_metrics(results)
    
    # Write to file with UTF-8 encoding to handle special characters
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("===== GNN Performance Report =====\n\n")
        f.write(f"Evaluation on {len(test_samples)} samples\n")
        f.write(f"Total inference time: {results['total_time']:.4f} seconds\n")
        f.write(f"Average per sample: {np.mean(results['per_sample_time'])*1000:.4f} ms\n")
        f.write(f"Max per sample: {np.max(results['per_sample_time'])*1000:.4f} ms\n")
        f.write(f"Min per sample: {np.min(results['per_sample_time'])*1000:.4f} ms\n\n")
        
        f.write("===== Continuous Output Metrics =====\n")
        for key in ['area_ratio', 'perim_ratio', 'horiz_pos', 'vert_pos']:
            f.write(f"\n{key.replace('_', ' ').title()}:\n")
            f.write(f"  MAE: {metrics_report[key]['mae']:.6f}\n")
            f.write(f"  Max Error: {metrics_report[key]['max_error']:.6f}\n")
            f.write(f"  Median Error: {metrics_report[key]['median_error']:.6f}\n")
            f.write(f"  Std Deviation: {metrics_report[key]['std_dev']:.6f}\n")
            f.write(f"  R² Score: {metrics_report[key]['r_squared']:.6f}\n")
        
        f.write("\n===== Boolean Output Metrics =====\n")
        for key in ['intersection', 'contains1in2', 'contains2in1']:
            f.write(f"\n{key.replace('_', ' ').title()}:\n")
            f.write(f"  Accuracy: {metrics_report[key]['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics_report[key]['precision']:.4f}\n")
            f.write(f"  Recall: {metrics_report[key]['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics_report[key]['f1']:.4f}\n")
            f.write(f"  Confusion Matrix: {metrics_report[key]['confusion_matrix']}\n")
    
    print(f"Performance report saved to {output_file}")
    return metrics_report

def calculate_metrics(results):
    """Compute comprehensive evaluation statistics"""
    metrics_report = {}
    
    # Continuous output analysis
    for key in ['area_ratio', 'perim_ratio', 'horiz_pos', 'vert_pos']:
        abs_errors = results['metrics'][key]['abs_error']
        true_vals = results['metrics'][key]['true']
        pred_vals = results['metrics'][key]['pred']
        
        metrics_report[key] = {
            'mae': np.mean(abs_errors),
            'max_error': np.max(abs_errors),
            'median_error': np.median(abs_errors),
            'std_dev': np.std(abs_errors),
            'r_squared': r2_score(true_vals, pred_vals)
        }
    
    # Boolean output analysis
    for key in ['intersection', 'contains1in2', 'contains2in1']:
        true = results['metrics'][key]['true']
        pred = results['metrics'][key]['thresholded']
        
        metrics_report[key] = {
            'accuracy': accuracy_score(true, pred),
            'precision': precision_score(true, pred, zero_division=0),
            'recall': recall_score(true, pred, zero_division=0),
            'f1': f1_score(true, pred, zero_division=0),
            'confusion_matrix': confusion_matrix(true, pred).tolist()
        }
    
    return metrics_report

# ====== Main Execution ======
if __name__ == "__main__":
    # Load model
    model, device = load_model()
    
    # Mode selection
    print("Select mode:")
    print("1. Interactive prediction")
    print("2. Performance evaluation on dataset")
    mode = input("Enter choice (1 or 2): ")
    
    if mode == "1":
        # Get input from console
        print("Enter coordinates for two polygons:")
        poly1 = read_polygon_from_console(1)
        poly2 = read_polygon_from_console(2)
        
        # Predict
        prediction = predict_polygons(model, device, poly1, poly2)
        print("\nPredicted parameters:", prediction.round(4))
        
    elif mode == "2":
        # Performance evaluation mode
        dataset_path = input("Enter dataset path: ")
        max_samples = int(input("How many test cases to evaluate? "))
        output_file = input("Enter output file name (default: performance_report.txt): ") or "performance_report.txt"
        
        # Load test data
        test_samples = load_test_data(dataset_path, max_samples)
        
        # Evaluate performance
        metrics = evaluate_model_performance(model, device, test_samples, output_file)
        
        # Print summary
        print("\n===== Performance Summary =====")
        print(f"Evaluation completed on {len(test_samples)} samples")
        print(f"Results saved to {output_file}")
        
    else:
        print("Invalid mode selected")