import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.vector_fields import VectorField
from models.neural_odes import NeuralODE
from utils.viz import plot_trajectories, plot_vector_field

def train():
    print("Generating data...")
    # Generate make_moons data
    n_samples = 1000 # Smaller for quick test
    noise = 0.05
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

    # Convert to tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # Define target means
    target_means = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])

    # Generate targets with some variance
    target_noise = 0.05
    targets = target_means[y_tensor] + torch.randn_like(X_tensor) * target_noise

    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = 2
    hidden_dims = [64, 64]
    time_embed_dim = 16

    vf = VectorField(features=features, hidden_dims=hidden_dims, time_embed_dim=time_embed_dim)
    model = NeuralODE(vector_field=vf, solver='rk4').to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10 # Short run for verification
    batch_size = 256
    t_span = torch.tensor([0., 1.]).to(device)

    dataset = torch.utils.data.TensorDataset(X_tensor, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_target in dataloader:
            batch_x = batch_x.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            out = model(batch_x, t_span)
            x_1 = out[-1]
            
            # MSE Loss
            loss = nn.MSELoss()(x_1, batch_target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    print("Training finished successfully.")
    
    # Test visualization (save to file instead of show)
    print("Testing visualization...")
    subset_indices = np.random.choice(len(X), 100, replace=False)
    x0_subset = X_tensor[subset_indices].to(device)
    y_subset = y_tensor[subset_indices]
    t_eval = torch.linspace(0, 1, 20).to(device)
    
    # We modify viz functions to save if path provided, which I added.
    # But let's just run them to ensure no errors.
    # I need to make sure viz.py handles non-interactive backends if needed, 
    # but standard plt usually works fine or just warns.
    
    try:
        plot_trajectories(model, x0_subset, t_eval, labels=y_subset, save_path='results/test_traj.png')
        plot_vector_field(model, t=0.5, save_path='results/test_vf.png')
        print("Visualization saved to results/")
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train()
