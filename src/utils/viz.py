import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(model, x0, t_span, labels=None, save_path=None):
    """
    Plots the trajectories of the points from t=0 to t=1.
    
    Args:
        model: NeuralODE model
        x0: Initial state (batch, features)
        t_span: Time span to integrate over
        labels: Class labels for coloring
        save_path: Path to save the plot
    """
    model.eval()
    with torch.no_grad():
        # Integrate
        traj = model(x0, t_span) # (len(t), batch, features)
        traj = traj.cpu().numpy()
        
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Colors based on labels
    if labels is not None:
        labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
        colors = ['red' if l == 0 else 'blue' for l in labels]
    else:
        colors = 'blue'
        
    # Plot trajectories
    # traj shape: (time, batch, 2)
    for i in range(traj.shape[1]):
        ax.plot(traj[:, i, 0], traj[:, i, 1], c=colors[i], alpha=0.1)
        
    # Plot start and end points
    ax.scatter(traj[0, :, 0], traj[0, :, 1], c=colors, marker='o', label='Start (t=0)')
    ax.scatter(traj[-1, :, 0], traj[-1, :, 1], c=colors, marker='x', label='End (t=1)')
    
    ax.set_title("Trajectories x(t)")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_vector_field(model, t, x_range=[-3, 3], y_range=[-3, 3], n_grid=20, save_path=None):
    """
    Plots the vector field f(x, t) at a specific time t.
    
    Args:
        model: NeuralODE model
        t: Time t to evaluate f(x, t)
        x_range: Range for x-axis
        y_range: Range for y-axis
        n_grid: Number of grid points
        save_path: Path to save the plot
    """
    model.eval()
    
    # Create grid
    x = np.linspace(x_range[0], x_range[1], n_grid)
    y = np.linspace(y_range[0], y_range[1], n_grid)
    X, Y = np.meshgrid(x, y)
    
    # Flatten for batch processing
    grid_points = np.stack([X.flatten(), Y.flatten()], axis=1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(next(model.parameters()).device)
    
    # Time tensor
    t_tensor = torch.tensor(t, dtype=torch.float32).to(grid_tensor)
    
    with torch.no_grad():
        # Evaluate vector field: dx/dt = f(x, t)
        # NeuralODE.vf is the vector field module
        dxdt = model.vf(t_tensor, grid_tensor)
        dxdt = dxdt.cpu().numpy()
        
    U = dxdt[:, 0].reshape(X.shape)
    V = dxdt[:, 1].reshape(Y.shape)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(X, Y, U, V)
    ax.set_title(f"Vector Field f(x, t={t:.2f})")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
