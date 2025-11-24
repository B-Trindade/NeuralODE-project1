import torch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from models.vector_fields import VectorField
from models.neural_odes import NeuralODE

def test_neural_ode_forward():
    print("Testing NeuralODE forward pass...")
    
    # Dimensions
    batch_size = 4
    features = 2
    time_embed_dim = 8
    
    # Create models
    vf = VectorField(features=features, time_embed_dim=time_embed_dim)
    model = NeuralODE(vector_field=vf, solver='rk4') # Using rk4 for simple test
    
    # Dummy input
    x0 = torch.randn(batch_size, features)
    t_span = torch.linspace(0, 1, 10)
    
    # Forward pass
    try:
        out = model(x0, t_span)
        print(f"Output shape: {out.shape}")
        
        # Expected shape: (len(t_span), batch_size, features)
        expected_shape = (len(t_span), batch_size, features)
        
        if out.shape == expected_shape:
            print("SUCCESS: Output shape matches expected shape.")
        else:
            print(f"FAILURE: Expected shape {expected_shape}, got {out.shape}")
            
    except Exception as e:
        print(f"FAILURE: Exception during forward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_neural_ode_forward()
