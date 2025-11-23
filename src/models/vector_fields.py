import torch
import torch.nn as nn

class VectorField(nn.Module):
    """
    Parametriza dx/dt = f(x, t) usando rede neural.
    Args:
    features: dimensão dos dados
    hidden_dims: lista com dimensões das camadas ocultas
    time_embed_dim: dimensão do embedding temporal
    """

    def __init__(self, features, hidden_dims=[64, 64], time_embed_dim=16):
        super().__init__()
        self.features = features
        self.time_embed_dim = time_embed_dim

        # Dicas:
        # 1. Time embedding: usar sinusoidal encoding
        # 2. Network: MLP simples ou com skip connections
        # 3. Inicialização: última camada com pesos pequenos (σ=0.01)

        layers = []
        input_dim = features + time_embed_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.Tanh()) # Garantia de suavidade
            input_dim = h_dim

        final_layer = nn.Linear(input_dim, features)
        layers.append(final_layer)
        self.network = nn.Sequential(*layers) # Dica 2

        # Dica 3: inicializacao da ultima camada com pesos pequenos
        final_layer.weight.data.normal_(mean=0.0, std=0.01)
        final_layer.bias.data.fill_(0)

    def time_embedding(self, t):
        """
        Sinusoidal time embedding.
        Args:
        t: (batch,) ou escalar
        Returns:
        embedded: (batch, time_embed_dim)
        """
        # t_emb[2i] = sin(t / 10000^(2i/d))
        # t_emb[2i+1] = cos(t / 10000^(2i/d))

        half_dim = self.time_embed_dim // 2 
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)

        # t[:, None] shape: (batch, 1)
        # emb[None, :] shape: (1, half_dim)
        emb = t[:, None] * emb[None, :] # (batch x half_dim)

        # Concatenar sin e cos para obter time_embed_dim
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        # Caso dim % 2 == 1 adicionar padding
        if self.time_embed_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, t, x):
        """
        Calcula f(x, t).
        Args:
            t: tempo (escalar ou (batch,))
            x: estado (batch, features)
        Returns:
            dx_dt: (batch, features)
        """
    
        # 1. Expandir t para batch se necessário
        # 2. Time embedding
        # 3. Concatenar [x, t_emb]
        # 4. Passar pela rede

        # Dica 1: expandir t
        batch_size = x.shape[0]
        if (t.dim() == 0) or (t.shape[0] != batch_size):
            t = t.expand(batch_size)
        
        # Dica 2: time embedding
        t_emb = self.time_embedding(t)

        # Dica 3: Concatenar [x:(batch, features), t_emb:(batch, time_embed_dim)]
        x_input = torch.cat([x, t_emb], dim=1)

        # Dica 4: Passar pela rede
        dx_dt = self.network(x_input)
        return dx_dt