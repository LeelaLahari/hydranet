"""Regional LSTM with basin embeddings, quantile heads, MC-dropout inference."""
import torch
import torch.nn as nn

class RegionalLSTM(nn.Module):
    def __init__(self, n_dyn, n_static, n_basins, emb_dim=8, hidden=128,
                 layers=1, dropout=0.25, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.emb = nn.Embedding(n_basins, emb_dim)
        self.lstm = nn.LSTM(n_dyn + n_static + emb_dim, hidden, layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, len(quantiles))
        self.quantiles = quantiles

    def forward(self, x_dyn, x_static, basin_id, mc_dropout=False):
        if mc_dropout:
            self.train()  # keep dropout ON at inference (MC-dropout UQ)
        e = self.emb(basin_id)
        B, T, _ = x_dyn.shape
        s = torch.cat([x_static, e], dim=-1).unsqueeze(1).expand(B, T, -1)
        out, _ = self.lstm(torch.cat([x_dyn, s], dim=-1))
        return self.head(self.drop(out[:, -1]))  # (B, n_quantiles)

def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles) -> torch.Tensor:
    """pred (B,Q), target (B,). Mean pinball over quantiles."""
    target = target.unsqueeze(1).expand_as(pred)
    errs = target - pred
    qs = torch.tensor(quantiles, device=pred.device).view(1, -1)
    return torch.maximum(qs * errs, (qs - 1) * errs).mean()

@torch.no_grad()
def mc_predict(model, x_dyn, x_static, basin_id, n_samples=30):
    """Return (median, epistemic std) in normalized space."""
    model.eval()
    preds = torch.stack([model(x_dyn, x_static, basin_id, mc_dropout=True)
                         for _ in range(n_samples)])  # (S,B,Q)
    q50 = preds[..., 1] if preds.shape[-1] > 1 else preds.squeeze(-1)
    return q50.mean(0), q50.std(0), preds
