import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import KNNImputer


# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def mae_np(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =========================
@dataclass
class Standardizer:
    means: Dict[str, float]
    stds: Dict[str, float]

    @staticmethod
    def fit(df: pd.DataFrame, cols: List[str]) -> "Standardizer":
        means, stds = {}, {}
        for c in cols:
            x = pd.to_numeric(df[c], errors="coerce")
            m = float(x.mean())
            s = float(x.std(ddof=0))
            if s == 0 or math.isnan(s):
                s = 1.0
            means[c], stds[c] = m, s
        return Standardizer(means, stds)

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = df.copy()
        for c in cols:
            x = pd.to_numeric(out[c], errors="coerce")
            out[c] = (x - self.means[c]) / self.stds[c]
        return out

    def inverse_transform_array(self, arr: np.ndarray, cols: List[str]) -> np.ndarray:
        arr2 = arr.copy()
        for j, c in enumerate(cols):
            arr2[..., j] = arr2[..., j] * self.stds[c] + self.means[c]
        return arr2


# =========================
class ResMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.proj(x)

class BipartiteImputer(nn.Module):
    def __init__(
        self,
        num_samples: int,
        num_features: int,
        embed_dim: int = 128,
        hidden: int = 256,
        num_layers: int = 2, 
        dropout: float = 0.2, 
    ):
        super().__init__()
        self.N = num_samples
        self.F = num_features
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        self.register_buffer("type_sample", torch.tensor([1.0, 0.0]))
        self.register_buffer("type_feature", torch.tensor([0.0, 1.0]))

        self.feature_emb = nn.Embedding(num_features, embed_dim)

        self.val_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.feat_fusion = nn.Linear(embed_dim * 2, embed_dim)

        self.sample_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        node_in_dim = embed_dim + 2
        edge_in_dim = node_in_dim * 2 + 1

        self.edge_mlps = nn.ModuleList([ResMLP(edge_in_dim, hidden, hidden, dropout) for _ in range(num_layers)])
        self.node_mlps = nn.ModuleList([ResMLP(node_in_dim + hidden, hidden, node_in_dim, dropout) for _ in range(num_layers)])
        self.edge_w_mlps = nn.ModuleList([ResMLP(edge_in_dim, hidden, 1, dropout) for _ in range(num_layers)])

        self.attn_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(edge_in_dim, hidden // 2),
                nn.Tanh(),
                nn.Linear(hidden // 2, 1, bias=False) 
            ) for _ in range(num_layers)
        ])

  
        imp_in_dim = node_in_dim * 2 + 1 
        
        self.imputer = nn.Sequential(
            nn.Linear(imp_in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1), 
        )

    def init_node_features(self, W: torch.Tensor, M: torch.Tensor):
        device = W.device
        f_idx = torch.arange(self.F, device=device)
        f_base = self.feature_emb(f_idx)
        
        W_filled = torch.nan_to_num(W, nan=0.0)
        val_enc = self.val_encoder(W_filled.unsqueeze(-1)) 
        f_base_expanded = f_base.unsqueeze(0).expand(self.N, -1, -1)
        
        combined = self.feat_fusion(torch.cat([val_enc, f_base_expanded], dim=-1))
        
        masked_combined = combined * M.unsqueeze(-1)
        sum_feats = masked_combined.sum(dim=1)
        counts = M.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_feats = sum_feats / counts
        
        s_base = self.sample_proj(mean_feats)
        s_feat = torch.cat([s_base, self.type_sample.expand(self.N, -1)], dim=1) 
        f_feat = torch.cat([f_base, self.type_feature.expand(self.F, -1)], dim=1) 
        return s_feat, f_feat

    def forward(self, W: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        
        s_feat, f_feat = self.init_node_features(W, M)
        
        w = torch.nan_to_num(W, nan=0.0).unsqueeze(-1)

        for l in range(self.num_layers):
            s_expand = s_feat.unsqueeze(1).expand(-1, self.F, -1)
            f_expand = f_feat.unsqueeze(0).expand(self.N, -1, -1)
            e_in = torch.cat([s_expand, f_expand, w], dim=-1)

            e_msg = self.edge_mlps[l](e_in)
            attn_raw = self.attn_proj[l](e_in).squeeze(-1)
            attn_raw = attn_raw.masked_fill(M == 0, -1e9)

            attn_s = F.softmax(attn_raw, dim=1).unsqueeze(-1)
            agg_s = (e_msg * attn_s).sum(dim=1)
            s_feat = self.node_mlps[l](torch.cat([s_feat, agg_s], dim=1))

            attn_f = F.softmax(attn_raw, dim=0).unsqueeze(-1)
            agg_f = (e_msg * attn_f).sum(dim=0)
            f_feat = self.node_mlps[l](torch.cat([f_feat, agg_f], dim=1))

            e_in2 = torch.cat([s_expand, f_expand, w], dim=-1)
            w_delta = self.edge_w_mlps[l](e_in2)
            w = w + w_delta 

        # --- Final Prediction with Residual Connection ---
        s_expand = s_feat.unsqueeze(1).expand(-1, self.F, -1)
        f_expand = f_feat.unsqueeze(0).expand(self.N, -1, -1)
        
      
        w_input_raw = torch.nan_to_num(W, nan=0.0).unsqueeze(-1)
        
        imp_in = torch.cat([s_expand, f_expand, w_input_raw], dim=-1) 
        
        
        correction = self.imputer(imp_in).squeeze(-1)
        
        
        W_hat = W + correction

        return W_hat


# =========================
@dataclass
class TrainConfig:
    lr: float = 5e-4      
    epochs: int = 3000
    patience: int = 150    
    weight_decay: float = 1e-5 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    embed_dim: int = 64
    hidden: int = 128
    num_layers: int = 2   
    dropout: float = 0.2   

def build_matrix(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    return X

def random_mask_traits(
    df: pd.DataFrame,
    trait_cols: List[str],
    missing_rate: float,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = df.copy()

    T = out[trait_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = rng.random(T.shape) < missing_rate 
    T_masked = T.copy()
    T_masked[mask] = np.nan

    out.loc[:, trait_cols] = T_masked
    return out, mask.astype(np.bool_)


# =========================
def train_imputer_kfold(
    df_masked: pd.DataFrame,
    feature_cols_all: List[str],
    cfg: TrainConfig,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    save_path: str,
) -> float:
    device = torch.device(cfg.device)
    N = df_masked.shape[0]
    F = len(feature_cols_all)

    X = build_matrix(df_masked, feature_cols_all)
    
    
    print("  [Init] Running KNN Imputer for warm start...")
    knn_imp = KNNImputer(n_neighbors=10)
    X_filled_knn = knn_imp.fit_transform(X)
    
    W_knn_t = torch.tensor(X_filled_knn, dtype=torch.float32, device=device)
    M_true = ~np.isnan(X)
    M_true_t = torch.tensor(M_true.astype(np.float32), device=device)

    model = BipartiteImputer(
        num_samples=N,
        num_features=F,
        embed_dim=cfg.embed_dim,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    loss_fn = nn.L1Loss(reduction="none") 

    best_val = float("inf")
    bad = 0
    mask_ratio = 0.2

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()

        # --- Mask Generation ---
        row_mask = torch.zeros(N, device=device, dtype=torch.bool)
        row_mask[train_idx] = True
        valid_points = (M_true_t > 0.5) & row_mask.unsqueeze(1)

        rand_prob = torch.rand_like(M_true_t)
        drop_mask = (rand_prob < mask_ratio) & valid_points
        keep_mask = valid_points & (~drop_mask)

        M_in = torch.clone(M_true_t)
        M_in[train_idx] = 0.0
        M_in[keep_mask] = 1.0
        
        
        W_in = W_knn_t.clone()
        
        
        W_hat = model(W_in, M_in)

        
        loss_mat = loss_fn(W_hat, W_knn_t)
        
        train_loss = (loss_mat * drop_mask.float()).sum() / (drop_mask.sum() + 1e-8)
        
        
        recon_loss = (loss_mat * keep_mask.float()).sum() / (keep_mask.sum() + 1e-8)
        
        total_loss = train_loss + 0.2 * recon_loss 

        total_loss.backward()
        opt.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            M_val_in = torch.clone(M_true_t)
            M_val_in[val_idx] = 0.0 
            
            W_hat_val = model(W_knn_t, M_val_in)
            
            val_loss_mat = loss_fn(W_hat_val, W_knn_t)
            
            val_row_mask = torch.zeros(N, device=device, dtype=torch.bool)
            val_row_mask[val_idx] = True
            val_target_mask = (M_true_t > 0.5) & val_row_mask.unsqueeze(1)
            
            val_loss = (val_loss_mat * val_target_mask.float()).sum() / (val_target_mask.sum() + 1e-8)

        val_f = float(val_loss.item())
        if val_f < best_val - 1e-6:
            best_val = val_f
            bad = 0
            torch.save(model.state_dict(), save_path)
        else:
            bad += 1

        if bad >= cfg.patience:
            break
            
    return best_val


# =========================
def eval_fold(
    df_full: pd.DataFrame, 
    df_masked: pd.DataFrame, 
    trait_cols: List[str],
    feature_cols_all: List[str],
    trait_missing_mask: np.ndarray, 
    model_ckpt: str,
    cfg: TrainConfig,
    standardizer: Standardizer,
    test_idx: np.ndarray,
) -> float:
    device = torch.device(cfg.device)
    N = df_masked.shape[0]
    F = len(feature_cols_all)

    X = build_matrix(df_masked, feature_cols_all)
    knn_imp = KNNImputer(n_neighbors=10)
    X_filled_knn = knn_imp.fit_transform(X)
    W_knn_t = torch.tensor(X_filled_knn, dtype=torch.float32, device=device)
    
    M_true = ~np.isnan(X)
    M_t = torch.tensor(M_true.astype(np.float32), device=device)
    
    M_in = M_t.clone()
    M_in[test_idx] = 0.0
    
    model = BipartiteImputer(
        num_samples=N,
        num_features=F,
        embed_dim=cfg.embed_dim,
        hidden=cfg.hidden,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout
    ).to(device)
    
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        W_hat = model(W_knn_t, M_in).detach().cpu().numpy()

    trait_idx = [feature_cols_all.index(c) for c in trait_cols]
    W_hat_traits = W_hat[:, trait_idx] 

    true_traits = build_matrix(df_full, trait_cols)
    pred_denorm = standardizer.inverse_transform_array(W_hat_traits, trait_cols)
    
    eval_mask = np.zeros_like(trait_missing_mask, dtype=bool)
    eval_mask[test_idx, :] = trait_missing_mask[test_idx, :]
    
    if np.sum(eval_mask) == 0:
        return 0.0

    mae = mae_np(pred_denorm[eval_mask], true_traits[eval_mask])
    return mae

def main():
    set_seed(42)
    
    data_path = "missing_data.csv"
    out_dir = "/result" 
    ensure_dir(out_dir)

    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df_raw = pd.read_csv(data_path)
    
   
    env_cols = [
        "T_max_mean", " ", "...", 
    ]
    trait_cols = [
        "Trait_1", "Trait_2", "Trait_n", "...",
    ]
    feature_cols = env_cols + trait_cols

    df_original = df_raw.copy()
    cfg = TrainConfig()

    missing_rates = [0.10, 0.20, 0.30]
    final_summary = []
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for r in missing_rates:
        print(f"\n>>> Running 5-Fold CV for Missing Rate = {r*100}% (Method: Residual Learning)")
        tag = f"miss_{int(r*100)}"
        run_dir = os.path.join(out_dir, tag)
        ensure_dir(run_dir)

        df_masked_raw, trait_mask = random_mask_traits(df_original, trait_cols, missing_rate=r, seed=42)
        fold_maes = []
        
        for fold_idx, (train_full_idx, test_idx) in enumerate(kf.split(df_masked_raw)):
            fold_dir = os.path.join(run_dir, f"fold_{fold_idx}")
            ensure_dir(fold_dir)
            
           
            train_data_raw = df_masked_raw.iloc[train_full_idx]
            fold_standardizer = Standardizer.fit(train_data_raw, feature_cols)
            df_masked_std = fold_standardizer.transform(df_masked_raw, feature_cols)
            
            tr_idx, va_idx = train_test_split(train_full_idx, test_size=0.2, random_state=42)
            ckpt_path = os.path.join(fold_dir, "ckpt.pt")
            
            best_val_loss = train_imputer_kfold(
                df_masked=df_masked_std,
                feature_cols_all=feature_cols,
                cfg=cfg,
                train_idx=tr_idx,
                val_idx=va_idx,
                save_path=ckpt_path
            )
            
            fold_mae = eval_fold(
                df_full=df_original, 
                df_masked=df_masked_std, 
                trait_cols=trait_cols,
                feature_cols_all=feature_cols,
                trait_missing_mask=trait_mask,
                model_ckpt=ckpt_path,
                cfg=cfg,
                standardizer=fold_standardizer, 
                test_idx=test_idx
            )
            
            print(f"  [Fold {fold_idx}] Impute MAE: {fold_mae:.6f} (Best Val: {best_val_loss:.6f})")
            fold_maes.append(fold_mae)

        mean_mae = np.mean(fold_maes)
        std_mae = np.std(fold_maes, ddof=1)
        stderr_mae = std_mae / np.sqrt(5)

        print(f"[RESULT] Missing Rate {r:.2f}: Mean MAE = {mean_mae:.6f} ± {stderr_mae:.6f}")
        final_summary.append({"missing_rate": r, "mae_mean": mean_mae, "mae_stderr": stderr_mae})

    res_df = pd.DataFrame(final_summary)
    res_path = os.path.join(out_dir, "imputation_cv_summary.csv")
    res_df.to_csv(res_path, index=False)
    print(f"\n[DONE] Saved to: {res_path}")

if __name__ == "__main__":
    main()
