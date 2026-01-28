import torch
import pandas as pd
import numpy as np
import argparse
import math
from pathlib import Path
import torch.nn as nn
import os

# --- LSTM MODEL MİMARİSİ (Eğitimdeki ile aynı) ---
class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim=2, hid=64, layers=2, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hid, 2 * horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.head(out[:, -1, :])
        return out.view(-1, self.horizon, 2)

def get_direction_label(angle_deg):
    """Açıyı (derece) metinsel yöne çevirir."""
    # 0 derece sağ, 90 aşağı, 180 sol, -90 yukarı
    if -45 <= angle_deg <= 45: return "SAG (RIGHT)"
    elif 45 < angle_deg <= 135: return "ASAGI (DOWN)"
    elif -135 <= angle_deg < -45: return "YUKARI (UP)"
    else: return "SOL (LEFT)"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, help="Input track CSV")
    ap.add_argument("--model", required=True, help="Model .pt path")
    ap.add_argument("--out", required=True, default="outputs/forecast_aug.csv")
    ap.add_argument("--hist", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    # Klasör kontrolü
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Modeli ve Parametreleri Yükle
    print(f"[Info] Loading model from {args.model}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {args.model}")

    # --- KRİTİK DÜZELTME BURADA YAPILDI ---
    # weights_only=False ekleyerek PyTorch 2.6+ hatasını çözüyoruz
    try:
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    except TypeError:
        # Eğer eski PyTorch sürümü varsa weights_only parametresini desteklemez
        checkpoint = torch.load(args.model, map_location=device)
    
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    mn = checkpoint.get('mn', None)
    scale = checkpoint.get('scale', None)
    
    model = LSTMSeq2Seq(in_dim=2, hid=64, layers=2, horizon=args.horizon).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Veriyi Hazırla
    if not os.path.exists(args.track):
        raise FileNotFoundError(f"Track dosyası bulunamadı: {args.track}")
        
    df = pd.read_csv(args.track)
    if 'cx_smooth' in df.columns:
        coords = df[['cx_smooth', 'cy_smooth']].ffill().bfill().values
    else:
        coords = df[['cx', 'cy']].ffill().bfill().values

    results = []

    print("[Info] Running per-frame forecasting...")
    for i in range(len(coords)):
        if i < args.hist:
            results.append({
                "frame_idx": i,
                "pred_cx": coords[i][0], "pred_cy": coords[i][1],
                "angle": 0.0, "direction": "BEKLENIYOR...", "speed_px": 0.0
            })
            continue

        window = coords[i-args.hist : i]
        
        # Normalize
        if mn is not None and scale is not None:
            window_norm = (window - mn) / scale
        else:
            window_norm = window

        inp = torch.FloatTensor(window_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(inp).cpu().numpy()[0]

        # Denormalize
        if mn is not None and scale is not None:
            pred = pred_norm * scale + mn
        else:
            pred = pred_norm
        
        # Yön Hesabı
        target_x, target_y = pred[-1][0], pred[-1][1]
        current_x, current_y = coords[i][0], coords[i][1]
        
        dx = target_x - current_x
        dy = target_y - current_y
        
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        dist = math.sqrt(dx**2 + dy**2)
        
        if dist < 2.0:
            direction_label = "DURGUN"
        else:
            direction_label = get_direction_label(angle_deg)

        results.append({
            "frame_idx": i,
            "pred_cx": float(target_x),
            "pred_cy": float(target_y),
            "angle": float(angle_deg),
            "direction": direction_label,
            "speed_px": float(dist)
        })

    # Kaydet
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"[Success] Augmented forecast saved to {args.out}")

if __name__ == "__main__":
    main()