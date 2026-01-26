import torch
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import torch.nn as nn

# LSTM Model Mimarisi - Eğitimdeki mimari ile birebir aynı olmalı
class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim=2, hid=64, layers=2, horizon=5):
        super().__init__()
        self.horizon = horizon
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True)
        # Çıkış Katmanı: gelecek horizon adım için her biri 2 koordinat (x,y) çıkartır
        self.head = nn.Linear(hid, 2 * horizon)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.head(out[:, -1, :])
        return out.view(-1, self.horizon, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, help="track_video.csv yolu")
    ap.add_argument("--model", required=True, help="lstm_xy.pt yolu")
    ap.add_argument("--out", required=True, help="Çıkış forecast_video.csv")
    ap.add_argument("--hist", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    # Cihaz Seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model Yükleme
    model = LSTMSeq2Seq(in_dim=2, hid=64, layers=2, horizon=args.horizon).to(device)
    
    # Güvenli yükleme (Weights_only hatasını aşmak için)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # Model sözlük yapısındaysa (state_dict) ayıkla
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    
    # Normalizasyon parametrelerini yükle (eğitim sırasında kaydedilmişler)
    mn = checkpoint.get('mn', None)
    scale = checkpoint.get('scale', None)
    
    if mn is None or scale is None:
        print("[warning] Normalizasyon parametreleri bulunamadı. Normalizasyon yapılmayacak.")
        normalize = False
    else:
        normalize = True
        print(f"[info] Normalizasyon parametreleri yüklendi: min={mn}, scale={scale}")
    
    model.eval()

    # Veri Okuma ve Hazırlama
    df = pd.read_csv(args.track)
    
    # Akıllı Kolon Seçimi: cx_smooth yoksa cx'i kullan
    # Eğer filtre uygulanmış temiz veri (cx_smooth) varsa onu kullanır, yoksa ham veriyi alır
    x_col = 'cx_smooth' if 'cx_smooth' in df.columns else 'cx'
    y_col = 'cy_smooth' if 'cy_smooth' in df.columns else 'cy'
    # Boş verileri doldur veya temizle 
    # Eksik verileri (NaN) bir önceki kareyle doldurarak sistemin çökmesini önler
    df = df.ffill() 
    
    # Koordinatları numpy dizisine çevir (Kritik: coords burada tanımlandı)
    coords = df[[x_col, y_col]].values
    
    all_forecasts = []
    
    print(f"[info] '{x_col}' kolonları kullanılarak tahmin yapılıyor...")
    
    # Kayar pencere (sliding window) ile tahmin döngüsü
    for i in range(len(coords)):
        if i < args.hist:
            # Yeterli geçmiş yoksa mevcut konumu tahmin olarak ata
            all_forecasts.append({
                "frame_idx": i,
                "x_pred": coords[i, 0],
                "y_pred": coords[i, 1]
            })
            continue
            
        # Son 'hist' (30) kareyi pencere olarak al
        window = coords[i-args.hist : i]
        
        # NORMALİZASYON: Eğitim sırasında normalize edilmişse, tahmin sırasında da yapmalıyız
        if normalize:
            window_norm = (window - mn) / scale
        else:
            window_norm = window
        
        inp = torch.FloatTensor(window_norm).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Model tahmini yap [batch=1, horizon, 2]
            pred_norm = model(inp).cpu().numpy()[0]  # [horizon, 2]
            
            # DENORMALIZASYON: Tahminleri gerçek koordinat sistemine geri çevir
            if normalize:
                pred = pred_norm * scale + mn
            else:
                pred = pred_norm
            
            # En yakın gelecek tahmini (horizon içindeki ilk kare) kaydedilir
            all_forecasts.append({
                "frame_idx": i,
                "x_pred": float(pred[0, 0]), 
                "y_pred": float(pred[0, 1])
            })

    # Sonuçları CSV'ye kaydet
    out_df = pd.DataFrame(all_forecasts)
    out_df.to_csv(args.out, index=False)
    print(f"[ok] Tahminler başarıyla kaydedildi: {args.out}")

if __name__ == "__main__":
    main()