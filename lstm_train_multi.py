import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import glob

# --- VERİ SETİ SINIFI ---
# Hazırlanmış parça parça verileri PyTorch'un anlayacağı formata çevirir
class SeqDS(Dataset):
    def __init__(self, X_list, y_list):
        # Listeleri birleştirip tensor yapıyoruz
        # X_list içinde her videodan gelen [örnek_sayisi, 30, 2] boyutunda veriler var
        self.X = torch.tensor(np.concatenate(X_list, axis=0), dtype=torch.float32)
        self.y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# --- LSTM MODEL MİMARİSİ (Mevcut ile aynı) ---
class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim=2, hid=64, layers=2, horizon=5):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hid, 2 * horizon)
        self.horizon = horizon
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        yhat = self.head(last)
        return yhat.view(-1, self.horizon, 2)

def main():
    ap = argparse.ArgumentParser()
    # --folder: İçinde birden fazla track_xx.csv olan klasör yolu
    ap.add_argument("--folder", type=str, required=True, help="CSV dosyalarinin oldugu klasor")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--hist", type=int, default=30)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--out", type=str, default="outputs/models/lstm_best_multi.pt")
    args = ap.parse_args()

    # 1. TÜM CSV DOSYALARINI BUL
    # Verilen klasördeki .csv ile biten her dosyayı listeler
    csv_files = glob.glob(f"{args.folder}/*.csv")
    print(f"📂 Bulunan dosya sayisi: {len(csv_files)}")
    
    if len(csv_files) == 0:
        print("❌ Hata: Klasörde hiç .csv dosyası bulunamadı!")
        return

    # 2. HER DOSYAYI ÖNCE OKU (Global Normalizasyon için)
    # Tüm videolardaki en küçük ve en büyük koordinatları bulmamız lazım ki
    # hepsini aynı ölçeğe (0 ile 1 arasına) sıkıştırabilelim.
    all_raw_data = [] 
    valid_X_list = []
    valid_y_list = []

    print("📊 Veriler okunuyor ve analiz ediliyor...")
    for f in csv_files:
        try:
            df_temp = pd.read_csv(f)
            # Sadece cx, cy olan ve boş olmayan satırları al
            if "cx" not in df_temp.columns or "cy" not in df_temp.columns:
                print(f"   ⚠️ Atlandı (cx/cy yok): {Path(f).name}")
                continue
                
            df_temp = df_temp.dropna(subset=["cx", "cy"])
            arr_temp = df_temp[["cx", "cy"]].to_numpy(dtype=np.float32)
            
            # Eğer video çok kısaysa (eğitim için yetersizse) atla
            if len(arr_temp) < (args.hist + args.horizon + 5):
                print(f"   ⚠️ Atlandı (Çok kısa): {Path(f).name}")
                continue
                
            all_raw_data.append(arr_temp)
            print(f"   ✅ Eklendi: {Path(f).name} ({len(arr_temp)} kare)")
            
        except Exception as e:
            print(f"   ❌ Hata ({Path(f).name}): {e}")

    if not all_raw_data:
        print("❌ Hata: Hiçbir dosyadan geçerli veri alınamadı.")
        return

    # 3. GLOBAL ÖLÇEĞİ HESAPLA (Min - Max)
    big_arr = np.concatenate(all_raw_data, axis=0)
    mn = big_arr.min(axis=0, keepdims=True)
    mx = big_arr.max(axis=0, keepdims=True)
    scale = (mx - mn)
    scale[scale == 0] = 1.0 # Sıfıra bölünmeyi önle

    print(f"📏 Global Ölçek: min={mn.flatten()}, scale={scale.flatten()}")

    # 4. HER VİDEOYU AYRI AYRI PARÇALA (Sliding Window)
    # Burası kritik: Her videoyu kendi içinde parçalıyoruz, birbirine bağlamıyoruz.
    for arr in all_raw_data:
        # Normalize et
        arrn = (arr - mn) / scale
        
        # Parçala (Geçmiş 30 -> Gelecek 5)
        X_sub, y_sub = [], []
        for i in range(len(arrn) - args.hist - args.horizon + 1):
            X_sub.append(arrn[i : i + args.hist])
            y_sub.append(arrn[i + args.hist : i + args.hist + args.horizon])
        
        if len(X_sub) > 0:
            valid_X_list.append(np.array(X_sub))
            valid_y_list.append(np.array(y_sub))

    # 5. VERİ SETİNİ OLUŞTUR
    full_ds = SeqDS(valid_X_list, valid_y_list)
    
    # %80 Eğitim, %20 Doğrulama (Test) olarak ayır
    total_len = len(full_ds)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    ds_tr, ds_va = torch.utils.data.random_split(full_ds, [train_len, val_len])

    # Shuffle=True ile verileri karıştırıyoruz (farklı videolardan gelen örnekler karışsın)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True) 
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False)
    
    print(f"🚀 Eğitim Başlıyor: Toplam {total_len} örnek (Train: {train_len}, Val: {val_len})")

    # 6. EĞİTİM DÖNGÜSÜ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️ Çalışma ortamı: {device}")
    
    model = LSTMSeq2Seq(in_dim=2, hid=64, layers=2, horizon=args.horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    
    for ep in range(args.epochs):
        model.train()
        tr_loss_sum = 0
        for X, y in dl_tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(X)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            tr_loss_sum += loss.item()
            
        model.eval()
        va_loss_sum = 0
        with torch.no_grad():
            for X, y in dl_va:
                X, y = X.to(device), y.to(device)
                yhat = model(X)
                va_loss_sum += loss_fn(yhat, y).item()
        
        tr_avg = tr_loss_sum/len(dl_tr)
        va_avg = va_loss_sum/len(dl_va)
        
        # Her 5 turda bir bilgi ver
        if (ep+1) % 5 == 0:
            print(f"Ep {ep+1}/{args.epochs} | Train Loss: {tr_avg:.5f} | Val Loss: {va_avg:.5f}")
            
        # En iyi modeli kaydet
        if va_avg < best_loss:
            best_loss = va_avg
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "mn": mn, "scale": scale # Normalizasyon parametrelerini de içine gömüyoruz
            }, args.out)

    print(f"✅ Eğitim tamamlandı. Model kaydedildi: {args.out}")

if __name__ == "__main__":
    main()