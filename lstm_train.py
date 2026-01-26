from pathlib import Path
import argparse, numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

#VERİ HAZIRLAMA SINIFI (Dataset)
# Bu sınıf, uzun bir koordinat listesini yapay zekanın öğrenebileceği "geçmiş-gelecek" çiftlerine böler.
class SeqDS(Dataset):
    def __init__(self, arr, hist=30, horizon=5):
        self.X, self.y = [], []
        # kayar pencere yöntemi 30 kareye bak, sonraki 5 kareyi hedefle.
        for i in range(len(arr) - hist - horizon + 1):
            self.X.append(arr[i:i+hist])           # Modelin eski verisi
            self.y.append(arr[i+hist:i+hist+horizon]) # Modelin tahmin etmesi gereken yani gelecek verisi
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

# LSTM MODEL MİMARİSİ
#sarkaç gibi periyodik hareketleri hafızasında tutabilen özel bir sinir ağı yapısıdır.
class LSTMSeq2Seq(nn.Module):
    def __init__(self, in_dim=2, hid=64, layers=2, horizon=5):
        super().__init__()
        # LSTM Katmanı Zaman içerisindeki örüntüyü ve momentumu yakalar.
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True)
        # Çıkış Katmanı: gelecek horizon adım için her biri 2 koordinat (x,y) çıkartır
        # horizon=5 için: 2 * 5 = 10 çıkış
        self.head = nn.Linear(hid, 2 * horizon)
        self.horizon = horizon
        
    def forward(self, x):
        out,_ = self.lstm(x)    # önce veriyi LSTM'den geçir
        last = out[:, -1, :]      # Dizideki en son bilgiyi odak noktası al
        yhat = self.head(last)  # Son hidden state'den gelecek tahminlerini üret [batch, 2*horizon]
        yhat = yhat.view(-1, self.horizon, 2)  # [batch, horizon, 2] şekline dönüştür
        return yhat

def main():
    #  AYARLAR VE VERİ OKUMA
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=str, required=True) # YOLO'dan çıkan takip CSV dosyası
    ap.add_argument("--epochs", type=int, default=50)   # Eğitim tur sayısı (mesela: 50 defa verinin üzerinden geç)
    ap.add_argument("--batch", type=int, default=16)    # Batch boyutu
    ap.add_argument("--hist", type=int, default=30)     # Geçmiş veri pencere boyutu
    ap.add_argument("--horizon", type=int, default=5)   # Tahmin edilen gelecek adım sayısı
    ap.add_argument("--out", type=str, default="outputs/models/lstm_xy.pt") # modelin kaydedileceği yer
    args = ap.parse_args()

    df = pd.read_csv(args.track)
    df = df.dropna(subset=["cx", "cy"])
    # smoothlaştırılmış  verileri eğitim için diziye çevir.
    arr = df[["cx","cy"]].to_numpy(dtype=np.float32)    # NORMALİZASYON

    if len(arr) < 40: # Çok az veri varsa uyar
        print("Hata: Eğitim için yeterli veri yok (en az 40-50 kare sarkaç tespiti gerekli).")
        return

    # Koordinatları 0 ile 1 arasına sıkıştırır. Bu, yapay zekanın çok daha hızlı ve kararlı öğrenmesini sağlar.
    mn = arr.min(axis=0, keepdims=True)
    mx = arr.max(axis=0, keepdims=True)
    scale = (mx - mn); scale[scale==0] = 1.0
    arrn = (arr - mn) / scale

    # VERİ BÖLME (Eğitim ve Doğrulama)
    # Verinin 70faizni öğrenmek için, geri kalanını kendini validation etmek için kullanır.
    n = len(arrn)
    n_tr = int(n*0.7); n_va = int(n*0.85)
    ds_tr = SeqDS(arrn[:n_tr], args.hist, args.horizon)
    ds_va = SeqDS(arrn[n_tr:n_va], args.hist, args.horizon)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=False)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False)

    # EĞİTİM DÖNGÜSÜ
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMSeq2Seq(in_dim=2, hid=64, layers=2, horizon=args.horizon).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3) # Ağırlıkları güncelleyen algoritma
    loss_fn = nn.MSELoss() # Hata payı hesaplayıcı (Tahmin ile gerçek arasındaki farkın karesi)

    best_loss = float("inf")
    for ep in range(args.epochs):
        model.train()
        tr_loss_sum = 0
        for X,y in dl_tr:
            X,y = X.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(X)
            loss = loss_fn(yhat, y) # hatayı hesapla
            loss.backward()         # Hatayı geriye doğru yay (Öğrenme aşaması)
            opt.step() # Modeli güncelle
            tr_loss_sum += loss.item()
        
        # VALIDATION - Modeli test verisiyle kontrol etmek
        model.eval()
        va_loss_sum = 0
        with torch.no_grad():  # Gradient hesaplamaya gerek yok, sadece tahmin et
            for X,y in dl_va:
                X,y = X.to(device), y.to(device)
                yhat = model(X)
                va_loss = loss_fn(yhat, y)
                va_loss_sum += va_loss.item()
        
        # Ortalama hataları hesapla
        tr_loss_avg = tr_loss_sum / len(dl_tr)
        va_loss_avg = va_loss_sum / len(dl_va)
        
        # Her 10 epoch'ta bir konsola yazdır
        if (ep + 1) % 10 == 0:
            print(f"Epoch {ep+1}/{args.epochs} | Train Loss: {tr_loss_avg:.6f} | Val Loss: {va_loss_avg:.6f}")
            
        #  MODELİ KAYDETME
        # Eğer bu turdaki validation hata payı şimdiye kadarki en düşükse, bu son modeli diske kaydet.
        if va_loss_avg < best_loss:
            best_loss = va_loss_avg
            outp = Path(args.out).resolve()
            outp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "mn": mn, "scale": scale # normalizasyon değerlerini de sakla ki tahmin yaparken kullanılsın
            }, outp)
            print(f"  ✓ Saved best model (val_loss: {va_loss_avg:.6f})")
    if 'outp' in locals():
        print(f"Saved LSTM → {outp}")
    else:
        print("Model eğitilemedi, veri setini kontrol edin.")

if __name__ == "__main__":
    main()
