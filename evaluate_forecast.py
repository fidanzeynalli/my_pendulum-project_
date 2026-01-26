from pathlib import Path
import argparse, numpy as np, pandas as pd

def main():
    # Karşılaştırma yapılacak dosyaları tanımlıyoruz
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=str, required=True)    #  YOLO'dan gelen gerçek takip verileri
    ap.add_argument("--forecast", type=str, required=True) # LSTM'den çıkan tahmin verileri
    ap.add_argument("--hist", type=int, default=30)        # Geçmiş veri pencere boyutu
    ap.add_argument("--horizon", type=int, default=5)      # Tahmin edilen gelecek adım sayısı
    args = ap.parse_args()

    # VERİ OKUMA Gerçek verileri ve tahminleri CSV'den yüklüyoruz
    trk = pd.read_csv(args.track)
    fc  = pd.read_csv(args.forecast)

    #  VERİ AYRIŞTIRMA Gerçek videodaki sarkaç konumunun son 'horizon'
    # Yapay zekanın ürettiği tahmin edilen koordinatları alıyoruz. kadarını (en güncel konumlar) alıyoruz.
    lastH = trk[["cx_smooth","cy_smooth"]].to_numpy()[-args.horizon:]
    pred  = fc[["x_pred","y_pred"]].to_numpy()

    # BOYUT Karşılaştırma yapabilmek için veri uzunluklarını eşitliyoruz
    n = min(len(lastH), len(pred))
    lastH, pred = lastH[:n], pred[:n]

    # 5. HATA HESAPLAMA  'diff' değişkeni ile hata miktarını temsil ediliyo
    diff = pred - lastH
    
    # MAE /Mean Absolute Error (X ve Y için ayrı ayrı)
    mae_xy = np.mean(np.abs(diff), axis=0)
    
    # RMSE (Büyük hataları daha çok cezalandırır)
    rmse_xy= np.sqrt(np.mean(diff**2, axis=0))
    
    # Radyal Hata X ve Y bileşenlerinin bileşkesi (Öklid mesafesi üzerinden hata hesabı)
    mae_r  = np.mean(np.sqrt(np.sum((diff)**2, axis=1)))
    rmse_r = np.sqrt(np.mean(np.sum((diff)**2, axis=1)))

    # Hata istatistiklerini terminale yazdırıyoruz 
    print(f"Pairs compared: {n}")
    print(f"MAE  x,y  (px): {mae_xy[0]:.3f}, {mae_xy[1]:.3f}")
    print(f"RMSE x,y  (px): {rmse_xy[0]:.3f}, {rmse_xy[1]:.3f}")
    print(f"MAE  radial (px): {mae_r:.3f}")
    print(f"RMSE radial (px): {rmse_r:.3f}")

if __name__ == "__main__":
    main()