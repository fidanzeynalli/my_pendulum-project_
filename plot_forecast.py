from pathlib import Path
import argparse, pandas as pd, matplotlib.pyplot as plt

def main():
    #grafiği oluşturmak için gereken verileri tanımlıyoruz
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=str, required=True)    #gerçek takip verileri (YOLO çıktıları)
    ap.add_argument("--forecast", type=str, required=True) # tahmin verileri (LSTM çıktıları)
    #Çıkış: grafiğin kaydedileceği klasör ve dosya ismi
    ap.add_argument("--out", type=str, default="outputs/plots/trajectory_with_forecast.png")
    args = ap.parse_args()

    #CSV dosyalarını hafızaya yani DataFrame'e yüklüyoruz
    trk = pd.read_csv(args.track)
    fc  = pd.read_csv(args.forecast)

    #çıktı klasörü (outputs/plots) yoksa oluşturulur
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    
    #Visualization
    plt.figure() # Yeni bir grafik penceresi açıyor
    
    #YOLO'dan gelen temizlenmiş (cx_smooth, cy_smooth) koordinatları mavi çizgiyle çizer
    plt.plot(trk["cx_smooth"], trk["cy_smooth"], label="track")
    
    #LSTM'den gelen tahminleri (x_pred, y_pred) kesikli çizgili noktalarla ("o--") çizer
    plt.plot(fc["x_pred"], fc["y_pred"], "o--", label="forecast")

    #KOORDİNAT SİSTEMİ DÜZENLEME
    # Bilgisayarlı görüde Y ekseni yukarıdan aşağıya doğru artar. 
    # Grafiğin video karesiyle aynı durması için Y eksenini ters çeviriyoruz.
    plt.gca().invert_yaxis()   

    # ETİKETLEME VE KAYIT
    plt.legend() 
    plt.title("Trajectory + Forecast") 
    plt.xlabel("x (px)") 
    plt.ylabel("y (px)")

    #hazırlanan grafiği belirtilen yola yüksek çözünürlükte (dpi=160) kaydeder
    plt.savefig(out, dpi=160); plt.close()
    print(f"Saved plot → {out}")

if __name__ == "__main__":
    main()