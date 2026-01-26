
from pathlib import Path
import argparse, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", type=str, required=True)# Giriş: YOLO ve smoothing sonrası elde edilen takip verisi (CSV)
    ap.add_argument("--outdir", type=str, default="outputs/plots")# Çıkış: Grafiklerin kaydedileceği klasör
    args = ap.parse_args()

    trk_path = Path(args.track)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    #Analiz edilmiş takip verilerini yüklüyoruz [cite: 42]
    df = pd.read_csv(trk_path)

    # Trajectory (yukarı ekseni görüntü koordinatında ters çeviriyoruz)
    # Sarkaç topunun 2D düzlemde izlediği yolu (x vs y) çizer [cite: 39]
    plt.figure()
    plt.plot(df["cx_smooth"], df["cy_smooth"])
    plt.gca().invert_yaxis()
    plt.title("Trajectory (cx_smooth vs cy_smooth)")
    plt.xlabel("x (px)"); plt.ylabel("y (px)")
    plt.savefig(outdir / f"{trk_path.stem}_trajectory.png", dpi=160)
    plt.close()

    # x(t)
    # Sarkaç salınımının zaman içerisindeki yatay değişimini (periyodunu) gösterir [cite: 41, 170]
    plt.figure()
    plt.plot(df["t"], df["cx_smooth"])
    plt.title("x(t)"); plt.xlabel("t (s)"); plt.ylabel("x (px)")
    plt.savefig(outdir / f"{trk_path.stem}_x_t.png", dpi=160)
    plt.close()

    # y(t)
    # Sarkaç topunun dikeydeki (yükseklik) salınımını çizer [cite: 39]
    plt.figure()
    plt.plot(df["t"], df["cy_smooth"])
    plt.title("y(t)"); plt.xlabel("t (s)"); plt.ylabel("y (px)")
    plt.savefig(outdir / f"{trk_path.stem}_y_t.png", dpi=160)
    plt.close()

    # speed(t) varsa
    #Eğer track_and_export.py modülünde hız hesaplanmışsa bu grafiği çizer
    if "speed_px_s" in df.columns:
        plt.figure()
        plt.plot(df["t"], df["speed_px_s"])
        plt.title("speed(t)"); plt.xlabel("t (s)"); plt.ylabel("speed (px/s)")
        plt.savefig(outdir / f"{trk_path.stem}_speed_t.png", dpi=160)
        plt.close()

    print(f"Saved plots → {outdir}")

if __name__ == "__main__":
    main()

