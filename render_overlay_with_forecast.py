from pathlib import Path
import argparse, cv2, pandas as pd, numpy as np

def main():
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True)     # Giriş: Orijinal sarkaç videosu
    ap.add_argument("--track", type=str, required=True)     #  YOLO'dan gelen takip verileri
    ap.add_argument("--forecast", type=str, required=True)  #LSTM'den gelen tahmin verileri
    ap.add_argument("--out", type=str, default="outputs/overlay_with_fc.avi") # Çıkış yolu
    ap.add_argument("--codec", type=str, default="XVID")    # Video sıkıştırma formatı (XVID uyumluluğu yüksektir)
    ap.add_argument("--trail", type=int, default=40)        # Görsel efekt: Sarkacın arkasında bıraktığı izin uzunluğu
    args = ap.parse_args()

    vid = Path(args.video); trk = Path(args.track); fcsv = Path(args.forecast)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trk)
    fc = pd.read_csv(fcsv)

  
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened(): raise RuntimeError("cannot open video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W,H = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*args.codec), fps, (W,H))
    if not writer.isOpened(): raise RuntimeError("writer failed")

    pts=[]; i=0 # İz bırakma noktaları için liste ve kare sayacı
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    while True:
        ok, frame = cap.read() # Kare kare videoyu oku
        if not ok or i>=len(df): break
        row=df.iloc[i]

        #TRAIL VE MERKEZ ÇİZİM
        #NaN kontrolü
        if not (np.isnan(row["cx"]) or np.isnan(row["cy"])):
            # 'c' değişkenini burada tanımlıyoruz
            c = (int(row["cx"]), int(row["cy"])) 
            pts.append(c)
            pts = pts[-args.trail:] # Son 40 karelik izi tut
            
            # İzi yeşil çizgilerle çiz
            for k in range(1, len(pts)):
                cv2.line(frame, pts[k-1], pts[k], (0, 255, 0), 2)
            
            # Mevcut merkeze kırmızı nokta koy
            cv2.circle(frame, c, 6, (0, 0, 255), -1)

        #GELECEK TAHMİNLERİNİN ÇİZİMİ 
        # Analiz bittiğinde (son kareye gelindiğinde) LSTM tahminlerini mavi noktalarla ekle
        if i == len(df)-1:
            for j in range(len(fc)):
                x=int(fc.loc[j,"x_pred"]); y=int(fc.loc[j,"y_pred"])
                # Mavi tahmin noktaları (Gelecek konumu temsil eder)
                cv2.circle(frame, (x,y), 5, (255,0,0), -1)
                #tahmin noktalarının yanına kare numarasını (+1, +2...) yaz
                cv2.putText(frame, f"+{j+1}", (x+6,y-6), font, 0.5, (255,0,0), 1)

        writer.write(frame); i+=1 #işlenen kareyi yeni videoya yaz

    cap.release(); writer.release()
    print(f"Saved overlay + forecast → {out}")

if __name__ == "__main__":
    main()