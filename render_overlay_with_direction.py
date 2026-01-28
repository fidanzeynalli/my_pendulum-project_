from pathlib import Path
import argparse
import cv2
import pandas as pd
import numpy as np

# --- VİDEO ÜZERİNE ANALİZ VERİLERİNİ ÇİZME MODÜLÜ ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Orijinal video dosyası yolu")
    ap.add_argument("--track", required=True, help="Takip verilerini içeren CSV (track.csv)")
    ap.add_argument("--forecast", required=True, help="Tahmin verilerini içeren CSV (forecast.csv)")
    ap.add_argument("--out", required=True, help="Çıktı videosunun kaydedileceği yol")
    ap.add_argument("--codec", default="mp4v", help="Video yazma formatı")  
    args = ap.parse_args()

    # 1. VERİLERİN YÜKLENMESİ
    # Takip ve tahmin verilerini DataFrame olarak oku.
    track_df = pd.read_csv(args.track)
    fc_df = pd.read_csv(args.forecast)

    # Hızlı erişim için verileri sözlük (dictionary) yapısına çevir (Anahtar: Kare İndeksi)
    track_dict = {int(row['frame']): row for _, row in track_df.iterrows()}
    fc_dict = {int(row['frame_idx']): row for _, row in fc_df.iterrows()}

    # 2. VİDEO AYARLARI
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Çıktı videosunu oluşturacak nesneyi tanımla.
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    frame_idx = 0 # Kare sayacı

    # 3. KARE KARE İŞLEME DÖNGÜSÜ
    while True:
        ret, frame = cap.read()
        if not ret: break # Video biterse döngüden çık

        # Mevcut kareye ait verileri sözlükten çek
        tr = track_dict.get(frame_idx)
        fc = fc_dict.get(frame_idx)

        # --- GÖRSELLEŞTİRME BAŞLANGICI ---

        # A) ZAMAN BİLGİSİ: Sol üst köşeye saniye bilgisini yazdırır.
        time_sec = frame_idx / fps
        cv2.putText(frame, f"Time: {time_sec:.2f}s", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # B) YÖN BİLGİSİ: Gelecek tahmininden gelen yönü (SAĞ/SOL) al.
        dir_label = "BELIRSIZ"
        if fc is not None:
            dir_label = str(fc.get('dir_label', 'BELIRSIZ'))

        # C) NESNE TESPİTİ VE ETİKETLEME (KIRMIZI TEMA)
        if tr is not None and not pd.isna(tr.get('cx')):
            cx, cy = int(tr['cx']), int(tr['cy'])
            
            # Nesneyi vurgulamak için KIRMIZI ÇERÇEVE çiziyoruz.
            box_w, box_h = 60, 60 
            x1, y1 = cx - (box_w // 2), cy - (box_h // 2)
            x2, y2 = cx + (box_w // 2), cy + (box_h // 2)

            # mavi(Çerçeve) []
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # b) Etiket Bandı (Kırmızı zemin üzerine siyah yazı)
            label_text = f"ball | {dir_label} | Frame:{frame_idx}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 0, 255), -1)
            
            # Yazıyı yazdır
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # c) Merkez Noktası (Mavi küçük nokta)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

        # D) LSTM GELECEK TAHMİNİ (MAVİ HALKA)
        if fc is not None:
            xp_raw, yp_raw = fc.get('x_pred'), fc.get('y_pred')
            if not pd.isna(xp_raw):
                # Koordinat normalizasyon kontrolü (0-1 arasıysa piksele çevir)
                xp = int(xp_raw * w) if xp_raw < 2 else int(xp_raw)
                yp = int(yp_raw * h) if yp_raw < 2 else int(yp_raw)
                
                # Gelecekteki konumu temsil eden Mavi Halka
                cv2.circle(frame, (xp, yp), 8, (255, 255, 0), 2)

        # --- GÖRSELLEŞTİRME SONU ---

        vw.write(frame) # Hazırlanan kareyi yeni videoya yaz
        frame_idx += 1

    # Kaynakları serbest bırak
    cap.release()
    vw.release()
    print(f"✅ Analiz videosu başarıyla oluşturuldu: {args.out}")

if __name__ == "__main__":
    main()