from pathlib import Path
import argparse
import cv2
import pandas as pd
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--forecast", required=True)
    ap.add_argument("--out", required=True)
    # Codec'i varsayılan olarak mp4v yapalım ki tarayıcıda sorun çıkmasın
    ap.add_argument("--codec", default="mp4v")  
    args = ap.parse_args()

    # Verileri Oku
    track_df = pd.read_csv(args.track)
    fc_df = pd.read_csv(args.forecast)

    # Hızlı erişim için sözlüğe çevir
    track_dict = {int(row['frame']): row for _, row in track_df.iterrows()}
    fc_dict = {int(row['frame_idx']): row for _, row in fc_df.iterrows()}

    # Video Ayarları
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video Yazıcıyı Başlat
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    vw = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        tr = track_dict.get(frame_idx)
        fc = fc_dict.get(frame_idx)

        # --- ÇİZİM İŞLEMLERİ BAŞLANGICI ---

        # 1. Zaman Bilgisi (Sol Üst Köşe)
        time_sec = frame_idx / fps
        cv2.putText(frame, f"Time: {time_sec:.2f}s", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Yön bilgisini önceden alalım
        dir_label = "BELIRSIZ"
        if fc is not None:
            dir_label = str(fc.get('dir_label', 'BELIRSIZ'))

        # 2. Bounding Box ve Etiket Çizimi (Sarkacın Etrafına)
        if tr is not None and not pd.isna(tr.get('cx')):
            cx, cy = int(tr['cx']), int(tr['cy'])
            
            # Kutu boyutu (Sabit bir değer, örn: 60x60 piksel)
            # Sarkacın boyutuna göre bu değerleri değiştirebilirsiniz.
            box_w, box_h = 60, 60 
            x1, y1 = cx - (box_w // 2), cy - (box_h // 2)
            x2, y2 = cx + (box_w // 2), cy + (box_h // 2)

            # a) Yeşil Çerçeve (Kutu) Çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # b) Etiket ve Yön Bilgisini Birleştirip Kutunun Üstüne Yaz
            # Örn: "ball | SAG"
            label_text = f"ball | {dir_label}|Frame:{frame_idx}"
            
            # Yazının arka planı için siyah bir bant çekelim (daha okunaklı olur)
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            
            # Yazıyı yaz
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # c) Merkeze kırmızı bir nokta koy
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # 3. Tahmin Noktası (Mavi)
        if fc is not None:
            xp_raw, yp_raw = fc.get('x_pred'), fc.get('y_pred')
            if not pd.isna(xp_raw):
                # Koordinatları normalize edilmişse piksele çevir
                xp = int(xp_raw * w) if xp_raw < 2 else int(xp_raw)
                yp = int(yp_raw * h) if yp_raw < 2 else int(yp_raw)
                # Tahmin noktası (Mavi)
                cv2.circle(frame, (xp, yp), 6, (255, 255, 0), 2)

        # --- ÇİZİM İŞLEMLERİ SONU ---

        vw.write(frame)
        frame_idx += 1

    cap.release()
    vw.release()
    print(f"✅ Video başarıyla oluşturuldu: {args.out}")

if __name__ == "__main__":
    main()