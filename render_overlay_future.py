import cv2
import pandas as pd
import argparse
import numpy as np
from pathlib import Path

def draw_hud(frame, data_row, W, H):
    overlay = frame.copy()
    
    # Kutu Çizimi: Sol Üst Köşe (10, 10)
    # (350, 150) kutunun sağ alt köşesidir, metin sığsın diye genişlettim.
    cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
    
    # Şeffaflık ekle
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 255, 255) # Sarı renk
    
    # --- İSTEDİĞİNİZ BİLGİLER BURADA YAZILIYOR ---
    lines = [
        f"Frame: {int(data_row['frame_idx'])}",
        f"Gelecek Merkez X: {int(data_row['pred_cx'])}",
        f"Gelecek Merkez Y: {int(data_row['pred_cy'])}",
        f"Gidis Yonu: {data_row['direction']}",
        f"Tahmini Mesafe: {data_row['speed_px']:.1f} px"
    ]
    
    y = 35
    for line in lines:
        cv2.putText(frame, line, (20, y), font, 0.6, color, 1, cv2.LINE_AA)
        y += 25 # Satır aralığı

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--forecast", required=True)
    ap.add_argument("--out", default="outputs/FINAL_AUGMENTED.mp4")
    args = ap.parse_args()

    track_df = pd.read_csv(args.track)
    fc_df = pd.read_csv(args.forecast)
    
    cap = cv2.VideoCapture(args.video)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))

    idx = 0
    print("[Info] Rendering per-frame overlay...")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        if idx < len(track_df) and idx < len(fc_df):
            # Mevcut Konum
            if 'cx_smooth' in track_df.columns:
                cur_x, cur_y = track_df.loc[idx, 'cx_smooth'], track_df.loc[idx, 'cy_smooth']
            else:
                cur_x, cur_y = track_df.loc[idx, 'cx'], track_df.loc[idx, 'cy']
            
            # Gelecek Tahmin
            fc_row = fc_df.loc[idx]
            pred_x, pred_y = fc_row['pred_cx'], fc_row['pred_cy']
            
            if not (np.isnan(cur_x) or np.isnan(pred_x)):
                cur_pt = (int(cur_x), int(cur_y))
                pred_pt = (int(pred_x), int(pred_y))

                # 1. Mevcut Konum (Yeşil)
                cv2.circle(frame, cur_pt, 6, (0, 255, 0), -1)
                
                # 2. Gelecek Merkez Noktası (Kırmızı Hedef)
                cv2.circle(frame, pred_pt, 6, (0, 0, 255), 2)
                cv2.line(frame, (pred_pt[0]-5, pred_pt[1]), (pred_pt[0]+5, pred_pt[1]), (0,0,255), 1)
                cv2.line(frame, (pred_pt[0], pred_pt[1]-5), (pred_pt[0], pred_pt[1]+5), (0,0,255), 1)
                
                # 3. Yön Oku
                if fc_row['direction'] != "DURGUN":
                    cv2.arrowedLine(frame, cur_pt, pred_pt, (255, 0, 255), 3, tipLength=0.3)

                # 4. Sol Üst Köşe Bilgisi
                draw_hud(frame, fc_row, W, H)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"[Success] Video saved to {args.out}")

if __name__ == "__main__":
    main()