from pathlib import Path
import argparse, pandas as pd, numpy as np


def centroid_distance(p1, p2):
    """İki nokta arasındaki Öklid mesafesini hesapla"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def simple_centroid_tracking(df, max_distance=50.0):
    """
    Basit Centroid Tracking: Ardışık frameler arasındaki merkez noktaları eşleştir.
    Eğer tespit yoksa NaN bırak (interpolasyon tarafından doldurulacak).
    """
    df = df.copy()
    df['track_id'] = np.nan
    
    track_id = 1
    prev_centroid = None
    
    for idx, row in df.iterrows():
        # Eğer bu frame'de tespit varsa (cx ve cy NaN değilse)
        if pd.notna(row['cx']) and pd.notna(row['cy']):
            curr_centroid = (row['cx'], row['cy'])
            
            # Eğer bir önceki frame'de de tespit varsa
            if prev_centroid is not None:
                dist = centroid_distance(prev_centroid, curr_centroid)
                # Eğer mesafe makul ise aynı nesne olarak kabul et
                if dist < max_distance:
                    df.at[idx, 'track_id'] = track_id
                else:
                    # Çok uzak atlamışsa yeni nesne olabilir (ama tek nesne için genellikle olmaz)
                    track_id += 1
                    df.at[idx, 'track_id'] = track_id
            else:
                # İlk tespit
                df.at[idx, 'track_id'] = track_id
            
            prev_centroid = curr_centroid
        else:
            # Bu frame'de tespit yok, track_id NaN kalacak (sonra interpolasyon yapılacak)
            prev_centroid = None
    
    return df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", type=str, required=True, help="Input detections CSV")
    parser.add_argument("--out", type=str, default=str(Path("outputs/track.csv")), help="Output track CSV")
    parser.add_argument("--ema", type=float, default=0.2, help="EMA alpha (0-1)")
    parser.add_argument("--interp_limit", type=int, default=5, help="Max consecutive NaNs to interpolate")
    parser.add_argument("--max_track_dist", type=float, default=50.0, help="Max distance for centroid matching (px)")
    parser.add_argument("--px_per_cm", type=float, default=None, help="If set, also export cm units")
    args = parser.parse_args()


    det_csv = Path(args.detections).resolve()
    out_csv = Path(args.out).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(det_csv)

    # CENTROID TRACKING (Nesne Kimliği Sürekliliği)
    # Ardışık frameler arasındaki merkez noktaları eşleştir
    df = simple_centroid_tracking(df, max_distance=args.max_track_dist)

    # zaman ekseni
    # time_mode=cfr ise t sütunu zaten frame/fps ile; vfr ise POS_MSEC/1000
    # t yoksa son çare: frame/fps

    if "t" not in df.columns or df["t"].isna().all():
        if "fps" in df.columns and df["fps"].iloc[0] and "frame" in df.columns:
            df["t"] = df["frame"] / float(df["fps"].iloc[0])
        else:
            raise RuntimeError("No usable time axis found.")



    # centroid sütunları (YOLO aşamasında da geliyor ama NaN olabilir)
    if "cx" not in df.columns or "cy" not in df.columns:
        df["cx"] = (df["x1"] + df["x2"]) / 2.0
        df["cy"] = (df["y1"] + df["y2"]) / 2.0


    # interpolate kısa boşluklar
    df["cx"] = df["cx"].interpolate(limit=args.interp_limit)
    df["cy"] = df["cy"].interpolate(limit=args.interp_limit)

    # EMA smoothing
    alpha = args.ema
    df["cx_smooth"] = df["cx"].ewm(alpha=alpha, adjust=False).mean()
    df["cy_smooth"] = df["cy"].ewm(alpha=alpha, adjust=False).mean()



    # hız/ivme (piksel birimi)
    t = df["t"].values
    cx = df["cx_smooth"].values
    cy = df["cy_smooth"].values


    dt = np.gradient(t)
    vx = np.gradient(cx, t)
    vy = np.gradient(cy, t)
    speed = np.sqrt(vx**2 + vy**2)


    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    df["speed_px_s"] = speed
    df["ax_px_s2"] = ax
    df["ay_px_s2"] = ay

    # fiziksel birime dönmek istersen
    if args.px_per_cm and args.px_per_cm > 0:
        s = float(args.px_per_cm)
        df["x_cm"] = df["cx_smooth"] / s
        df["y_cm"] = df["cy_smooth"] / s
        df["speed_cm_s"] = df["speed_px_s"] / s
        df["ax_cm_s2"] = df["ax_px_s2"] / s
        df["ay_cm_s2"] = df["ay_px_s2"] / s

    # önemli kolonları seçip kaydet
    keep = [c for c in ["frame","t","track_id","cx","cy","cx_smooth","cy_smooth",
                        "speed_px_s","ax_px_s2","ay_px_s2",
                        "x_cm","y_cm","speed_cm_s","ax_cm_s2","ay_cm_s2"] if c in df.columns]
    out = df[keep]
    out.to_csv(out_csv, index=False)
    print(f"Saved track → {out_csv}")
    
    # Tracking istatistikleri
    unique_ids = out['track_id'].nunique()
    print(f"Number of unique track IDs: {unique_ids}")

if __name__ == "__main__":
    main()