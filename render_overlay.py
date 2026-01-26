from pathlib import Path
import argparse, cv2, pandas as pd, numpy as np

def draw_trail(img, pts, color=(0,255,0), thickness=2):
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None: continue
        cv2.line(img, pts[i-1], pts[i], color, thickness)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Input video path (mp4/avi)")
    ap.add_argument("--track", type=str, required=True, help="Track CSV")
    ap.add_argument("--out",   type=str, required=True, help="Output file (.mp4 or .avi)")
    ap.add_argument("--trail", type=int, default=30, help="Trail length")
    ap.add_argument("--codec", type=str, default="mp4v", help="FourCC: mp4v, XVID, MJPG")
    ap.add_argument("--limit", type=int, default=None, help="Process first N frames")
    args = ap.parse_args()

    vid = Path(args.video); trk = Path(args.track); out = Path(args.out)
    print(f"[info] video: {vid.resolve()}")
    print(f"[info] track: {trk.resolve()}")
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(trk)
    print(f"[info] track rows: {len(df)}")
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise RuntimeError(f"[error] cannot open video: {vid}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[info] meta: fps={fps:.3f}, size={W}x{H}, frames≈{int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)}")

    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(out), fourcc, fps, (W,H))
    print(f"[info] writer opened: {writer.isOpened()} codec={args.codec} out={out}")
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"[error] VideoWriter failed. Try: --codec XVID and use .avi")

    pts = []; written = 0; i = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ok, frame = cap.read()
        if not ok: print("[warn] read failed/end"); break
        if i >= len(df): print("[warn] track ended"); break
        if args.limit and i >= args.limit: print("[info] limit reached"); break

        row = df.iloc[i]
        if not (np.isnan(row.get("cx_smooth", np.nan)) or np.isnan(row.get("cy_smooth", np.nan))):
            c = (int(row["cx_smooth"]), int(row["cy_smooth"]))
            pts.append(c); pts = pts[-args.trail:]
            for k in range(1,len(pts)):
                cv2.line(frame, pts[k-1], pts[k], (0,255,0), 2)
            cv2.circle(frame, c, 6, (0,0,255), -1)

        t = float(row.get("t", i / fps))
        spd = row.get("speed_px_s", np.nan)
        cv2.putText(frame, f"t={t:.2f}s", (10,30), font, 1, (255,255,255), 2)
        if not np.isnan(spd):
            cv2.putText(frame, f"speed={float(spd):.1f} px/s", (10,65), font, 0.8, (255,255,255), 2)

        writer.write(frame); written += 1
        if written % 100 == 0:
            print(f"[info] written={written}")

        i += 1

    cap.release(); writer.release()
    print(f"[ok] Saved overlay → {out.resolve()}  (frames={written})")

if __name__ == "__main__":
    main()
