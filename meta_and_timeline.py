from pathlib import Path
import os, cv2, pandas as pd
from typing import Dict, Any, List

#KÖK DİZİNİ HER ZAMAN DOSYANIN KONUMUNA SABİTLE!
BASE_DIR = Path(__file__).resolve().parent

#KULLANICI AYARI 
VIDEO_PATH = BASE_DIR / "data" / "videos" / "video.mp4"   #videonun yolu
MAX_PROBE_FRAMES = 300
OUT_DIR = BASE_DIR / "outputs"
META_CSV = OUT_DIR / "video_metadata.csv"
TIMELINE_CSV = OUT_DIR / "video_timeline_probe.csv"

OUT_DIR.mkdir(parents=True, exist_ok=True)

os.makedirs(OUT_DIR, exist_ok=True)

def get_video_info(path: str) -> Dict[str, Any]:
    info = {
        "path": path, "exists": os.path.exists(path),
        "opened": False, "fps": None, "width": None, "height": None,
        "frame_count": None, "duration_sec_by_header": None, "fourcc": None,
    }
    if not info["exists"]:
        return info
    cap = cv2.VideoCapture(path)
    info["opened"] = cap.isOpened()
    if not info["opened"]:
        cap.release(); return info

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    four= int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    dur = (n / fps) if (fps and n) else None
    fourcc_str = "".join([chr((four >> 8*i) & 0xFF) for i in range(4)]) if four else None

    cap.release()
    info.update({
        "fps": float(fps) if fps else None,
        "width": w or None,
        "height": h or None,
        "frame_count": n or None,
        "duration_sec_by_header": float(dur) if dur else None,
        "fourcc": fourcc_str
    })
    return info

def probe_timeline(path: str, max_frames: int = 200) -> Dict[str, Any]:
    """
    POS_MSEC zaman damgalarını örnekler; CFR/VFR ayrımı ve gerçek dt analizi için.
    """
    result = {
        "sampled_frames": 0, "pos_msec_first_10": [],
        "avg_dt_ms": None, "std_dt_ms": None, "min_dt_ms": None, "max_dt_ms": None,
    }
    if not os.path.exists(path): return result
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): cap.release(); return result

    times_ms: List[float] = []
    frames_read = 0
    while frames_read < max_frames:
        ret, _ = cap.read()
        if not ret: break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        times_ms.append(float(t_ms))
        frames_read += 1
    cap.release()

    result["sampled_frames"] = len(times_ms)
    result["pos_msec_first_10"] = times_ms[:10]
    if len(times_ms) >= 2:
        dts = [times_ms[i] - times_ms[i-1] for i in range(1, len(times_ms))]
        s = pd.Series(dts)
        result.update({
            "avg_dt_ms": float(s.mean()),
            "std_dt_ms": float(s.std(ddof=0)),
            "min_dt_ms": float(s.min()),
            "max_dt_ms": float(s.max()),
        })
    return result

def main():
    info = get_video_info(VIDEO_PATH)
    meta_df = pd.DataFrame([info])
    meta_df.to_csv(META_CSV, index=False)

    tl = probe_timeline(VIDEO_PATH, MAX_PROBE_FRAMES)
    tl_df = pd.DataFrame([{"path": VIDEO_PATH, **tl}])
    tl_df.to_csv(TIMELINE_CSV, index=False)

    print("=== HEADER METADATA ===")
    for k,v in info.items(): print(f"{k}: {v}")
    print("\n=== TIMELINE PROBE ===")
    for k,v in tl.items(): print(f"{k}: {v}")

    # Zaman eksenini öner: CFR mi VFR mı?
    fps = info.get("fps") or 0
    std_dt = (tl.get("std_dt_ms") or 0)
    avg_dt = (tl.get("avg_dt_ms") or 0)
    if fps > 0 and avg_dt > 0:
        target_dt = 1000.0 / fps
        rel_jitter = abs(std_dt) / target_dt if target_dt else 0
        print("\nJitter (std_dt / ideal_dt):", round(rel_jitter, 3))
        if rel_jitter < 0.05:
            print(">> Zaman ekseni önerisi: CFR kabul edilebilir. t = frame_idx / fps kullan.")
        else:
            print(">> Zaman ekseni önerisi: VFR olası. t = POS_MSEC/1000 kullan.")
    else:
        print("\n>> FPS ya da time probe uygun değil; POS_MSEC üzerinden zaman kullanmanı öneririm.")

if __name__ == "__main__":
    main()
