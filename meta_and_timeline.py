from pathlib import Path
import os, cv2, pandas as pd
from typing import Dict, Any, List

# --- DİZİN AYARLARI ---
# BASE_DIR: Kodun çalıştığı klasörü kök dizin olarak belirler (Göreceli yollar için güvenli yöntem).
BASE_DIR = Path(__file__).resolve().parent

# Analiz edilecek videonun yolu ve örnekleme sayısı.
VIDEO_PATH = BASE_DIR / "data" / "videos" / "video.mp4" 
MAX_PROBE_FRAMES = 300 # Analiz için taranacak maksimum kare sayısı.

# Çıktı klasörlerini tanımlar ve yoksa oluşturur.
OUT_DIR = BASE_DIR / "outputs"
META_CSV = OUT_DIR / "video_metadata.csv" # Video başlık bilgilerinin kaydedileceği dosya.
TIMELINE_CSV = OUT_DIR / "video_timeline_probe.csv" # Zamanlama analizinin kaydedileceği dosya.

OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- FONKSİYON 1: VİDEO ÜST VERİLERİNİ (METADATA) AL ---
def get_video_info(path: str) -> Dict[str, Any]:
    """
    Videonun başlık (header) kısmındaki genel teknik bilgileri okur.
    """
    info = {
        "path": path, "exists": os.path.exists(path),
        "opened": False, "fps": None, "width": None, "height": None,
        "frame_count": None, "duration_sec_by_header": None, "fourcc": None,
    }
    if not info["exists"]: return info

    cap = cv2.VideoCapture(path)
    info["opened"] = cap.isOpened()
    if not info["opened"]:
        cap.release(); return info

    # OpenCV özelliklerini (Properties) kullanarak verileri çeker.
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0          # Kare hızı
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)  # Genişlik
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) # Yükseklik
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)  # Toplam kare sayısı
    four= int(cap.get(cv2.CAP_PROP_FOURCC) or 0)       # Video çözücü (Codec) kodu
    
    # Süre hesaplaması (Kare Sayısı / FPS).
    dur = (n / fps) if (fps and n) else None
    
    # FourCC kodunu okunabilir metne çevirir (Örn: 'XVID', 'H264').
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

# --- FONKSİYON 2: ZAMAN AKIŞI ANALİZİ (TIMELINE PROBE) ---
def probe_timeline(path: str, max_frames: int = 200) -> Dict[str, Any]:
    """
    Videonun gerçek zaman damgalarını (POS_MSEC) örnekleyerek;
    CFR (Sabit Kare Hızı) veya VFR (Değişken Kare Hızı) ayrımı için istatistik çıkarır.
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
        # Her bir karenin gerçek milisaniye değerini alır.
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        times_ms.append(float(t_ms))
        frames_read += 1
    cap.release()

    result["sampled_frames"] = len(times_ms)
    result["pos_msec_first_10"] = times_ms[:10]
    
    # Kareler arası süre farklarını (dt) analiz eder.
    if len(times_ms) >= 2:
        dts = [times_ms[i] - times_ms[i-1] for i in range(1, len(times_ms))]
        s = pd.Series(dts)
        result.update({
            "avg_dt_ms": float(s.mean()),          # Ortalama bekleme süresi
            "std_dt_ms": float(s.std(ddof=0)),     # Standart sapma (Zamanlama tutarlılığı)
            "min_dt_ms": float(s.min()),
            "max_dt_ms": float(s.max()),
        })
    return result

# --- ANA ÇALIŞTIRMA BETİĞİ ---
def main():
    # 1. Adım: Genel video bilgilerini al ve CSV'ye kaydet.
    info = get_video_info(VIDEO_PATH)
    meta_df = pd.DataFrame([info])
    meta_df.to_csv(META_CSV, index=False)

    # 2. Adım: Zamanlama probe analizini yap ve kaydet.
    tl = probe_timeline(VIDEO_PATH, MAX_PROBE_FRAMES)
    tl_df = pd.DataFrame([{"path": VIDEO_PATH, **tl}])
    tl_df.to_csv(TIMELINE_CSV, index=False)

    # Sonuçları terminale yazdır.
    print("=== HEADER METADATA ===")
    for k,v in info.items(): print(f"{k}: {v}")
    print("\n=== TIMELINE PROBE ===")
    for k,v in tl.items(): print(f"{k}: {v}")

    # 3. Adım: Zaman Ekseni Kararı (CFR mi VFR mi?)
    # Jitter (titreme) analizi yaparak en doğru zamanlama yöntemini önerir.
    fps = info.get("fps") or 0
    std_dt = (tl.get("std_dt_ms") or 0)
    avg_dt = (tl.get("avg_dt_ms") or 0)
    
    if fps > 0 and avg_dt > 0:
        target_dt = 1000.0 / fps # Olması gereken ideal süre
        rel_jitter = abs(std_dt) / target_dt if target_dt else 0
        print("\nJitter (Standart Sapma / İdeal Süre):", round(rel_jitter, 3))
        
        # Eğer zamanlama hatası %5'ten küçükse Sabit Kare Hızı (CFR) kullanılabilir.
        if rel_jitter < 0.05:
            print(">> Öneri: CFR (Sabit) hız uygun. t = frame_idx / fps kullanın.")
        else:
            print(">> Öneri: VFR (Değişken) hız algılandı. t = POS_MSEC/1000 kullanın.")
    else:
        print("\n>> Öneri: Video bilgileri yetersiz; POS_MSEC kullanılması daha güvenlidir.")

if __name__ == "__main__":
    main()