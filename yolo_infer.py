from pathlib import Path

import argparse, os, cv2, pandas as pd, numpy as np

from ultralytics import YOLO



def probe_timeline(path: Path, max_frames: int = 200):

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():

        cap.release()

        return None

    times = []

    i = 0

    while i < max_frames:

        ok, _ = cap.read()

        if not ok: break

        # Her karenin milisaniye bazında gerçek zaman damgasını alır

        times.append(cap.get(cv2.CAP_PROP_POS_MSEC))

        i += 1

    cap.release()

    if len(times) < 2:

        return None

    dts = np.diff(times)# Kareler arasındaki süre farklarını hesaplar

    return {

        "avg_dt_ms": float(np.mean(dts)),

        "std_dt_ms": float(np.std(dts)),

        "min_dt_ms": float(np.min(dts)),

        "max_dt_ms": float(np.max(dts))

    }



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--video", type=str, default=str(Path("data/videos/video.mp4")),

                        help="Input video path")# Giriş videosu

    parser.add_argument("--model", type=str, default="yolov8s.pt",

                        help="YOLOv8 model path (pt)")

    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    parser.add_argument("--class_id", type=int, default=None,

                        help="Keep only this class id (None=keep best box overall)")

    parser.add_argument("--out", type=str, default=str(Path("outputs/detections.csv")),

                        help="Output CSV path")

    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")

    args = parser.parse_args()



    video_path = Path(args.video).resolve()

    out_csv = Path(args.out).resolve()

    out_csv.parent.mkdir(parents=True, exist_ok=True)



    # Header info

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        raise RuntimeError(f"Cannot open video: {video_path}")

    fps  = cap.get(cv2.CAP_PROP_FPS) or 0.0

    W    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    H    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    cap.release()



    # Timeline probe (CFR/VFR önerisi)

    probe = probe_timeline(video_path, max_frames=200) or {}

    time_mode = "cfr"

    if fps and probe.get("avg_dt_ms"):

        ideal = 1000.0 / fps

        jitter = (probe["std_dt_ms"] / ideal) if ideal else 0.0

        if jitter >= 0.05:

            time_mode = "vfr"

    else:

        time_mode = "vfr"



    model = YOLO(args.model)



    rows = []

    cap = cv2.VideoCapture(str(video_path))

    frame_idx = 0

    while True:

        ret, frame = cap.read()

        if not ret: break



        # t (saniye)

        if time_mode == "vfr":

            t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0

        else:

            t = frame_idx / fps if fps > 0 else None



        res = model.predict(source=frame, conf=args.conf, imgsz=args.imgsz, verbose=False)

        boxes = res[0].boxes

        best = None



        if boxes is not None and len(boxes) > 0:

            # class filtresi varsa uygula

            idxs = list(range(len(boxes)))

            if args.class_id is not None and boxes.cls is not None:

                idxs = [i for i in idxs if int(boxes.cls[i].item()) == args.class_id]

            if idxs:

                # en yüksek skorlu

                confs = [boxes.conf[i].item() for i in idxs]

                best_i = idxs[int(np.argmax(confs))]

                b = boxes[best_i]

                x1,y1,x2,y2 = b.xyxy[0].tolist()

                conf = float(b.conf[0].item())

                cls  = int(b.cls[0].item()) if b.cls is not None else None

                cx = (x1 + x2) / 2.0

                cy = (y1 + y2) / 2.0

                rows.append({

                    "frame": frame_idx, "t": t,

                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,

                    "cx": cx, "cy": cy, "conf": conf, "cls": cls,

                    "width": W, "height": H, "fps": fps, "time_mode": time_mode

                })

            else:

                rows.append({"frame": frame_idx, "t": t,

                             "x1": None,"y1": None,"x2": None,"y2": None,

                             "cx": None,"cy": None,"conf": None,"cls": None,

                             "width": W, "height": H, "fps": fps, "time_mode": time_mode})

        else:

            rows.append({"frame": frame_idx, "t": t,

                         "x1": None,"y1": None,"x2": None,"y2": None,

                         "cx": None,"cy": None,"conf": None,"cls": None,

                         "width": W, "height": H, "fps": fps, "time_mode": time_mode})

        frame_idx += 1



    cap.release()

    df = pd.DataFrame(rows)

    df.to_csv(out_csv, index=False)

    print(f"Saved detections → {out_csv}")

    if probe:

        print("Timeline probe:", probe)

        print("Time mode:", time_mode)



if __name__ == "__main__":

    main()