import os
import random
import shutil
import cv2
from ultralytics import YOLO

def run_auto_labeling(video_path, model_path, conf=0.1):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # [DEĞİŞİKLİK B] Session 2 klasörleri
    img_dir = "data/auto_labels/session_2/images"
    lbl_dir = "data/auto_labels/session_2/labels"
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, conf=conf, verbose=False)
        if len(results[0].boxes) > 0:
            # [DEĞİŞİKLİK C] Dosya isimleri 'hizli_' ile başlasın
            img_name = f"hizli_{count}.jpg"
            lbl_name = f"hizli_{count}.txt"
            
            cv2.imwrite(os.path.join(img_dir, img_name), frame)
            
            # Sınıfı 0 (Sarkaç) olarak zorla ve kaydet
            with open(os.path.join(lbl_dir, lbl_name), 'w') as f:
                for box in results[0].boxes:
                    # Koordinatları al (xywh formatında normalize edilmiş)
                    xywh = box.xywhn[0].tolist()
                    f.write(f"0 {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n")
        count += 1
    cap.release()
    print(f"✅ Etiketleme bitti. Tüm nesneler 'Sarkaç (0)' olarak kaydedildi.")

def distribute_data():
    # [DEĞİŞİKLİK D] Dağıtım kaynağı Session 2 olsun
    src_images = "data/auto_labels/session_2/images"
    src_labels = "data/auto_labels/session_2/labels"
    
    if not os.path.exists(src_images):
        print("❌ Hata: Kaynak resim klasörü bulunamadı!")
        return

    all_imgs = [f for f in os.listdir(src_images) if f.endswith('.jpg')]
    random.shuffle(all_imgs)
    
    total = len(all_imgs)
    if total == 0:
        print("❌ Hata: Etiketlenecek resim bulunamadı. Lütfen videoyu ve modeli kontrol edin.")
        return

    train_count = int(total * 0.82)
    val_count = int(total * 0.09)
    
    # Hedef klasörleri hazırlama (Bunlar sabit kalır, ortak havuz)
    for t in ["train", "val", "test"]:
        os.makedirs(f"data/images/{t}", exist_ok=True)
        os.makedirs(f"data/labels/{t}", exist_ok=True)

    for i, img_name in enumerate(all_imgs):
        label_name = img_name.replace('.jpg', '.txt')
        target = "train" if i < train_count else ("val" if i < (train_count + val_count) else "test")
            
        shutil.move(os.path.join(src_images, img_name), f"data/images/{target}/{img_name}")
        if os.path.exists(os.path.join(src_labels, label_name)):
            shutil.move(os.path.join(src_labels, label_name), f"data/labels/{target}/{label_name}")

    print(f"✅ {total} adet dosya başarıyla paylaştırıldı.")

if __name__ == "__main__":
    # AYARLAR:
    # [DEĞİŞİKLİK A] Yeni video yolu
    VIDEO_PATH = "data/videos/hizli_video.mp4" 
    
    # İpucu: Eğer daha önceki eğitiminden çıkan 'best.pt' varsa buraya onun yolunu yazabilirsin.
    # Yoksa 'yolov8s.pt' ile devam et.
    MODEL_PATH = "yolov8s.pt" 
    
    # ÖNCE Etiketle
    run_auto_labeling(VIDEO_PATH, MODEL_PATH, conf=0.1)
    
    # SONRA Dağıt
    distribute_data()