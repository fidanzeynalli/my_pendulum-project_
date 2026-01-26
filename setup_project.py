import os
import shutil
import random
from pathlib import Path

# Ana dizin ayarı
base_path = Path("data")
images_path = base_path / "images"
labels_path = base_path / "labels"

# Gerekli alt klasör listesi
sub_folders = ["train", "val", "test"]

# Klasörleri oluştur
for folder in sub_folders:
    (images_path / folder).mkdir(parents=True, exist_ok=True)
    (labels_path / folder).mkdir(parents=True, exist_ok=True)

print("✅ Klasör yapısı (train/val/test) başarıyla oluşturuldu.")