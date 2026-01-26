# ===== PENDULUM PROJESİ SUNUM MODU =====
Write-Host ">>> PENDULUM PROJESİ BAŞLATILIYOR..." -ForegroundColor Cyan

# 1. Sanal Ortamı Kontrol Et ve Aktif Et
if (-not (Get-Command "python" -ErrorAction SilentlyContinue).Source.Contains(".venv")) {
    Write-Host "-> Sanal ortam aktif ediliyor..." -ForegroundColor Yellow
    & .\.venv\Scripts\Activate.ps1
}

# Hata olursa durması için
$ErrorActionPreference = 'Stop'

# 2. Mevcut çıktıları temizle (Temiz bir başlangıç için)
Write-Host "-> Eski çıktılar temizleniyor..." -ForegroundColor Yellow
Remove-Item .\outputs\overlay_sunum.avi -ErrorAction SilentlyContinue
Remove-Item .\outputs\plots\sunum_grafik.png -ErrorAction SilentlyContinue

# 3. Nesne Tespiti (YOLOv8)
Write-Host "-> [1/5] Nesne Tespiti yapılıyor (YOLOv8)..." -ForegroundColor Green
python .\yolo_infer.py --video .\data\videos\video.mp4 --out .\outputs\detections_video.csv --conf 0.30 --imgsz 640

# 4. Takip ve Hız Hesabı
Write-Host "-> [2/5] Hareket Takibi ve Hız Hesaplanıyor..." -ForegroundColor Green
python .\track_and_export.py --detections .\outputs\detections_video.csv --out .\outputs\track_video.csv --ema 0.2 --interp_limit 5

# 5. Görselleştirme (Video Oluşturma)
Write-Host "-> [3/5] Sonuç Videosu Oluşturuluyor..." -ForegroundColor Green
# Hızlı olması için limit 300 kare ile sınırladık, tüm video için --limit'i kaldırabilirsin.
python .\render_overlay.py --video .\data\videos\video.mp4 --track .\outputs\track_video.csv --out .\outputs\overlay_sunum.avi --codec XVID --trail 40 --limit 300

# 6. Gelecek Tahmini (LSTM)
Write-Host "-> [4/5] Yapay Zeka (LSTM) Gelecek Tahmini Yapıyor..." -ForegroundColor Green
python .\forecast.py --track .\outputs\track_video.csv --model .\outputs\models\lstm_xy.pt --out .\outputs\forecast_video.csv

# 7. Grafikler
Write-Host "-> [5/5] Analiz Grafikleri Çiziliyor..." -ForegroundColor Green
python .\plot_forecast.py --track .\outputs\track_video.csv --forecast .\outputs\forecast_video.csv --out .\outputs\plots\sunum_grafik.png

# 8. Sonuçları Ekrana Aç
Write-Host ">>> İŞLEM TAMAMLANDI! Sonuçlar açılıyor..." -ForegroundColor Cyan
Start-Process .\outputs\overlay_sunum.avi
Start-Process .\outputs\plots\sunum_grafik.png