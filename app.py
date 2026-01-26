import streamlit as st
import subprocess
import os
import sys
import pandas as pd
from pathlib import Path
import static_ffmpeg
static_ffmpeg.add_paths()

# Arayüz Konfigürasyonu
st.set_page_config(page_title="Sarkaç Analiz Sistemi", layout="wide")
st.title(" Sarkaç Hareket ve Yön Analiz Uygulaması")

# Çalışma dizini ayarı
current_dir = Path(__file__).parent.absolute()
os.chdir(current_dir)

# Klasörlerin varlığından emin ol
(current_dir / "data" / "videos").mkdir(parents=True, exist_ok=True)
(current_dir / "outputs").mkdir(parents=True, exist_ok=True)
(current_dir / "outputs" / "models").mkdir(parents=True, exist_ok=True) # Model klasörü eklendi

# Dosya yükleme alanı
with st.sidebar:
    st.header("⚙️ Ayarlar")
    uploaded_file = st.file_uploader("Video Dosyası Seçin", type=["mp4", "avi", "mov"])
    process_btn = st.button("Analizi Başlat")

if uploaded_file is not None:
    temp_video_path = current_dir / "data" / "videos" / "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.info("Video yüklendi. İşlem başlatılmaya hazır.")

    if process_btn:
        try:
            with st.status("🔍 Analiz Yürütülüyor...", expanded=True) as status:
                py_path = sys.executable

                # --- GÜÇLENDİRİLMİŞ İŞLEM MOTORU BAŞLANGICI ---

                # 1. YOLOv8 Nesne Tespiti (Özel Model ve Yüksek Hassasiyet Entegrasyonu)
                # Görseldeki %91 güven oranını (conf=0.25) ve Tesla T4 hızını (device=0) kullanır.
                st.write("🎯 YOLOv8 Özel Model ile Nesne Tespiti...")
                subprocess.run([
                    py_path, "yolo_infer.py", 
                    "--video", str(temp_video_path), 
                    "--model", "runs/detect/train5/weights/best.pt", # Senin eğittiğin en iyi ağırlık
                    "--conf", "0.25",                      # Güven eşiği ayarı
                    "--out", "outputs/temp_track.csv"
                ], check=True)
                
                # 2. LSTM Tahmini
                st.write("📈 LSTM Tahmini (Gelecek Konum Öngörüsü)...")
                subprocess.run([
                    py_path, "forecast.py", 
                    "--track", "outputs/temp_track.csv", 
                    "--model", "outputs/models/lstm_xy.pt", 
                    "--out", "outputs/temp_fc.csv", 
                    "--horizon", "5"
                ], check=True)
                
                # 3. Yön Analizi
                st.write("↔️ Dinamik Yön Belirleniyor (Sağ/Sol)...")
                subprocess.run([
                    py_path, "add_direction_to_forecast.py", 
                    "--forecast", "outputs/temp_fc.csv", 
                    "--out", "outputs/temp_fc_dir.csv", 
                    "--threshold", "0.00001"
                ], check=True)
                
                # 4. Görselleştirme (Tahmin ve Tespit Yazılarının Videoya İşlenmesi)
                st.write("🎬 Final Videosu Oluşturuluyor...")
                # .avi yerine .mp4 yapıyoruz
                final_out = "outputs/FINAL_APP_OUTPUT.mp4" 

                subprocess.run([
                    py_path, "render_overlay_with_direction.py", 
                    "--video", str(temp_video_path), 
                    "--track", "outputs/temp_track.csv", 
                    "--forecast", "outputs/temp_fc_dir.csv", 
                    "--out", final_out,
                    "--codec", "mp4v"  # mp4v tarayıcı uyumluluğu için daha iyidir
                ], check=True)

                st.write("🔄 Tarayıcı uyumluluğu için video dönüştürülüyor...")
                compatible_out = "outputs/FINAL_COMPATIBLE.mp4"

                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", final_out, 
                        "-vcodec", "libx264", "-f", "mp4", 
                        compatible_out
                    ], check=True)
                    final_out = compatible_out # Artık Streamlit bu dosyayı gösterecek
                except Exception as e:
                    st.error(f"FFmpeg hatası: {e}. Bilgisayarınızda FFmpeg kurulu olduğundan emin olun.")
                # --- GÜÇLENDİRİLMİŞ İŞLEM MOTORU SONU ---
                
                status.update(label="Analiz Başarıyla Tamamlandı!", state="complete")

            # Analiz Sonuçlarını Göster
            if os.path.exists(final_out):
                st.subheader("📽️ Analiz Edilmiş Video (Tahmin ve Yön Bilgisi)")
                with open(final_out, 'rb') as f:
                    st.video(f.read())
            
            if os.path.exists("outputs/temp_fc_dir.csv"):
                st.subheader("📊 Sayısal Analiz Verileri (Son 10 Kare)")
                df = pd.read_csv("outputs/temp_fc_dir.csv")
                st.dataframe(df.tail(10))
            
        except Exception as e:
            st.error(f"Hata detayı: {e}")