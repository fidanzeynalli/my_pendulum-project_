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
    
    # --- [YENİ EKLENEN CHECKBOX] ---
    use_advanced_overlay = st.checkbox("🔮 Show LSTM Future Prediction (Overlay)", value=False)
    
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

                # 1. YOLOv8 Nesne Tespiti
                st.write("🎯 YOLOv8 Özel Model ile Nesne Tespiti...")
                subprocess.run([
                    py_path, "yolo_infer.py", 
                    "--video", str(temp_video_path), 
                    "--model", "runs/detect/train5/weights/best.pt",
                    "--conf", "0.25",
                    "--out", "outputs/temp_track.csv"
                ], check=True)
                
                # 2. LSTM Tahmini (Standart)
                st.write("📈 LSTM Tahmini (Gelecek Konum Öngörüsü)...")
                subprocess.run([
                    py_path, "forecast.py", 
                    "--track", "outputs/temp_track.csv", 
                    "--model", "outputs/models/lstm_best_multi.pt", # [GÜNCELLENDİ] Yeni çoklu eğitim modeli
                    "--out", "outputs/temp_fc.csv", 
                    "--horizon", "5"
                ], check=True)
                
                # 3. Yön Analizi (Standart)
                st.write("↔️ Dinamik Yön Belirleniyor (Sağ/Sol)...")
                subprocess.run([
                    py_path, "add_direction_to_forecast.py", 
                    "--forecast", "outputs/temp_fc.csv", 
                    "--out", "outputs/temp_fc_dir.csv", 
                    "--threshold", "0.00001"
                ], check=True)
                
                # 4. Görselleştirme (Standart)
                st.write("🎬 Final Videosu Oluşturuluyor...")
                final_out = "outputs/FINAL_APP_OUTPUT.mp4" 

                subprocess.run([
                    py_path, "render_overlay_with_direction.py", 
                    "--video", str(temp_video_path), 
                    "--track", "outputs/temp_track.csv", 
                    "--forecast", "outputs/temp_fc_dir.csv", 
                    "--out", final_out,
                    "--codec", "mp4v"
                ], check=True)

                st.write("🔄 Tarayıcı uyumluluğu için video dönüştürülüyor...")
                compatible_out = "outputs/FINAL_COMPATIBLE.mp4"

                try:
                    subprocess.run([
                        "ffmpeg", "-y", "-i", final_out, 
                        "-vcodec", "libx264", "-f", "mp4", 
                        compatible_out
                    ], check=True)
                    final_out = compatible_out 
                except Exception as e:
                    st.error(f"FFmpeg hatası: {e}")
                
                # --- [YENİ BLOK] GELİŞMİŞ OVERLAY İŞLEMİ ---
                if use_advanced_overlay:
                    st.write("🔮 Gelişmiş Gelecek Tahmini ve Yön Analizi Hazırlanıyor...")
                    
                    # 1. Gelişmiş Tahmin Scriptini Çağır
                    subprocess.run([
                        py_path, "forecast_augmented.py",
                        "--track", "outputs/temp_track.csv",
                        "--model", "outputs/models/lstm_best_multi.pt", # [GÜNCELLENDİ] Yeni çoklu eğitim modeli
                        "--out", "outputs/forecast_aug.csv",
                        "--horizon", "5"
                    ], check=True)
                    
                    # 2. Gelişmiş Render Scriptini Çağır
                    aug_video_out = "outputs/FINAL_AUGMENTED.mp4"
                    subprocess.run([
                        py_path, "render_overlay_future.py",
                        "--video", str(temp_video_path),
                        "--track", "outputs/temp_track.csv",
                        "--forecast", "outputs/forecast_aug.csv",
                        "--out", aug_video_out
                    ], check=True)
                    
                    # 3. FFmpeg ile Convert
                    compatible_aug = "outputs/FINAL_AUGMENTED_WEB.mp4"
                    try:
                        subprocess.run([
                            "ffmpeg", "-y", "-i", aug_video_out,
                            "-vcodec", "libx264", "-f", "mp4",
                            compatible_aug
                        ], check=True)
                        final_out = compatible_aug 
                    except Exception as e:
                        st.error(f"Gelişmiş video için FFmpeg hatası: {e}")

                status.update(label="Analiz Başarıyla Tamamlandı!", state="complete")

            # --- VİDEO VE GRAFİK GÖSTERİMİ ---
            if os.path.exists(final_out):
                st.subheader("📽️ Analiz Edilmiş Video")
                with open(final_out, 'rb') as f:
                    st.video(f.read())
            
            # --- [BURASI ENTEGRE EDİLDİ] ---
            if use_advanced_overlay and os.path.exists("outputs/forecast_aug.csv"):
                st.divider()
                st.subheader("📈 LSTM Gelecek Tahmini Analizi")
                
                df_aug = pd.read_csv("outputs/forecast_aug.csv")
                
                st.markdown("**1. Tahmin Edilen X ve Y Konumları (Piksel)**")
                chart_data = df_aug.set_index("frame_idx")[["pred_cx", "pred_cy"]]
                st.line_chart(chart_data)
                
                st.markdown("**2. Tahmin Edilen Hareket Hızı (Piksel/Horizon)**")
                st.area_chart(df_aug.set_index("frame_idx")["speed_px"], color="#FF4B4B")

                with st.expander("📄 Detaylı Veri Tablosunu Görüntüle"):
                    st.dataframe(df_aug)
            
            # Eski mod çıktıları (Gelişmiş mod kapalıysa görünür)
            elif os.path.exists("outputs/temp_fc_dir.csv"):
                st.subheader("📊 Sayısal Analiz Verileri (Son 10 Kare)")
                df = pd.read_csv("outputs/temp_fc_dir.csv")
                st.dataframe(df.tail(10))
            
        except Exception as e:
            st.error(f"Hata detayı: {e}")