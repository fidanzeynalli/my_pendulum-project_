import pandas as pd    
import numpy as np    
import argparse       

def main():
    #  Dışarıdan hangi dosyaların geleceğini tanımlıyoruz
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", required=True)  # Giriş: LSTM'den çıkan tahmin dosyası
    ap.add_argument("--out", required=True)       # Çıkış: Yön etiketli yeni dosyanın yolu
    ap.add_argument("--threshold", type=float, default=0.5) # Hassasiyet: Hareketin yön sayılması için gereken min. değişim
    args = ap.parse_args()

    # VERİ OKUMA: Tahmin dosyasını hafızaya yüklüyoruz
    df = pd.read_csv(args.forecast)

    #  MATEMATİKSEL ANALİZ (Türevsel Yaklaşım)
    # 'x_pred' sütunundaki ardışık değerler arasındaki farkı (dx) hesaplıyoruz.
    # .diff() farkı alır, .shift(-1) ise sonucu bir satır yukarı kaydırarak mevcut kare ile gelecek kareyi kıyaslar.
    df['dx'] = df['x_pred'].diff().shift(-1)

    #  YÖN KARAR MEKANİZMASI
    # dx (değişim miktarı) pozitifse sağa, negatifse sola hareket var demektir.
    conditions = [
        (df['dx'] > args.threshold),  # Değişim eşik değerinden büyükse SAĞA
        (df['dx'] < -args.threshold)  # Değişim eşik değerinin negatifinden küçükse SOLA
    ]
    
    # Koşullara karşılık gelen etiketler
    choices = ["SAGA GIDIYOR", "SOLA GIDIYOR"]

    #  ETİKETLEME Eğer yukarıdaki iki koşul da sağlanmıyorsa (hareket çok azsa) "DURAGAN" kabul et.
    df['dir_label'] = np.select(conditions, choices, default="DURAGAN")

    # KAYIT: Yön bilgilerinin eklendiği yeni CSV dosyasını oluşturuyoruz.
    df.to_csv(args.out, index=False)
    print(f"[ok] Yon etiketleri eklendi: {args.out}")

if __name__ == "__main__":
    main()