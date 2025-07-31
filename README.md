# Signature & Stamp Segmentation Project (YOLOv11)

Bu proje, **imza (signature) ve kaşe (stamp) segmentasyonu** için YOLOv11 tabanlı derin öğrenme modelinin geliştirilmesini ve test edilmesini içerir.

## 📂 Proje Akışı

1. **Veri Seti Hazırlığı**

   - Roboflow'dan YOLOv11 formatında indirilen veri seti kullanıldı. [Roboflow](https://universe.roboflow.com/sig-and-stamps/sig-stamps)
   - Kaggle Datasets üzerine yüklenerek online ve GPU destekli ortamda kolayca eğitim/test yapıldı. [Kaggle Dataset](https://www.kaggle.com/datasets/mertaslayilmaz/signature-stamp-segmentation-dataset) / [Kaggle Notebook](https://www.kaggle.com/code/mertaslayilmaz/signature-stamp-segmentation-yolov11)

2. **Model Eğitimi**

   - Ultralytics YOLO framework ile segmentasyon modeli (`yolov11s-seg.pt`) transfer learning olarak eğitildi.
   - Eğitimde **ılımlı augmentation (veri artırma)** teknikleri kullanıldı (döndürme, zoom, flip, renk, mosaic vb.).
   - Eğitim çıktıları: `best.pt` (en iyi model), eğitim metrikleri ve loss/precision/mAP grafik dosyaları.

3. **Eğitim Sonrası Analiz**

   - Eğitim metrikleri (`results.csv` ve `results.png`) görselleştirilerek modelin başarısı kontrol edildi.
   - Overfitting/underfitting ve metriklerin genel eğilimi analiz edildi.

4. **Test ve İnference (Tahmin)**

   - Test görselleri model üzerinde çalıştırıldı.
   - Tahmin sonuçları görsel olarak notebook ortamında analiz edildi.
   - Modelin segmentasyon çıktıları doğrudan görselleştirildi.

5. **İmza ve Kaşe Crop & Kolaj**

   - Tespit edilen imza ve kaşeler her belge için **otomatik olarak crop’landı**.
   - Her test belgesi için: Orijinal belge + tüm imza ve kaşe crop'ları **yan yana küçük görseller şeklinde kolaj** olarak gösterildi.

6. **Başarısız (Tespit Edilemeyen) Örneklerin Analizi**

   - Modelin imza veya kaşe bulamadığı test belgeleri otomatik olarak ayrı bir klasöre kaydedildi.
   - Bu örnekler notebook’ta küçük görseller halinde hızlıca incelendi.

7. **Modelin Kaydedilmesi ve Yeniden Kullanımı**
   - Eğitim tamamlandıktan sonra **en iyi ağırlık dosyası (`best.pt`) indirildi**.
   - Farklı bir ortamda (Colab, local, server) aynı model dosyasıyla kolayca tekrar tahmin, test ve analiz yapılabilir.

## 🚀 Kullanım ve Çalıştırma

### 1. Modeli Eğitmek

```python
results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=640,
    batch=32,
    task='segment',
    degrees=5,
    scale=0.5,
    shear=1,
    perspective=0.0005,
    flipud=0.1,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    mosaic=1.0,
    mixup=0.1,
    project='/kaggle/working/yolo_output',
    name='full_runS_aug'
)
```

### 2. Eğitim Sonrası Analiz

- `results.csv` ve `results.png` dosyaları ile metrikler ve eğitim eğrileri analiz edildi.
- Kodda otomatik görselleştirme yapıldı.

### 3. Test ve Kolaj

- Test görsellerinde model tahmini alındı.
- İmza ve kaşe crop’ları yan yana küçük görseller halinde kolaj yapıldı.
- Başarısız tespitler otomatik kaydedilip görselleştirildi.

### 4. Modelin Yeniden Kullanımı

- Sadece `best.pt` dosyası indirildikten sonra, başka bir ortamda aynı şekilde yüklenip inference yapılabilir:

```python
from ultralytics import YOLO
model = YOLO('best.pt')
results = model('test_image.jpg')
```

---
