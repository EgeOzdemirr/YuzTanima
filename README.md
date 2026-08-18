# YuzTanima

Kamera görüntüsünden gerçek zamanlı yüz tespiti, tanıma ve 3D yüz modeli üretimi yapan
Windows tabanlı bir sistem.

**Hat:** InsightFace SCRFD (tespit) → AdaFace ONNX (512-boyutlu embedding) →
FAISS (hızlı arama) → ByteTrack (takip) → web arayüzü.
3D yüz modelleri 3DDFA-V3 ile üretilip GLB olarak sunulur.

> **Not:** Bu depo yalnızca **kaynak kodu** içerir. Model ağırlıkları ve fotoğraflar
> depoda yoktur — modeller aşağıdaki komutlarla indirilir, fotoğrafları ise
> kendiniz sağlarsınız. Ayrıntı için [Depoda bulunmayanlar](#depoda-bulunmayanlar).

---

## Gereksinimler

| | |
|---|---|
| İşletim sistemi | Windows 10/11 (64-bit) |
| Python | 3.13 ([indir](https://www.python.org/downloads/)) — kurulumda "Add python.exe to PATH" işaretli olmalı |
| RAM | En az 8 GB (16 GB önerilir) |
| Disk | ~20 GB boş alan (model ağırlıkları ~850 MB, venv ve bağımlılıklar geri kalanı) |
| Kamera | USB veya dahili webcam (canlı tanıma için) |
| GPU *(opsiyonel)* | NVIDIA GTX 1060+, sürücü 525+, CUDA 11.8, cuDNN 8.x |

GPU yoksa sistem CPU ile çalışır, yalnızca yavaştır — bkz. [GPU / CPU modu](#gpu--cpu-modu).

---

## Kurulum

### 1. Depoyu submodule'lerle birlikte klonlayın

```bash
git clone --recurse-submodules https://github.com/EgeOzdemirr/YuzTanima.git
cd YuzTanima
```

`--recurse-submodules` olmadan klonladıysanız:

```bash
git submodule update --init --recursive
```

### 2. Sanal ortamı oluşturun

Klasör adı **`3ddfav3` olmak zorunda** — başlatıcı bu yolu sabit olarak arıyor.

```bash
python -m venv 3ddfav3
3ddfav3\Scripts\python.exe -m pip install --upgrade pip
3ddfav3\Scripts\python.exe -m pip install -r requirements.txt
```

GPU kullanacaksanız PyTorch'un CUDA sürümü ayrıca gerekebilir:
<https://pytorch.org/get-started/locally/>

### 3. Model ağırlıklarını indirin

```bash
3ddfav3\Scripts\python.exe scripts\download_models.py
```

Bu komut şunları indirir:

- `services/core/models/weights/adaface_ir_101.onnx` — AdaFace tanıma modeli (~260 MB)
- 3DDFA-V3 assets: `face_model.npy`, `net_recon.pth`, `similarity_Lm3D_all.mat`

InsightFace SCRFD dedektörü (`buffalo_l`) ayrı bir komut gerektirmez; program ilk
çalıştırıldığında `insightface` paketi eksik modelleri kendisi indirir
(internet bağlantısı gerekir).

### 4. 3DDFA-V3 NumPy 2.x yamasını uygulayın

3DDFA-V3 upstream kodu NumPy 1.x'e göre yazılmış. `requirements.txt` NumPy sürümünü
sabitlemediği için NumPy 2.x kurulur ve 3D üretimi yama olmadan **çöker**
(`np.VisibleDeprecationWarning` kaldırıldı, tek elemanlı diziye `float()` artık hata veriyor).

```bash
cd services\tddfa\repo\3DDFA-V3
git apply ..\..\..\..\patches\3ddfa-v3-numpy2-compat.patch
cd ..\..\..\..
```

Alternatif olarak `pip install "numpy<2"` ile de çözebilirsiniz, ancak yama önerilir.

### 5. Kendi fotoğraflarınızı ekleyin

Depoda fotoğraf **yoktur**. Tanınmasını istediğiniz kişilerin fotoğraflarını
proje kökünde `arananlar/` klasörü açıp içine koyun.
**Dosya adı kişinin adı olur:**

```
arananlar/
├── Ahmet YILMAZ.jpg
├── Ayşe DEMİR.png
└── ...
```

Desteklenen biçimler: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`

### 6. Galeriyi ve arama indeksini oluşturun

```bash
:: Fotograflari kisi klasorlerine donustur (public/gallery/persons/)
3ddfav3\Scripts\python.exe tools\galeri_olusturma.py

:: Embedding'leri ve FAISS indeksini uret (data/gallery/)
3ddfav3\Scripts\python.exe services\gallery_builder\build_gallery.py --config services\gallery_builder\config.yaml
```

3D avatarları da üretmek isterseniz (uzun sürer, Blender gerektirir):

```bash
3ddfav3\Scripts\python.exe services\gallery_builder\build_3d_gallery.py --config services\gallery_builder\config.yaml
```

`build_3d_gallery` için Blender yolunu `services/gallery_builder/config.yaml`
içindeki `build3d.blender_bin` alanından ayarlayın.

### 7. Çalıştırın

```bash
3ddfav3\Scripts\python.exe scripts\windows_launcher.py
```

Başlatıcı UI sunucusunu ve kamera servisini birlikte ayağa kaldırır, ardından
tarayıcıyı açar:

**<http://127.0.0.1:9000/ui/index.html>**

Durdurmak için konsol penceresinde `Ctrl+C`. (Tarayıcıyı kapatmak programı durdurmaz.)

Servisleri tek tek çalıştırmak isterseniz:

```bash
3ddfav3\Scripts\python.exe services\gallery_builder\gallery_ui_server.py
3ddfav3\Scripts\python.exe services\camera_runtime\run_camera.py --config services\camera_runtime\config.yaml
```

---

## Yapılandırma

Tüm ayarlar iki YAML dosyasında:

- [`services/camera_runtime/config.yaml`](services/camera_runtime/config.yaml) — kamera, takip, eşleştirme
- [`services/gallery_builder/config.yaml`](services/gallery_builder/config.yaml) — galeri üretimi, 3D

### Kamera seçimi

```yaml
camera:
  type: webcam          # webcam | rtsp | http_mjpeg | video_file
  webcam:
    device_index: 0     # kamera goruntusu gelmiyorsa 1, 2 ... deneyin
    backend: dshow      # sorun cikarsa msmf veya auto
```

RTSP için `type: rtsp` yapıp `rtsp.url` alanını doldurun.

### GPU / CPU modu

Her iki config dosyasında da `models.detector.device` ve `models.embedder.device`
alanlarını `cuda` (GPU) veya `cpu` olarak ayarlayın. CUDA hatası alıyorsanız `cpu` yapın.

### Tanıma hassasiyeti

```yaml
match:
  threshold: 0.55   # yukseltmek -> daha secici (az yanlis alarm)
                    # dusurmek   -> daha hassas (daha kolay eslesme)
  margin: 0.05      # top1 ile farkli kisinin en iyisi arasindaki asgari fark
```

---

## Depoda bulunmayanlar

Aşağıdakiler bilinçli olarak `.gitignore` ile dışarıda bırakılmıştır:

| Dışarıda bırakılan | Neden | Nasıl elde edilir |
|---|---|---|
| `arananlar/`, `yuz_tanima_fotoğraflar/` | **Kişisel veri** | Kendiniz sağlarsınız (Adım 5) |
| `public/gallery/`, `public/captures/` | İşlenmiş kişisel veri | Adım 6 üretir |
| `data/gallery/*` (FAISS indeksi, embeddings) | Türetilmiş veri | Adım 6 üretir |
| `data/events/` | Çalışma zamanı çıktısı | Otomatik oluşur |
| `services/core/models/weights/` (~850 MB) | GitHub 100 MB dosya sınırı | Adım 3 indirir |
| `3ddfav3/` | Sanal ortam | Adım 2 oluşturur |
| `logs/` | Çalışma zamanı logları | Otomatik oluşur |
| `Baslat.exe` | PyInstaller derleme çıktısı | Bkz. aşağıdaki not |

**`Baslat.exe` hakkında:** Derlenmiş başlatıcı depoda değildir ve derleme betiği de
bulunmamaktadır. Aynı işi `scripts\windows_launcher.py` görür (Adım 7).

> **Gizlilik uyarısı:** Bu sistem biyometrik veri (yüz embedding'leri) işler.
> Fotoğrafları ve üretilen `data/gallery/` içeriğini herkese açık bir depoya
> yüklemeyin; KVKK/GDPR kapsamında kişisel veridir.

---

## Proje yapısı

```
YuzTanima/
├── scripts/
│   ├── windows_launcher.py     # UI + kamera servislerini birlikte baslatir
│   └── download_models.py      # Model agirliklarini indirir
├── services/
│   ├── camera_runtime/         # Kamera dongusu, tanima hatti, ByteTrack, WebRTC
│   ├── core/
│   │   ├── models/             # SCRFD dedektor, AdaFace embedder, FAISS eslestirici
│   │   ├── io/                 # Galeri / olay / yakalama depolari
│   │   └── utils/              # Goruntu on isleme, kimlik yardimcilari
│   ├── gallery_builder/        # Galeri indeksi, 3D uretim, UI sunucusu
│   └── tddfa/repo/3DDFA-V3/    # 3DDFA-V3 (git submodule)
├── tools/
│   ├── galeri_olusturma.py     # Fotograflari kisi klasorlerine donusturur
│   └── 3d/                     # GLB uretimi ve multiview render araclari
├── ui/                         # Web arayuzu (HTML/JS/CSS)
├── tests/                      # Kimlik ve eslestirme testleri
├── patches/                    # 3DDFA-V3 uyumluluk yamalari
└── requirements.txt
```

---

## Sorun giderme

| Sorun | Çözüm |
|---|---|
| `Python ortami bulunamadi: 3ddfav3\Scripts\python.exe` | Sanal ortam adı tam olarak `3ddfav3` olmalı (Adım 2) |
| `9000 portu zaten dolu` | Önceki başlatıcı hâlâ çalışıyor; Görev Yöneticisi'nden sonlandırın |
| Kamera görüntüsü gelmiyor | `config.yaml` içinde `device_index` değerini 0, 1, 2... deneyin; `backend` için `msmf` veya `auto` |
| CUDA / GPU hatası | Her iki config'de `device: cpu` yapın |
| `faiss.index bulunamadı` | Galeri indeksi üretilmemiş — Adım 6'yı çalıştırın |
| 3D üretiminde NumPy hatası | Adım 4'teki yama uygulanmamış |
| Tarayıcı açılmıyor | Adresi elle girin: <http://127.0.0.1:9000/ui/index.html> |

Log dosyaları `logs/` klasöründedir; en yeni dosya en son çalıştırmaya aittir.

---

## Testler

```bash
3ddfav3\Scripts\python.exe -m pytest tests\
```

---

## Üçüncü taraf bileşenler

- [3DDFA-V3](https://github.com/wang-zidu/3DDFA-V3) — 3D yüz rekonstrüksiyonu (submodule)
- [InsightFace](https://github.com/deepinsight/insightface) — SCRFD yüz tespiti
- [AdaFace](https://github.com/mk-minchul/AdaFace) — yüz tanıma modeli
  ([ONNX dönüşümü](https://github.com/yakhyo/adaface-onnx))
- [FAISS](https://github.com/facebookresearch/faiss) — benzerlik araması
