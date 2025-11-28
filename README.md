# 🔬 Skin Cancer Classification API

[![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Run-4285F4?logo=google-cloud)](https://cancer-api-993742066618.europe-west1.run.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://www.python.org)

Machine Learning tabanlı cilt kanseri (melanoma) sınıflandırması için RESTful API. EfficientNet-B0 özellik vektörleri ile Random Forest algoritması kullanılarak %90.8 doğruluk oranı elde edilmiştir.

## 🌐 Live Demo

**API URL:** https://cancer-api-993742066618.europe-west1.run.app

**Swagger UI (İnteraktif Dokümantasyon):** https://cancer-api-993742066618.europe-west1.run.app/docs

## 📊 Model Performansı

- **Accuracy:** 90.80%
- **Precision (Benign):** 89%
- **Precision (Malignant):** 93%
- **Recall (Benign):** 93%
- **Recall (Malignant):** 89%

### Confusion Matrix
```
              Predicted
              Benign  Malignant
Actual Benign    464      36
      Malignant   56     444
```

## 🚀 Özellikler

- ✅ **FastAPI** ile yüksek performanslı REST API
- ✅ **Docker** containerization
- ✅ **Google Cloud Run** deployment
- ✅ **Otomatik API dokümantasyonu** (Swagger UI)
- ✅ **1000 özellik** ile tahmin
- ✅ **Binary classification** (benign/malignant)
- ✅ **Olasılık skorları** ile güven seviyesi

## 🏗️ Teknoloji Stack

- **Backend Framework:** FastAPI 0.104.1
- **ML Library:** scikit-learn 1.3.2
- **Model:** Random Forest Classifier (100 estimators)
- **Feature Extraction:** EfficientNet-B0 (pre-trained)
- **Deployment:** Google Cloud Run
- **Container:** Docker
- **Language:** Python 3.10

## 📦 Kurulum

### Gereksinimler
- Python 3.10+
- Docker (opsiyonel)
- Google Cloud SDK (deployment için)

### Yerel Kurulum

1. **Repository'yi klonlayın:**
```bash
git clone https://github.com/YOUR_USERNAME/cancer-classification.git
cd cancer-classification
```

2. **Virtual environment oluşturun:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Modeli eğitin** (veya hazır model kullanın):
```bash
python notebooks/train_model.py
```

5. **API'yi başlatın:**
```bash
python app/main.py
```

API şu adreste çalışacak: http://localhost:8000

## 🐳 Docker ile Çalıştırma
```bash
# Image'ı build edin
docker build -t cancer-api .

# Container'ı çalıştırın
docker run -p 8080:8080 cancer-api
```

API: http://localhost:8080

## ☁️ Google Cloud Deployment
```bash
# Google Cloud'a giriş yapın
gcloud auth login

# Projeyi seçin
gcloud config set project YOUR_PROJECT_ID

# Deploy edin
gcloud run deploy cancer-api \
  --source . \
  --platform managed \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

## 📖 API Kullanımı

### Endpoints

#### `GET /` - Ana Sayfa
```bash
curl https://cancer-api-993742066618.europe-west1.run.app/
```

#### `GET /health` - Health Check
```bash
curl https://cancer-api-993742066618.europe-west1.run.app/health
```

#### `POST /predict` - Tahmin Yap
```bash
curl -X POST "https://cancer-api-993742066618.europe-west1.run.app/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.0, 0.000026, 0.000077, ..., 0.000524]
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "benign",
  "confidence": 0.69,
  "probabilities": {
    "benign": 0.69,
    "malignant": 0.31
  }
}
```

#### `GET /model-info` - Model Bilgisi
```bash
curl https://cancer-api-993742066618.europe-west1.run.app/model-info
```

### Python Örneği
```python
import requests

# API URL
API_URL = "https://cancer-api-993742066618.europe-west1.run.app"

# 1000 özellik vektörü
features = [0.0] * 1000  # Gerçek verilerinizi kullanın

# Tahmin isteği
response = requests.post(
    f"{API_URL}/predict",
    json={"features": features}
)

result = response.json()
print(f"Tahmin: {result['prediction_label']}")
print(f"Güven: {result['confidence']:.2%}")
```

### JavaScript Örneği
```javascript
const API_URL = "https://cancer-api-993742066618.europe-west1.run.app";

// 1000 özellik vektörü
const features = new Array(1000).fill(0); // Gerçek verilerinizi kullanın

fetch(`${API_URL}/predict`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ features })
})
.then(response => response.json())
.then(data => {
  console.log(`Tahmin: ${data.prediction_label}`);
  console.log(`Güven: ${(data.confidence * 100).toFixed(2)}%`);
});
```

## 📂 Proje Yapısı
```
cancer-classification/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI uygulaması
│   └── models/
│       └── cancer_model.pkl # Eğitilmiş model
├── notebooks/
│   └── train_model.py       # Model eğitim scripti
├── data/                    # Dataset (gitignore'da)
├── Dockerfile               # Docker configuration
├── requirements.txt         # Python bağımlılıkları
├── .dockerignore
├── .gitignore
└── README.md
```

## 🔬 Veri Seti

- **Kaynak:** EfficientNet-B0 pre-trained model ile çıkarılmış özellikler
- **Özellik Sayısı:** 1000
- **Eğitim Seti:** 9,605 örnek
  - Benign: 5,000
  - Malignant: 4,605
- **Test Seti:** 1,000 örnek
  - Benign: 500
  - Malignant: 500

## 🧪 Model Detayları

**Algoritma:** Random Forest Classifier

**Hiperparametreler:**
- `n_estimators`: 100
- `max_depth`: 20
- `random_state`: 42

**Eğitim Süresi:** ~3 saniye

**Model Boyutu:** ~8 MB

## 📈 Gelecek Geliştirmeler

- [ ] Web arayüzü ekleme
- [ ] Görüntüden direkt özellik çıkarma
- [ ] Model versiyonlama sistemi
- [ ] A/B testing desteği
- [ ] Batch prediction endpoint
- [ ] API rate limiting
- [ ] Kullanıcı kimlik doğrulama
- [ ] Prometheus metrics entegrasyonu

## 📝 Lisans

MIT License - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👨‍💻 Geliştirici

**Ali Güneş**
- GitHub: [@aligunesgit](https://github.com/aligunesgit)
- LinkedIn: [Ali GUNES](https://linkedin.com/in/alisun)

## 🙏 Teşekkürler

- FastAPI topluluğu
- Google Cloud Platform
- scikit-learn geliştiricileri

## 📞 İletişim

Sorularınız için issue açabilir veya bana ulaşabilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!