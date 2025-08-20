# 📚 QA with Books (RAG Pipeline)

Bu proje, **LangChain** + **OpenAI** kullanarak kitap içerikleri üzerine soru-cevap yapan bir **Retrieval-Augmented Generation (RAG)** sistemidir.  
PDF/TXT belgeleri **ChromaDB** ile vektör veritabanına indekslenir ve sorulara belgelerden bağlam kullanılarak cevap üretilir.  

---

## 🚀 Özellikler
- 📖 PDF/TXT belgelerden içerik yükleme  
- 🔎 Semantik arama ile ilgili bölümleri bulma  
- 🤖 OpenAI LLM ile bağlam destekli cevap üretme  
- 🗂️ ChromaDB ile kalıcı vektör saklama  

---

## ⚙️ Kurulum

1. Depoyu klonla:
```bash
git clone https://github.com/<kullanıcı-adı>/qa-with-books.git
cd qa-with-books
```

2. Sanal ortam oluştur ve bağımlılıkları yükle:
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

3. `.env` dosyası oluştur ve API anahtarını ekle:
```env
OPENAI_API_KEY=your_api_key_here
```

---

## ▶️ Çalıştırma

### 1) Belgeleri indeksle
```bash
python app.py --ingest
```

### 2) Arayüzü başlat
```bash
python app.py --run
```
➡️ Gradio arayüzü açılacak ve tarayıcıdan kullanabileceksiniz.

---

## ❓ Örnek Sorular

| Soru | Beklenen Cevap Kaynağı |
|------|------------------------|
| **Boş Dolaplar’ın konusu nedir?** | PDF’deki *Annie Ernaux – Boş Dolaplar* kitabından özet |
| **Annie Ernaux kimdir?** | Belgelerden biyografi bilgisi |
| **Boş Dolaplar hangi temaları işler?** | Belgelerden temalar |
| **Kuru fasulye tarifi nedir?** | ❌ Yanıt verilmez (çünkü belgeyle alakasız) |

<img width="1918" height="1005" alt="mantıklı soru" src="https://github.com/user-attachments/assets/3f843cf2-d9f9-459f-b7ce-f3bc8665aff3" />

 **Not:** Konuyla alakalı olan keyword’leri seçerek cevaplıyor; **konu dışı** ise cevaplamıyor.

<img width="1918" height="1045" alt="saçma soru" src="https://github.com/user-attachments/assets/ebd51f0f-802e-48a8-962a-39fef1da0997" />

---

## 📂 Proje Yapısı
```
qa-with-books/
├── app.py                # Ana uygulama
├── requirements.txt      # Bağımlılıklar
├── README.md             # Dokümantasyon
├── .env                  # OpenAI API key (lokalde)
├── vectordb/             # ChromaDB veritabanı
└── docs/                 # PDF/TXT belgeler
```

---

## 🛠️ Teknolojiler
- [LangChain](https://www.langchain.com/)  
- [OpenAI GPT](https://platform.openai.com/)  
- [ChromaDB](https://www.trychroma.com/)  
- [Gradio](https://gradio.app/)  

---

## 📌 Notlar
- Bu proje yalnızca **belgelerdeki bilgiye dayalı cevaplar** üretir.  
- Belgeyle ilgisiz sorulara yanıt vermez.

---

👤 Geliştirici Ad Soyad: DAMLA ARPA

Bu proje, Kairu Bootcamp Eğitimleri kapsamında bir ödev/proje olarak geliştirilmiştir.
