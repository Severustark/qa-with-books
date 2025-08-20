# app.py
# Tek dosyada RAG: ingest (PDF/TXT -> Chroma) + Gradio arayüzü + OpenAI fallback
# Kullanım:
#   1) python app.py --ingest   # data/docs içindeki PDF/TXT'leri indeksle
#   2) python app.py --run      # arayüzü başlat

import os
import glob
import argparse
from typing import List, Tuple, Optional

from dotenv import load_dotenv


# ---- Ortam Değişkenleri / Ayarlar ----
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL     = os.getenv("CHAT_MODEL", "gpt-4o-mini")            # dilediğin modeli yaz
EMBED_MODEL    = os.getenv("EMBED_MODEL", "text-embedding-3-small")
DOCS_DIR       = os.getenv("DOCS_DIR", "data/docs")
VDB_DOCS       = os.getenv("VDB_DOCS", "vectordb/chroma_docs")
TOP_K          = int(os.getenv("TOP_K", "3"))
SIM_THRESHOLD  = float(os.getenv("SIM_THRESHOLD", "0.75"))         # eşik altı → fallback

# ---- LangChain bileşenleri ----
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- (Opsiyonel) UI ----
import gradio as gr


# =========================
#    Belge Yükleme & Ingest
# =========================
def load_all_documents(root: str):
    """
    DOCS_DIR altında PDF ve TXT dosyalarını toplayıp LangChain Document listesi döner.
    """
    files = glob.glob(os.path.join(root, "**", "*"), recursive=True)
    docs = []
    for f in files:
        lf = f.lower()
        if lf.endswith(".pdf"):
            docs.extend(PyPDFLoader(f).load())
        elif lf.endswith(".txt"):
            # Türkçe karakterler için encoding veriyoruz
            docs.extend(TextLoader(f, encoding="utf-8").load())
    return docs


def ingest_docs() -> int:
    """
    PDF/TXT -> chunk -> embedding -> Chroma persist
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyanı veya ortam değişkenlerini kontrol et.")

    os.makedirs(VDB_DOCS, exist_ok=True)

    documents = load_all_documents(DOCS_DIR)
    if not documents:
        print(f"⚠️  '{DOCS_DIR}' içinde PDF/TXT bulunamadı. Önce dosya yerleştirip tekrar deneyin.")
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBED_MODEL)
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=VDB_DOCS)
    vectordb.persist()

    print(f"✅ {len(chunks)} parça indekslendi → {VDB_DOCS}")
    return len(chunks)


# =========================
#        Soru-Cevap
# =========================
class DocsRAG:
    def __init__(self):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY bulunamadı. .env dosyanı veya ortam değişkenlerini kontrol et.")
        self.emb = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBED_MODEL)
        self.db  = Chroma(persist_directory=VDB_DOCS, embedding_function=self.emb)
        self.llm = ChatOpenAI(api_key=OPENAI_API_KEY, model=CHAT_MODEL, temperature=0.2)

        # Kitapla ilgi filtresi için anahtarlar
        self._book_keywords = [
            "kitap", "roman", "hikaye", "öykü", "yazar", "karakter", "tema", "konu",
            "alıntı", "sayfa", "bölüm",
            "boş dolaplar", "bos dolaplar", "annie ernaux", "ernaux"
        ]
        # Anahtarları normalize edilmiş halde sakla (karşılaştırma kolay olsun)
        self._norm_book_keywords = [self._normalize_tr(k) for k in self._book_keywords]

    @staticmethod
    def _normalize_tr(text: str) -> str:
        """Türkçe karakterleri sadeleştir + lower (örn: 'Boş' -> 'bos')."""
        table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
        return text.translate(table).lower()

    def _is_book_related(self, question: str) -> bool:
        qn = self._normalize_tr(question)
        return any(k in qn for k in self._norm_book_keywords)

    def _search(self, question: str, k: int = TOP_K):
        return self.db.similarity_search_with_relevance_scores(question, k=k)

    def _format_prompt(self, question: str, context: Optional[str] = None) -> str:
        if context:
            return f"Bağlam:\n{context}\n\nSoru: {question}\n\nCevap:"
        return f"Soru: {question}\n\nCevap:"

    def answer(self, question: str) -> Tuple[str, str]:
        """
        Akış:
        1) Soru kitapla ilgili mi? Değilse yanıt verme.
        2) Belgelerde semantik arama.
        3) En iyi skor eşik üzerindeyse bağlamla yanıt.
        4) Eşik altıysa 'bilgi yok'.
        """
        # 1) Kitap filtresi
        if not self._is_book_related(question):
            return "Bu soru kitap/kitap içeriği ile ilgili görünmüyor, yanıt veremem.", "Filtre: kitap dışı"

        # 2) Arama
        results = self._search(question, k=TOP_K)
        if not results:
            return "Üzgünüm, belgelerde bu soruya dair bir bilgi bulamadım.", "Kaynak: Yok"

        top_score = results[0][1]

        # 3) Eşik kontrolü
        if top_score >= SIM_THRESHOLD:
            context = "\n\n".join([r[0].page_content for r in results])
            prompt  = self._format_prompt(question, context)
            try:
                out = self.llm.invoke(prompt).content
            except Exception:
                # Ola ki LLM çağrısında sorun olursa, en alakalı parçayı dök
                out = results[0][0].page_content
            return out, f"Kaynak: Belgeler (benzerlik={top_score:.2f})"

        # 4) Eşik altı
        return "Üzgünüm, belgelerde bu soruya dair bir bilgi bulamadım.", f"Kaynak: Eşik altı (en iyi={top_score:.2f})"


# =========================
#          UI
# =========================
def launch_ui():
    rag = DocsRAG()

    def ask(q):
        if not q or not q.strip():
            return "Lütfen bir soru yazın.", ""
        ans, src = rag.answer(q.strip())
        return ans, f"_{src}_"

    with gr.Blocks(title="📚 QA with Books (RAG)") as demo:
        gr.Markdown("## 📚 QA with Books (RAG)\nPDF/TXT'ten cevap; bulunamazsa OpenAI’a düşer.")
        q = gr.Textbox(label="Soru", lines=2, placeholder="Örn: Bu kitabın ana teması nedir?")
        a = gr.Markdown()
        s = gr.Markdown()
        gr.Button("Sor").click(ask, inputs=q, outputs=[a, s])
    demo.launch()


# =========================
#        CLI / main
# =========================
def main():
    parser = argparse.ArgumentParser(description="Tek dosyalık RAG: ingest + run")
    parser.add_argument("--ingest", action="store_true", help="DOCS_DIR içindeki PDF/TXT'leri Chroma'ya indeksle")
    parser.add_argument("--run",    action="store_true", help="Gradio arayüzünü başlat")
    args = parser.parse_args()

    # Varsayılan klasörleri oluştur
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(VDB_DOCS, exist_ok=True)

    if not (args.ingest or args.run):
        parser.print_help()
        print("\n🔎 Örnek:")
        print("  python app.py --ingest   # önce indeks oluştur")
        print("  python app.py --run      # sonra arayüzü başlat")
        return

    if args.ingest:
        ingest_docs()

    if args.run:
        launch_ui()


if __name__ == "__main__":
    main()
