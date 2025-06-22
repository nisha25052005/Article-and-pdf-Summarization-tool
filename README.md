
# 📝 Article & PDF Summarization Tool

This project is a simple yet powerful web application built using **Streamlit** that provides two core features:

1. **Article Research Tool** – Analyze and summarize content from URLs.
2. **PDF Summarization Tool** – Upload a PDF, extract its text, and generate concise summaries.

The app leverages cutting-edge **NLP models** from Hugging Face, uses **FAISS** for vector search, and **BERT** for extractive summarization.

---

##  Features

- **Article Research Tool**:
  - Input up to 3 URLs.
  - Extracts article content.
  - Summarizes or answers questions using pre-trained language models.

- **PDF Summarization Tool**:
  - Upload any PDF file.
  - Extracts the text and summarizes it with ease.

---

## Requirements

- Python 3.8+
- Streamlit
- Transformers (Hugging Face)
- FAISS
- PyPDF2
- BERT Extractive Summarizer
- SpaCy
- LangChain

---

## 🛠️ Setup Instructions

**1. Clone the repo & install dependencies:**

```bash
git clone https://github.com/nisha25052005/Article-and-pdf-Summarization-tool.git
cd Article-and-pdf-Summarization-tool

# Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # On Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

**2. Download SpaCy Model:**

```bash
python -m spacy download en_core_web_md
```

---

## 💻 Run the App

```bash
streamlit run workingcombine.py
```

---

## 🧠 How to Use

* From the sidebar:

  * Choose between **Article Research Tool** or **PDF Summarization Tool**.
* For Article Research:

  * Paste 1–3 article URLs.
  * Click **"Process URLs"**.
  * Ask a question or type "summarize" to get an overview.
* For PDF Summarization:

  * Upload a PDF file.
  * Click **"Summarize"** to generate output.

---

## 📁 Project Structure

```bash
├── README.md                  # Project overview
├── requirements.txt           # List of dependencies
└── workingcombine.py          # Main Streamlit app
```

---

## 📝 Notes

* Ensure you have a FAISS index file (e.g., `faiss_store_pretrained.pkl`) if you're loading a pretrained vector store.
* If the FAISS file doesn't exist, the app will auto-create one after processing articles.
* You can tweak the summarization model (e.g., switch from `facebook/bart-large-cnn` to any Hugging Face model).

