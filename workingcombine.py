import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

import streamlit as st
import pickle
import time
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import PyPDF2
import spacy

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_md')
except OSError:
    st.error("SpaCy model 'en_core_web_md' not found. Please install it using: python -m spacy download en_core_web_md")
    st.stop()

# Use BART instead of BERT for summarization (simpler & no SSL issues)
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit App Title
st.title("Article & Summarization Tool 📝")

# Sidebar Navigation
app_mode = st.sidebar.radio("📘 Choose Application", ('Article Research Tool', 'PDF Summarization Tool'))

# Beautiful Sidebar with Markdown, Emojis, and Icons
st.sidebar.markdown("## :material/article: Welcome to the Tool")
st.sidebar.markdown("""
### :material/help_outline: How to Use

1. **Choose** an option from the menu:
   - 📚 `Article Research Tool`
   - 📄 `PDF Summarization Tool`
2. For Articles:
   - Paste **valid URLs** (https://...)
   - Click **Process URLs**
3. For PDFs:
   - Upload a PDF file
   - Click **Summarize**

---
""")
st.sidebar.markdown("### :material/tips_and_updates: Tips")
st.sidebar.markdown("""
- Use **good quality** PDFs with readable text  
- For articles, try **news or blog URLs**  
- Type `summarize` in your query for summaries
""")
st.sidebar.markdown("---")

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Summarize text using pipeline
def summarize_text(text):
    try:
        summary = summarization_pipeline(text[:1024], max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error during summarization: {e}")
        return None

# Article Research Tool
def article_research_tool():
    st.sidebar.title("🔗 Enter News Article URLs")
    urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
    process_url_clicked = st.sidebar.button("🚀 Process URLs")
    file_path = "faiss_store_pretrained.pkl"
    main_placeholder = st.empty()

    try:
        qa_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    if process_url_clicked:
        valid_urls = [url for url in urls if url.startswith(('http://', 'https://'))]
        if not valid_urls:
            st.error("Please enter at least one valid URL starting with http:// or https://")
            return

        try:
            docs = []
            for i, url in enumerate(valid_urls):
                with st.spinner(f"Loading data from URL {i+1}..."):
                    loader = UnstructuredURLLoader(urls=[url])
                    data = loader.load()

                    splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
                    url_docs = splitter.split_documents(data)

                    for doc in url_docs:
                        doc.metadata['source'] = url
                    docs.extend(url_docs)

            with st.spinner("Creating vector store..."):
                embeddings = HuggingFaceEmbeddings()
                vectorstore = FAISS.from_documents(docs, embeddings)
                with open(file_path, "wb") as f:
                    pickle.dump(vectorstore, f)

            st.success("✅ Processing completed!")

        except Exception as e:
            st.error(f"Error processing URLs: {e}")
            return

    query = st.text_input("🔍 Ask a question or type 'summarize' to get a summary:")
    if query:
        if not os.path.exists(file_path):
            st.error("Please process URLs first before asking questions")
            return

        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                retriever = vectorstore.as_retriever()
                retrieved_docs = retriever.get_relevant_documents(query)

                if not retrieved_docs:
                    st.warning("No relevant documents found.")
                    return

                combined_text = " ".join([doc.page_content for doc in retrieved_docs])

                if "summarize" in query.lower():
                    with st.spinner("Summarizing..."):
                        summary = summarization_pipeline(combined_text[:1024], max_length=150, min_length=30, do_sample=False)
                        st.subheader("📝 Summary")
                        st.write(summary[0]['summary_text'])
                else:
                    with st.spinner("Answering..."):
                        result = qa_pipeline(f"Question: {query} \nContext: {combined_text[:1024]}", max_length=500, num_return_sequences=1)
                        st.subheader("💬 Answer")
                        st.write(result[0]['generated_text'])

                sources = list(set(doc.metadata.get('source', 'No Source') for doc in retrieved_docs))
                if sources:
                    st.subheader("🔗 Sources")
                    for i, src in enumerate(sources):
                        st.write(f"{i+1}. {src}")

        except Exception as e:
            st.error(f"Query error: {e}")

# PDF Summarization Tool
def pdf_summarization_tool():
    st.sidebar.info("📄 Upload a PDF to extract and summarize.")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        with st.spinner("🔍 Extracting text..."):
            pdf_text = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            st.subheader("📄 Original Text (First 500 chars)")
            st.text(pdf_text[:500] + "...")

            if st.button("🧠 Summarize"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(pdf_text)
                    if summary:
                        st.subheader("📝 Summary")
                        st.write(summary)
        else:
            st.error("No text found in the PDF.")

# Run based on selected mode
if app_mode == 'Article Research Tool':
    article_research_tool()
elif app_mode == 'PDF Summarization Tool':
    pdf_summarization_tool()
