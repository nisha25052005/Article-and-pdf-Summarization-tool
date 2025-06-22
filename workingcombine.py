import os
import streamlit as st
import pickle
import time
from transformers import pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import PyPDF2
from summarizer import Summarizer
import spacy

# Load SpaCy model for better sentence segmentation
nlp = spacy.load('en_core_web_md')

# Initialize BERT model for summarization
pdf_summarizer_model = Summarizer()

# Streamlit App Interface
st.title("Article & Summarization Tool 📝")

# Sidebar: Select Application Mode
app_mode = st.sidebar.radio(
    "Choose Application",
    ('Article Research Tool', 'PDF Summarization Tool')
)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

# Function to summarize text from PDF
def summarize_text(text):
    # Use SpaCy to segment text into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Generate a summary with BERT Extractive Summarizer
    summary = pdf_summarizer_model(' '.join(sentences), min_length=60)

    return summary

# Article Research Tool Application
def article_research_tool():
    st.sidebar.title("News Article URLs")

    urls = []
    for i in range(3):
        url = st.sidebar.text_input(f"URL {i+1}")
        urls.append(url)

    process_url_clicked = st.sidebar.button("Process URLs")
    file_path = "faiss_store_pretrained.pkl"

    main_placeholder = st.empty()

    # Load pre-trained models for question answering and summarization from Hugging Face
    qa_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")  # For summarization and long-form answers
    summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

    if process_url_clicked:
        # Load data from URLs and assign proper metadata to each document
        docs = []
        for i, url in enumerate(urls):
            loader = UnstructuredURLLoader(urls=[url])
            main_placeholder.text(f"Loading data from URL {i+1}...✅✅✅")
            data = loader.load()

            # Split data into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                separators=['\n\n', '\n', '.', ','],
                chunk_size=1000
            )
            url_docs = text_splitter.split_documents(data)

            # Add metadata (source URL) to each document
            for doc in url_docs:
                doc.metadata['source'] = url

            docs.extend(url_docs)

        # Create embeddings and save to FAISS index
        embeddings = HuggingFaceEmbeddings()  # Free embeddings
        vectorstore = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

    # User can enter a query to search the articles
    query = main_placeholder.text_input("Question or Summarize: ")
    if query:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
                retriever = vectorstore.as_retriever()
                retrieved_docs = retriever.get_relevant_documents(query)

                # Combine the documents into one text passage for the QA pipeline
                combined_text = " ".join([doc.page_content for doc in retrieved_docs])

                # Check if the combined_text is not empty or too short
                if len(combined_text) < 50:
                    st.error("The text content is too short for summarization or answering. Try a different input or URL.")
                else:
                    # Check if the query contains the word "summarize" and perform summarization
                    if "summarize" in query.lower():
                        try:
            ~                # Trim the combined text to handle model token length
                            summary = summarization_pipeline(combined_text[:1024], max_length=150, min_length=30, do_sample=False)
                            st.header("Summary")
                            st.write(summary[0]['summary_text'])
                        except IndexError as e:
                            st.error(f"An error occurred while summarizing: {e}")
                    else:
                        try:
                            # Use the bart-large model for generating long, detailed answers
                            result = qa_pipeline(f"Question: {query} \nContext: {combined_text[:1024]}", max_length=500, num_return_sequences=1)

                            # Display the answer
                            st.header("Answer")
                            st.write(result[0]['generated_text'])
                        except IndexError as e:
                            st.error(f"An error occurred while answering the question: {e}")

                    # Retrieve and display distinct sources
                    sources = set(doc.metadata.get('source', 'No Source') for doc in retrieved_docs)
                    st.subheader("Sources:")
                    for i, source in enumerate(sources):
                        st.write(f"Source {i+1}: {source}")

# PDF Summarization Tool Application
def pdf_summarization_tool():
    st.title('PDF Summarization Tool')

    # File uploader in sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=['pdf'])

    if uploaded_file is not None:
        pdf_text = extract_text_from_pdf(uploaded_file)

        if pdf_text:
            st.subheader("Original Text from PDF")
            st.write(pdf_text)

            if st.button("Summarize"):
                try:
                    summary = summarize_text(pdf_text)
                    st.subheader("Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error during summarization: {e}")
        else:
            st.write("No text could be extracted from the PDF.")
    else:
        st.write("Please upload a PDF file.")

    st.sidebar.info("Upload a PDF file to extract and summarize its text.")

# App Mode Selector
if app_mode == 'Article Research Tool':
    article_research_tool()
elif app_mode == 'PDF Summarization Tool':
    pdf_summarization_tool()
