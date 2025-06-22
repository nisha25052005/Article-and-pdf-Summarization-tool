

# **# Article & Summarization Tool 📝**

This project is a web application built using Streamlit that offers two main functionalities:
1. **Article Research Tool**: Allows users to analyze and summarize content from URLs.
2. **PDF Summarization Tool**: Enables users to upload a PDF, extract its text, and generate a summary.

The application uses several NLP models and frameworks, including Hugging Face's transformers, FAISS for vector storage, and BERT for extractive summarization.

## **## Features**

- **Article Research Tool**: Input URLs, process articles, and answer queries or summarize the text using pre-trained NLP models.
- **PDF Summarization Tool**: Extracts text from uploaded PDF files and generates a summary.

### **## Requirements**

- Python 3.8+
- Streamlit
- Transformers (Hugging Face)
- FAISS
- PyPDF2
- Summarizer (BERT Extractive Summarizer)
- Spacy
- LangChain

**Install Dependencies: Create a virtual environment (optional but recommended) and install the required Python packages.**

```python
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
**Download Spacy Model:**
```python
python -m spacy download en_core_web_md
```

### **Running the Application
Start the Streamlit App:**

streamlit run app.py


### **Using the Tool:**

1.  Select between Article Research Tool and PDF Summarization Tool from the sidebar.
2.  For the Article Research Tool, input up to three URLs, and click Process URLs to analyze the content.
3.  For the PDF Summarization Tool, upload a PDF file, and click Summarize to extract and summarize its text.

**Folder Structure**
1. workingcombine.py: The main application script.
2. requirements.txt: Contains all Python dependencies.
3. README.md: This readme file.

#### **Example Usage**
- Article Research Tool
- Enter up to 3 article URLs in the sidebar.
- click Process URLs.
- Type a question or "summarize" followed by your query in the input box to get answers or summaries.

**PDF Summarization Tool**
- Upload a PDF file.
- Click Summarize to view the summarized text.

##### **Notes**
1. Ensure that faiss_store_pretrained.pkl (the saved FAISS index) is available or the app will create one upon 2. processing URLs.
2. Adjust the model as needed in the code (default is facebook/bart-large-cnn).


