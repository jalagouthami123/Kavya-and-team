import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    """
    Extracts all text from a given PDF file.
    """
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Load the summarization model
# We're using a small, efficient model for this example.
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

# Streamlit App Interface
st.title("PDF to Text Summarizer")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write("Extracting text from PDF...")
    
    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    if len(pdf_text) > 0:
        st.write("Text extracted successfully! Now summarizing...")
        
        # Split text into smaller chunks if it's too long
        max_chunk_size = 1024
        text_chunks = [pdf_text[i:i + max_chunk_size] for i in range(0, len(pdf_text), max_chunk_size)]
        
        # Summarize each chunk
        summaries = []
        for chunk in text_chunks:
            # Generate a summary for the current chunk
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
            
        # Combine the summaries
        final_summary = " ".join(summaries)
        
        st.subheader("Summarized Text:")
        st.write(final_summary)
    else:
        st.error("Could not extract text from the PDF. The file may be an image-based PDF.")