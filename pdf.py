import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Streamlit app configuration
st.set_page_config(page_title="PDF Question Answering", layout="wide")
st.title("Ask Questions from Your PDF")

# Initialize session state for vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.strip()

# Function to process PDF and create vector store
def process_pdf(file):
    # Extract text
    text = extract_text_from_pdf(file)
    
    if not text:
        st.error("No text could be extracted from the PDF. Ensure it contains readable text (not scanned images).")
        return None
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced chunk size for smaller inputs
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        st.error("No valid text chunks created. Try a different PDF.")
        return None
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# Function to initialize LLM
@st.cache_resource
def load_llm():
    model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Reduced to conserve token budget
        temperature=0.7,
        top_p=0.9
    )
    return HuggingFacePipeline(pipeline=pipe), tokenizer

# Custom prompt template to reduce token usage
prompt_template = """Use the following context to answer the question concisely. If the answer is not in the context, say so.

Context: {context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Process uploaded PDF
if uploaded_file:
    with st.spinner("Processing PDF..."):
        st.session_state.vector_store = process_pdf(uploaded_file)
    if st.session_state.vector_store:
        st.success("PDF processed successfully!")
    else:
        st.warning("PDF processing failed. Check the error message above.")

# Question input and answering
if st.session_state.vector_store:
    question = st.text_input("Ask a question about the PDF:", value="What is skill?")
    if question:
        with st.spinner("Generating answer..."):
            try:
                # Load LLM and tokenizer
                llm, tokenizer = load_llm()
                
                # Get retrieved documents
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(question)
                
                if not docs:
                    st.error("No relevant documents found for the question. Try rephrasing or uploading a different PDF.")
                    st.stop()
                
                # Truncate context to fit within token limit
                max_input_tokens = 800  # Reserve space for question and output
                context = " ".join([doc.page_content for doc in docs if doc.page_content.strip()])
                encoded_context = tokenizer.encode(context, add_special_tokens=False)
                if len(encoded_context) > max_input_tokens - 100:  # Leave 100 tokens for question
                    context = tokenizer.decode(encoded_context[:max_input_tokens - 100])
                    st.warning("Context truncated to fit model token limit.")
                
                encoded_question = tokenizer.encode(question, add_special_tokens=False)
                if len(encoded_question) > 100:
                    question = tokenizer.decode(encoded_question[:100])
                    st.warning("Question truncated to fit model token limit.")
                
                # Debug: Show token counts
                total_tokens = len(tokenizer.encode(prompt_template.format(context=context, question=question)))
                st.info(f"Total input tokens: {total_tokens}")
                
                # Create QA chain with custom prompt
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )
                
                # Get answer
                result = qa_chain({"query": question})
                answer = result["result"]
                st.write("**Answer:**")
                st.write(answer)
                
                # Display source documents
                with st.expander("Source Documents"):
                    for doc in result["source_documents"]:
                        st.write(doc.page_content)
                        st.write("---")
            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                st.write("Possible causes: Input too long, empty PDF content, or model limitations.")
                st.write("Try rephrasing the question, uploading a text-based PDF, or reducing context size.")

# Instructions for .env file
st.sidebar.markdown("""
### Setup Instructions
1. Create a `.env` file in the project root with:
```
HF_TOKEN=your_huggingface_token
```
2. Install required packages:
```bash
pip install streamlit PyPDF2 python-dotenv transformers
                    
    """)