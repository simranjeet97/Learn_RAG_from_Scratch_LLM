import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
import os

# Set the Google API Key in the environment
st.set_page_config(page_title="PDF QA App", page_icon="📄", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #eef2f3;
        color: #2c3e50;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-size: 16px;
        padding: 8px 16px;
        border: none;
        border-radius: 6px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #2980b9;
    }
    .stFileUploader {
        margin-bottom: 20px;
    }
    .stTextArea {
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📄 PDF QA App with Generative AI")
st.write("Upload a PDF document, ask a question, and get accurate, detailed responses.")

# Input: Google API Key
api_key = st.text_input(
    "Enter your Google API key:",
    type="password",
    help="Your API key is required to use the Google Generative AI services.",
)

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# File Upload Section
st.sidebar.header("📂 Upload a PDF")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file to extract content:",
    type=["pdf"],
)

# User Question Section
st.sidebar.header("📝 Ask a Question")
user_question = st.sidebar.text_area(
    "Enter your question:",
    placeholder="E.g., What is the main topic of the document?",
)

@st.cache_data
def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using PyPDF2.
    """
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def get_text_chunks(text):
    """
    Splits the loaded text into chunks for embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

@st.cache_resource
def get_vector_store(text_chunks):
    """
    Embeds the text chunks into a vector store for similarity search.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """
    Creates a conversational chain for QA using LangChain and Google Generative AI.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "Answer is not available in the context." Don't provide a wrong answer.
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Process Uploaded File
if uploaded_file and api_key and user_question:
    with st.spinner("Processing your PDF..."):
        try:
            # Step 1: Extract text from PDF
            raw_text = extract_text_from_pdf(uploaded_file)

            # Step 2: Split text into chunks
            text_chunks = get_text_chunks(raw_text)

            # Step 3: Embed text into a vector store
            vector_store = get_vector_store(text_chunks)

            # Step 4: Perform similarity search and generate the answer
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            # Display the results
            st.success("Answer Generated!")
            st.subheader("Your Question:")
            st.write(user_question)

            st.subheader("Generated Answer:")
            for line in response["output_text"].split("\n"):
                if line.strip():
                    st.write(line)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    if not api_key:
        st.warning("Please provide your Google API key.")
    if not uploaded_file:
        st.warning("Please upload a PDF file.")
    if not user_question:
        st.warning("Please enter a question.")

# Footer
st.markdown(
    """
    ---
    🌟 Powered by LangChain and Google Generative AI
    """
)