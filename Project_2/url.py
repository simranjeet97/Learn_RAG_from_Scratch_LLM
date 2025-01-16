import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
import os

# Set the Google API Key in the environment
st.set_page_config(page_title="Interactive QA App", page_icon="🧙‍♂️", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f0f4f5;
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px;
        margin: 10px;
        border-radius: 8px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.title("📚 Interactive QA App with Generative AI")
st.write("Ask detailed questions based on contextual data, and get accurate and rich responses.")

# Prompt user for API key
api_key = st.text_input(
    "Enter your Google API key:",
    type="password",
    help="Your API key is required to use the Google Generative AI services.",
)

if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key

# Load Data
@st.cache_data
def get_pdf_text(url):
    """
    Loads content from a predefined URL and processes it into a string.
    """
    loader = WebBaseLoader(url)
    documents = loader.load()
    text = "\n\n".join([doc.page_content for doc in documents])
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

st.sidebar.header("📝 Enter the URL")
st.sidebar.write("Provide a URL and ask a question based on the document's content.")
# Input for URL
url = st.sidebar.text_input("Enter the URL to extract context from:", 
    placeholder="e.g., https://en.wikipedia.org/wiki/Harry_Potter")


# Prepare Data
if api_key:
    st.sidebar.header("📋 Preparing Data...")

# Conversational Chain
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




# User Interaction
st.sidebar.header("📝 Ask a Question")
user_question = st.sidebar.text_area(
    "Enter your question:",
    placeholder="E.g., Name and Plot of Harry Potter First Movie.",
)

if st.sidebar.button("Get Answer"):
    if url and api_key and user_question:
        try:
            with st.spinner("Searching for the answer..."):
                text = get_pdf_text(url)
                chunks = get_text_chunks(text)
                vectorstore = get_vector_store(chunks)
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                docs = new_db.similarity_search(user_question)
                chain = get_conversational_chain()
                response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

            st.success("Answer Generated!")
            st.subheader("Your Question:")
            st.write(user_question)

            st.subheader("Generated Answer:")
            # Display the response line by line
            for line in response["output_text"].split("\n"):
                if line.strip(): 
                    st.write(line)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both the API key and a question.")

# Footer
st.markdown(
    """
    ---
    🤖 Powered by LangChain and Google Generative AI
    """
)
