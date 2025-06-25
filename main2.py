import streamlit as st
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ----------------- SETUP -----------------
st.set_page_config(page_title="Research Paper Chatbot", layout="wide")
st.title("📄 Research Paper Chatbot – RAG-based Academic Assistant")
st.write("Ask questions based on uploaded research papers (with math support).")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", api_key="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")

# ----------------- UPLOAD -----------------


uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Parsing and indexing your research paper..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()

        # Split using RecursiveTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        split_docs = text_splitter.split_documents(documents)

        # Vectorize using FAISS
        embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        st.session_state.vectorstore = vectorstore

    st.success("✅ PDF parsed and indexed successfully!")

# ----------------- CHAT DISPLAY -----------------
if st.session_state.chat_history:
    st.subheader("🗨️ Chat History")
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(
                f"""
                <div style='
                    text-align: right;
                    background-color: #000000;
                    color: white;
                    padding: 10px;
                    border-radius: 10px;
                    margin: 10px 0;
                '>
                    <strong>🧑 You:</strong><br>{msg}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style='
                    text-align: left;
                    background-color: #1a1a1a;
                    color: white;
                    padding: 10px;
                    border-radius: 10px;
                    margin: 10px 0;
                '>
                    <strong>🤖 Assistant:</strong><br>{msg}
                </div>
                """,
                unsafe_allow_html=True
            )
# ----------------- ASK YOUR QUESTION -----------------
if st.session_state.vectorstore:
    user_question = st.text_input("Ask your question:")

    if st.button("Submit") and user_question:
        # Perform similarity search
        relevant_docs = st.session_state.vectorstore.similarity_search(user_question, k=5)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Prompt format
        final_prompt = f"""
You are an academic assistant. Use the below context from a research paper to answer the question.

Context:
{context}

Question: {user_question}
Answer:
        """

        with st.spinner("Thinking..."):
            response = st.session_state.llm.invoke(final_prompt)

        # Save to session
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", response.content))
