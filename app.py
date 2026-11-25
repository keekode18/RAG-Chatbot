import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Please add it to your .env file.")
    st.stop()


DATA_FOLDER = "pdf_data"
CHROMA_FOLDER = "chroma_db"

os.makedirs(DATA_FOLDER, exist_ok=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    .source-box {
        padding: 12px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📚 PDF RAG Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload your PDF and ask questions about its content.</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"]
    )

    st.divider()

    st.write("### 📌 How it works")

    st.write("1. Upload a PDF")
    st.write("2. Extract text")
    st.write("3. Split into chunks")
    st.write("4. Create embeddings")
    st.write("5. Store in ChromaDB")
    st.write("6. Ask questions")

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("🔄 Reset Database"):

        if os.path.exists(CHROMA_FOLDER):
            shutil.rmtree(CHROMA_FOLDER)

        st.success("Database reset successfully!")
        st.rerun()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ============================================================
# PDF PROCESSING FUNCTION
# ============================================================

def process_pdf(pdf_path):

    loader = PyMuPDFLoader(pdf_path)

    documents = loader.load()

    if not documents:
        raise ValueError("No text could be extracted from the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_FOLDER
    )

    return vectorstore, len(documents), len(chunks)


# ============================================================
# LOAD EXISTING DATABASE
# ============================================================

def load_existing_database():

    if not os.path.exists(CHROMA_FOLDER):
        return None

    try:

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        vectorstore = Chroma(
            persist_directory=CHROMA_FOLDER,
            embedding_function=embeddings
        )

        return vectorstore

    except Exception:
        return None


# ============================================================
# HANDLE PDF UPLOAD
# ============================================================

if uploaded_file is not None:

    pdf_path = os.path.join(
        DATA_FOLDER,
        uploaded_file.name
    )

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.session_state.pdf_name != uploaded_file.name:

        with st.spinner("📖 Processing your PDF..."):

            try:

                vectorstore, page_count, chunk_count = process_pdf(
                    pdf_path
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.pdf_name = uploaded_file.name

                st.success(
                    f"✅ PDF processed successfully! "
                    f"{page_count} pages and {chunk_count} text chunks created."
                )

            except Exception as e:

                st.error(
                    f"❌ Error processing PDF: {str(e)}"
                )

else:

    if st.session_state.vectorstore is None:

        existing_db = load_existing_database()

        if existing_db is not None:
            st.session_state.vectorstore = existing_db


# ============================================================
# PDF STATUS
# ============================================================

if st.session_state.pdf_name:

    st.info(
        f"📄 Current PDF: **{st.session_state.pdf_name}**"
    )

elif uploaded_file is None:

    st.warning(
        "👈 Upload a PDF from the sidebar to start chatting."
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask something about your PDF..."
)


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

if user_question:

    if st.session_state.vectorstore is None:

        st.warning(
            "⚠️ Please upload a PDF before asking questions."
        )

        st.stop()


    # --------------------------------------------------------
    # Display user message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    with st.chat_message("user"):

        st.markdown(user_question)


    # --------------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("🔎 Searching the PDF..."):

            try:

                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                relevant_documents = retriever.invoke(
                    user_question
                )


                if not relevant_documents:

                    answer = (
                        "I couldn't find relevant information "
                        "in the uploaded PDF."
                    )

                    st.markdown(answer)

                else:

                    # ------------------------------------------------
                    # Create context
                    # ------------------------------------------------

                    context = "\n\n".join(
                        [
                            document.page_content
                            for document in relevant_documents
                        ]
                    )


                    # ------------------------------------------------
                    # Create prompt
                    # ------------------------------------------------

                    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question ONLY using the information
provided in the context below.

If the answer cannot be found in the context, say:

"I don't know based on the provided PDF."

Do not make up information.

Context:
{context}

Question:
{user_question}
"""


                    # ------------------------------------------------
                    # LLM
                    # ------------------------------------------------

                    llm = ChatOpenAI(
                        model="gpt-4o-mini",
                        temperature=0
                    )


                    response = llm.invoke(prompt)

                    answer = response.content


                    # ------------------------------------------------
                    # Display answer
                    # ------------------------------------------------

                    st.markdown(answer)


                    # ------------------------------------------------
                    # Display sources
                    # ------------------------------------------------

                    with st.expander("📚 View Sources"):

                        for i, document in enumerate(
                            relevant_documents
                        ):

                            page_number = (
                                document.metadata.get(
                                    "page",
                                    "Unknown"
                                )
                            )

                            source_name = (
                                document.metadata.get(
                                    "source",
                                    "PDF"
                                )
                            )

                            st.markdown(
                                f"""
**Source {i + 1}**

📄 File: `{os.path.basename(source_name)}`

📑 Page: `{page_number + 1 if isinstance(page_number, int) else page_number}`

---

{document.page_content[:700]}...
"""
                            )


                    # ------------------------------------------------
                    # Save assistant response
                    # ------------------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


            except Exception as e:

                error_message = f"""
❌ Something went wrong.

Error:

`{str(e)}`
"""

                st.error(error_message)