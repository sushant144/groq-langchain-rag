import streamlit as st
import os
import time

from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from groq import Groq

from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "groq-langchain-rag")
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

FAISS_INDEX_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
RELEVANCE_THRESHOLD = 0.3


@st.cache_data(show_spinner=False)
def verify_groq_model(model_id: str) -> tuple[bool, list[str]]:
    """Check the model exists in the user's Groq account. Returns (exists, available_ids)."""
    client = Groq(api_key=groq_api_key)
    available = [m.id for m in client.models.list().data]
    return model_id in available, available


def vector_embeddings():
    if os.path.exists(FAISS_INDEX_PATH):
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        st.session_state.vectors = FAISS.load_local(
            FAISS_INDEX_PATH,
            st.session_state.embeddings,
            allow_dangerous_deserialization=True,
        )
        st.write("FAISS index loaded from disk.")
    else:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:100]
        )
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings
        )
        st.session_state.vectors.save_local(FAISS_INDEX_PATH)
        st.write("FAISS index created and saved to disk.")


st.title("LangChain Groq")

model_ok, available_models = verify_groq_model(groq_model)
if not model_ok:
    st.error(
        f"Model '{groq_model}' is not available on your Groq account. "
        f"Available models: {', '.join(sorted(available_models))}"
    )
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model=groq_model)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the context below. If the question cannot be answered based on the context,
    say 'I don't know'.
    {context}
    Question: {input}
    """
)

user_input = st.text_input("Enter your question:")

if "embeddings_done" not in st.session_state:
    st.session_state.embeddings_done = False

if not st.session_state.embeddings_done:
    if st.button("Document Embeddings"):
        with st.spinner("Processing documents..."):
            vector_embeddings()
        st.session_state.embeddings_done = True
        st.write("Document embeddings created and stored in session state.")
        st.write("You can now ask questions.")
else:
    st.button("Document Embeddings", disabled=True, key="embed_done_button")
    st.write("Document embeddings already created. You can ask questions now.")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if user_input:
    if not st.session_state.embeddings_done:
        st.warning("Please create document embeddings first by clicking the button above.")
    else:
        scored_docs = st.session_state.vectors.similarity_search_with_relevance_scores(
            user_input, k=4
        )
        relevant_docs = [doc for doc, score in scored_docs if score >= RELEVANCE_THRESHOLD]
        top_score = max((score for _, score in scored_docs), default=0.0)

        if not relevant_docs:
            st.warning(
                f"This question doesn't appear to be covered by the indexed documents "
                f"(top relevance score: {top_score:.2f}, threshold: {RELEVANCE_THRESHOLD}). "
                f"Try rephrasing or ask about LangSmith / LangChain documentation topics."
            )
        else:
            rag_chain = (
                RunnablePassthrough.assign(context=lambda x: relevant_docs)
                .assign(
                    answer=(
                        lambda x: {"context": format_docs(x["context"]), "input": x["input"]}
                    )
                    | prompt
                    | llm
                    | StrOutputParser()
                )
            )

            start = time.process_time()
            response = rag_chain.invoke({"input": user_input})
            end = time.process_time()

            time_taken = end - start
            st.write("Time taken to get the response: ", round(time_taken, 4), "seconds")
            st.write(f"Top relevance score: {top_score:.2f}")
            st.write(response["answer"])

            with st.expander("Document Similarity Search"):
                for doc, score in scored_docs:
                    st.write(f"**Score: {score:.3f}**")
                    st.write(doc.page_content)
                    st.write("---")
