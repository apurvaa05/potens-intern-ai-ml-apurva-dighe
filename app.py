import streamlit as st

from deep_translator import GoogleTranslator

from utils.pdf_loader import load_pdfs
from utils.chunker import chunk_text
from utils.embeddings import create_embeddings, model

from utils.retriever import (
    build_faiss_index,
    retrieve,
    rerank_results
)

from utils.llm import generate_answer
from utils.contradict import check_contradiction

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #050816;
    color: white;
}

.main-container {
    max-width: 1200px;
    margin: auto;
    padding-top: 30px;
}

.hero-title {
    font-size: 60px;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 12px;
    color: white;
}

.hero-subtitle {
    font-size: 18px;
    color: #94a3b8;
    margin-bottom: 40px;
}

.section-title {
    font-size: 28px;
    font-weight: 600;
    margin-top: 50px;
    margin-bottom: 20px;
}

.stTextInput input {
    background-color: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    color: white;
    padding: 14px;
    font-size: 16px;
}

.stTextInput input:focus {
    border: 1px solid #3b82f6;
    box-shadow: none;
}

.stButton button {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 22px;
    font-weight: 600;
    transition: 0.2s ease;
}

.stButton button:hover {
    background-color: #1d4ed8;
    transform: translateY(-1px);
}

.answer-box {
    background: linear-gradient(
        145deg,
        #111827,
        #0f172a
    );

    border: 1px solid #1e293b;

    padding: 28px;

    border-radius: 18px;

    margin-top: 20px;

    line-height: 1.8;

    font-size: 17px;
}

.source-card {
    background: linear-gradient(
        145deg,
        #111827,
        #0f172a
    );

    border: 1px solid #1e293b;

    border-radius: 18px;

    padding: 22px;

    margin-bottom: 20px;

    transition: 0.2s ease;
}

.source-card:hover {
    transform: translateY(-2px);
    border-color: #2563eb;
}

.source-title {
    font-size: 18px;
    font-weight: 600;
    color: white;
    margin-bottom: 10px;
}

.chunk-text {
    color: #d1d5db;
    line-height: 1.7;
    margin-top: 15px;
}

.score {
    color: #60a5fa;
    font-size: 14px;
    margin-top: 10px;
}

.divider {
    height: 1px;
    background-color: #1e293b;
    margin-top: 60px;
    margin-bottom: 40px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-container">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-title">Enterprise AI Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">Multilingual RAG system with semantic retrieval, reranking, contradiction analysis, and citation-based responses.</div>',
    unsafe_allow_html=True
)

with st.spinner("Loading enterprise documents..."):

    docs = load_pdfs("documents")

    all_chunks = []
    chunks_metadata = []

    for doc in docs:

        chunks = chunk_text(doc["text"])

        for idx, chunk in enumerate(chunks):

            all_chunks.append(chunk)

            chunks_metadata.append({
                "file_name": doc["file_name"],
                "chunk_id": idx,
                "text": chunk
            })

    embeddings = create_embeddings(all_chunks)

    build_faiss_index(
        embeddings,
        chunks_metadata
    )

st.markdown(
    '<div class="section-title">Document Question Answering</div>',
    unsafe_allow_html=True
)

question = st.text_input(
    "Ask a question about the uploaded documents"
)

if st.button("Ask Question"):

    if question.strip() == "":

        st.warning("Please enter a question.")

    else:

        with st.spinner("Retrieving relevant information..."):

            translated_question = GoogleTranslator(
                source='auto',
                target='en'
            ).translate(question)

            query_embedding = model.encode(
                translated_question
            )

            results = retrieve(
                query_embedding,
                top_k=6
            )

            results = rerank_results(
                results,
                translated_question
            )

        if not results:

            st.error(
                "The documents do not contain enough information."
            )

        else:

            average_confidence = sum(
                result["confidence"]
                for result in results
            ) / len(results)

            if average_confidence < 35:

                st.warning(
                    "Low confidence retrieval detected. Please verify response manually."
                )

            context = ""

            for result in results:

                context += result["text"] + "\n"

            with st.spinner("Generating answer..."):

                answer = generate_answer(
                    question,
                    context
                )

            st.markdown(
                '<div class="section-title">Generated Answer</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
<div class="answer-box">
{answer}
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-title">Retrieved Sources</div>',
                unsafe_allow_html=True
            )

            for result in results:

                st.markdown(
                    f"""
<div class="source-card">

<div class="source-title">
{result['file_name']}
</div>

<b>Chunk ID:</b> {result['chunk_id']}

<div class="score">
Confidence Score: {result['confidence']}%
</div>

<div class="chunk-text">
{result['text'][:500]}...
</div>

</div>
""",
                    unsafe_allow_html=True
                )

st.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-title">Contradiction Analysis</div>',
    unsafe_allow_html=True
)

doc_names = list(set([
    chunk["file_name"]
    for chunk in chunks_metadata
]))

col1, col2 = st.columns(2)

with col1:

    doc1_name = st.selectbox(
        "Select First Document",
        doc_names
    )

with col2:

    doc2_name = st.selectbox(
        "Select Second Document",
        doc_names
    )

topic = st.text_input(
    "Enter topic for contradiction analysis"
)

if st.button("Analyze Contradiction"):

    doc1_text = ""
    doc2_text = ""

    for chunk in chunks_metadata:

        if chunk["file_name"] == doc1_name:
            doc1_text += chunk["text"] + "\n"

        if chunk["file_name"] == doc2_name:
            doc2_text += chunk["text"] + "\n"

    with st.spinner("Analyzing contradictions..."):

        contradiction_result = check_contradiction(
            doc1_text[:4000],
            doc2_text[:4000],
            topic
        )

    st.markdown(
        f"""
<div class="answer-box">
{contradiction_result}
</div>
""",
        unsafe_allow_html=True
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True
)