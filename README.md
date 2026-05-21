# Enterprise AI Knowledge Assistant

This project was built as part of the Potens AI/ML Internship Assignment 2026. Instead of building a generic chatbot, I wanted to create something that felt closer to a real internal enterprise knowledge assistant — a system that can search across documents, answer questions with citations, support multilingual queries, and avoid hallucinating information when the answer is not present in the documents.

A multilingual Retrieval-Augmented Generation (RAG) system built for the Potens AI/ML Internship Assignment 2026.

The system allows users to query enterprise documents in multiple languages, retrieve semantically relevant chunks using vector search, generate citation-based answers, detect contradictions between documents, and prevent hallucinated responses.

---

# Features

* Multilingual document querying (English + Hindi supported)
* Semantic search using Sentence Transformers + FAISS
* Citation-based responses with chunk references
* Contradiction analysis between documents
* Hallucination protection with low-confidence safeguards
* Retrieval reranking layer for improved relevance
* Confidence scoring for retrieved chunks
* Modern enterprise-style Streamlit UI
* Human-review recommendation for low-confidence retrievals

---

# Tech Stack

| Component       | Technology            |
| --------------- | --------------------- |
| Frontend UI     | Streamlit             |
| Embeddings      | Sentence Transformers |
| Vector Database | FAISS                 |
| LLM             | OpenRouter API        |
| Translation     | deep-translator       |
| Language        | Python                |

---

# Architecture Overview

```text
User Query
   ↓
Language Translation Layer
   ↓
Sentence Embedding Generation
   ↓
FAISS Semantic Retrieval
   ↓
Reranking Layer
   ↓
Context Assembly
   ↓
LLM Response Generation
   ↓
Citation-Based Answer
```

---

# Chunking Strategy

One of the main goals while building this project was improving retrieval quality instead of simply passing entire documents to the model. Large PDFs were split into smaller semantic chunks before embedding.

I used overlapping chunks so that important context would not be lost between sections. This helped improve semantic retrieval and reduced noisy responses from the LLM.

The uploaded PDFs are parsed and divided into smaller semantic chunks before embedding.

Why chunking?

* Improves retrieval quality
* Reduces irrelevant context
* Prevents token overload
* Improves semantic matching accuracy

The system uses fixed-size overlapping chunks to preserve contextual continuity across sections.

---

# Multilingual Retrieval Strategy

Most uploaded documents were in English, but the assignment required multilingual support. Instead of translating the documents themselves, I implemented a lightweight translation layer at the query boundary.

The user’s query is translated into English before retrieval, semantic search happens on the English embeddings, and the final response is generated in the same language as the original question.

This approach kept the pipeline simple while still supporting multilingual interaction.

The uploaded documents are primarily in English.

To support multilingual queries:

1. User query is translated into English using GoogleTranslator.
2. Semantic retrieval happens on English embeddings.
3. The LLM generates answers in the original query language.

This approach follows the assignment guideline that translation at the boundary is acceptable.

---

# Hallucination Prevention

A major focus of this project was preventing silent hallucinations.

Instead of forcing the model to answer every question, the system checks retrieval quality first. If relevant context is not found in the documents, the application explicitly responds that the documents do not contain enough information.

I also added confidence scoring and a low-confidence warning system to make the behavior more transparent.

The system avoids silent hallucinations.

If relevant information is not retrieved from the documents, the application explicitly returns:

```text
The documents do not contain enough information.
```

Additionally:

* Confidence scores are calculated
* Low-confidence retrievals trigger a human-review recommendation

---

# Retrieval Reranking

FAISS retrieval alone sometimes returned chunks that were semantically similar but not actually the most useful.

To improve this, I added a lightweight reranking layer that compares keyword overlap between the query and retrieved chunks. The reranker then prioritizes the most relevant chunks before sending context to the LLM.

This was a simple addition, but it noticeably improved retrieval quality.

After FAISS retrieval, a lightweight reranking layer improves relevance.

The reranker:

* compares query keyword overlap
* assigns rerank scores
* prioritizes semantically stronger chunks

This improves retrieval quality while keeping the architecture lightweight.

---

# Contradiction Analysis

Along with normal question answering, I added a contradiction analysis feature.

Users can select two documents and provide a topic for comparison. The system then analyzes whether the documents conflict with each other and generates reasoning for the decision.

This feature was added to make the project feel more like a real enterprise knowledge-review tool instead of only a basic chatbot.

The application includes a contradiction checker.

Users can:

* select two documents
* provide a topic
* analyze whether the documents contradict each other

The LLM returns:

* contradiction status
* reasoning

---

# Screenshots

## Home Interface

Add screenshot here:

```text
screenshots/home.jpeg
```

---

## Multilingual Query Example

Add screenshot here:

```text
screenshots/hindi-query.jpeg
```

---

## Citation-Based Retrieval

Add screenshot here:

```text
screenshots/sol1.jpeg
```

```text
screenshots/sol2.jpeg
```

---

## Contradiction Analysis

Add screenshot here:

```text
screenshots/contradiction.jpeg
```

```text
screenshots/contradiction2.jpeg
```
---

# How to Run

## 1. Clone Repository

```bash
git clone <your-repository-link>
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate Environment

### Windows

```bash
venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Add Environment Variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_api_key
```

---

## 6. Run Application

```bash
streamlit run app.py
```

---

# Example Questions

## English

```text
What are the responsibilities of the Chief Executive Officer?
```

```text
What happens if SAP fails to maintain availability?
```

## Hindi

```text
मानव संसाधन नीति क्या है?
```

```text
SAP agreement में availability guarantee क्या है?
```

---

# Project Structure

```text
potens-intern-ai-ml-apurva-dighe/
│
├── documents/
├── utils/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── llm.py
│   ├── contradict.py
│   └── pdf_loader.py
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

---

# Limitations

* Retrieval quality depends on chunk relevance
* Translation layer may occasionally reduce semantic precision
* Contradiction analysis depends on LLM reasoning quality
* No persistent database storage in current version

---

# Future Improvements

If I had more time, I would improve the project further by:

* adding hybrid retrieval using BM25 + vector search

* using a stronger reranker model

* adding persistent database storage

* improving contradiction visualization

* adding conversation memory

* supporting more Indian languages

* adding evaluation metrics for retrieval quality

* Add hybrid retrieval (BM25 + vector search)

* Add advanced reranking models

* Add persistent vector database support

* Add user authentication

* Add streaming responses

* Add conversation memory

* Improve contradiction visualization

---

# AI Use Log

| Tool           | Approx Usage | Purpose                                                                             


| ChatGPT        | Moderate        | Debugging, multilingual retrieval, UI improvements, contradiction analysis |

| GitHub Copilot | Minimal      | Small code suggestions                                                                        
|

| OpenRouter     | Minimal     | LLM-based answer generation                                                                     |

---

# Design Decisions

Some of the major decisions while building this project:

* FAISS was chosen because it is lightweight, fast, and simple to integrate.

* Multilingual embeddings were used to improve cross-language retrieval.

* A translation boundary approach was used instead of translating all documents.

* Confidence scoring and hallucination safeguards were added because enterprise systems should fail safely instead of generating misleading answers.

* The UI was intentionally designed to feel minimal and enterprise-oriented rather than overly flashy.

* Chose FAISS because it is lightweight and fast for semantic retrieval.

* Chose multilingual embeddings to support Hindi queries.

* Added reranking to improve retrieval quality.

* Added hallucination safeguards to prevent misleading answers.

* Built an enterprise-style UI to align with the assignment tone.

---

# Assignment Alignment

This project satisfies:

* Document ingestion and chunking
* Vector embeddings and storage
* Citation-based responses
* Contradiction detection
* Multilingual querying
* Hallucination prevention
* Streamlit UI
* Confidence scoring
* Retrieval reranking

---

# Author

Apurva Dighe
