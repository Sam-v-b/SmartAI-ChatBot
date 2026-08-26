# 🤖 SmartAI ChatBot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-00BFFF?style=for-the-badge)

*AI-powered PDF & CSV question-answering chatbot built with Streamlit, LangChain, and OpenAI*

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [RAG Pipeline](#-rag-pipeline)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Usage Guide](#-usage-guide)
- [Data Flow Diagrams](#-data-flow-diagrams)

---

## 🧠 Overview

**SmartAI ChatBot** is an intelligent document Q&A assistant that lets you upload PDF files and ask natural-language questions about their contents. It uses a **Retrieval-Augmented Generation (RAG)** pipeline — splitting documents into chunks, embedding them into a FAISS vector store, and fetching the most relevant context before generating precise answers with GPT-3.5 Turbo or GPT-4.

A companion **CSV Agent** module provides natural-language querying over tabular data files.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Ingestion | Upload and parse multi-page PDFs via PyPDF2 |
| 🔍 Semantic Search | FAISS vector store with OpenAI embeddings for similarity search |
| 🤖 GPT-powered Q&A | LCEL chain with configurable model (gpt-3.5-turbo / gpt-4) |
| 🌡️ Temperature Control | Adjustable creativity slider (0.0 – 1.0) |
| 📊 CSV Agent | Natural-language querying over tabular CSV files |
| 💬 Chat History | Full conversation memory within a session |
| 📤 History Export | Export chat as JSON, TXT, or CSV |
| 🔗 Source Citations | Each answer includes the source chunks used for context |
| ⚙️ Configurable Chunking | Adjustable chunk size (500–2000 tokens) and source count (1–5) |

---

## 🛠️ Tech Stack

### Core Framework
| Layer | Technology |
|---|---|
| UI / Server | Streamlit |
| LLM Orchestration | LangChain (LCEL) |
| LLM Provider | OpenAI (gpt-3.5-turbo, gpt-4) |

### RAG Pipeline
| Component | Technology |
|---|---|
| PDF Parsing | PyPDF2 |
| Text Chunking | LangChain CharacterTextSplitter |
| Embeddings | OpenAIEmbeddings |
| Vector Store | FAISS (faiss-cpu) |
| Output Parsing | LangChain StrOutputParser |

### Data & Utilities
| Component | Technology |
|---|---|
| Tabular Q&A | langchain-experimental CSV Agent |
| Env Management | python-dotenv |
| Runtime | Python 3.11 |

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph Client["🖥️ Streamlit Frontend"]
        UI["Chat Interface"]
        Sidebar["⚙️ Sidebar Controls\n(Model / Temp / Chunk Size)"]
        Upload["📎 File Uploader"]
        Export["📤 Export Chat"]
    end

    subgraph RAG["🔍 RAG Pipeline"]
        Parser["PyPDF2\nPDF Parser"]
        Splitter["CharacterTextSplitter\nChunking"]
        Embedder["OpenAIEmbeddings\nEmbedding"]
        VectorDB["FAISS\nVector Store"]
        Retriever["Similarity Search\nTop-K Chunks"]
    end

    subgraph LLM["🤖 LLM Layer"]
        Chain["LCEL Chain\nPromptTemplate → OpenAI → StrOutputParser"]
        GPT["OpenAI GPT\n(gpt-3.5-turbo / gpt-4)"]
    end

    subgraph State["🗂️ Session State"]
        ChatHistory["chat_history"]
        KB["knowledge_base\n(FAISS index)"]
        PDFName["pdf_name"]
    end

    subgraph CSV["📊 CSV Agent"]
        CSVLoader["CSV File Upload"]
        CSVAgent["create_csv_agent\n(langchain-experimental)"]
    end

    Upload --> Parser
    Parser --> Splitter
    Splitter --> Embedder
    Embedder --> VectorDB
    VectorDB --> KB

    UI -- "User Question" --> Retriever
    KB --> Retriever
    Retriever -- "Context Chunks" --> Chain
    Chain --> GPT
    GPT -- "Answer + Sources" --> UI
    UI --> ChatHistory

    Sidebar --> Chain
    Export --> ChatHistory
    CSVLoader --> CSVAgent
    CSVAgent --> GPT
```

---

## 🔄 RAG Pipeline

```mermaid
flowchart LR
    A["📄 PDF Upload"] --> B["PyPDF2\nText Extraction"]
    B --> C["CharacterTextSplitter\nchunk_size: 500–2000\nchunk_overlap: 200"]
    C --> D["OpenAIEmbeddings\ntext-embedding-ada-002"]
    D --> E["FAISS\nVector Store\n(in-memory)"]

    F["❓ User Question"] --> G["Query Embedding"]
    G --> H["similarity_search\nTop-K docs"]
    E --> H

    H --> I["PromptTemplate\nContext + Question"]
    I --> J["OpenAI LLM\ngpt-3.5-turbo / gpt-4"]
    J --> K["StrOutputParser"]
    K --> L["💬 Answer + Source Citations"]
```

---

## 📂 Project Structure

```
SmartAI-ChatBot/
├── app.py                  # Main Streamlit PDF chatbot application
├── csv_agent.py            # CSV Q&A agent (separate Streamlit app)
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python runtime version (3.11)
├── .env                    # Environment variables (not committed)
├── .streamlit/
│   └── config.toml         # Streamlit config (maxUploadSize = 102400 MB)
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.11
- An **OpenAI API key** — [Get one here](https://platform.openai.com/api-keys)
- `pip` package manager

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/SmartAI-ChatBot.git
cd SmartAI-ChatBot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...your-key-here...
```

### 5. Run the Application

**PDF Chatbot:**
```bash
streamlit run app.py
```

**CSV Agent:**
```bash
streamlit run csv_agent.py
```

Open your browser at `http://localhost:8501`

---

## 🔧 Configuration

### Sidebar Controls (`app.py`)

| Control | Default | Range | Description |
|---|---|---|---|
| Model | gpt-3.5-turbo | gpt-3.5-turbo, gpt-4 | LLM model to use |
| Temperature | 0.7 | 0.0 – 1.0 | Response creativity / randomness |
| Chunk Size | 1000 | 500 – 2000 | Token size per document chunk |
| Num Sources | 3 | 1 – 5 | Number of context chunks retrieved |

### Streamlit Config (`.streamlit/config.toml`)

```toml
[server]
maxUploadSize = 102400   # Max file upload size in MB
```

---

## 📖 Usage Guide

### PDF Q&A

1. Launch `app.py` and open the browser
2. Upload a PDF using the file uploader
3. Wait for the document to be processed (chunked + embedded)
4. Type your question in the chat input
5. The bot answers using retrieved context chunks, with source citations below each response
6. Export the full chat via **Export as JSON / TXT / CSV** buttons

### CSV Q&A

1. Launch `csv_agent.py`
2. Upload a `.csv` file
3. Ask natural-language questions about the data (e.g., *"What is the average sales in Q3?"*)
4. The LangChain CSV Agent generates and executes pandas code to answer

---

## 📊 Data Flow Diagrams

### Chat Interaction Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant KB as FAISS Knowledge Base
    participant LLM as OpenAI GPT

    User->>UI: Upload PDF
    UI->>UI: PyPDF2 → Chunk → Embed → FAISS
    UI-->>User: "PDF processed! Ask a question."

    User->>UI: Type question
    UI->>KB: similarity_search(question, k=N)
    KB-->>UI: Top-K relevant chunks
    UI->>LLM: PromptTemplate(context, question)
    LLM-->>UI: Generated answer
    UI-->>User: Display answer + source citations
    UI->>UI: Append to chat_history (session state)

    User->>UI: Click "Export Chat"
    UI-->>User: Download JSON / TXT / CSV
```

### CSV Agent Flow

```mermaid
sequenceDiagram
    actor User
    participant UI as Streamlit UI
    participant Agent as CSV Agent
    participant LLM as OpenAI GPT

    User->>UI: Upload CSV file
    User->>UI: Ask question about data
    UI->>Agent: create_csv_agent(llm, csv_path)
    Agent->>LLM: Generate pandas code for question
    LLM-->>Agent: Python code
    Agent->>Agent: Execute code on DataFrame
    Agent-->>UI: Result
    UI-->>User: Display answer
```

---

## 🌐 Deployment

### Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `OPENAI_API_KEY` in **Secrets** settings
5. Deploy — your app goes live instantly

### Local Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t smartai-chatbot .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... smartai-chatbot
```

---

## 📦 Key Dependencies

```
streamlit               # Web UI framework
langchain               # LLM orchestration
langchain-openai        # OpenAI LLM + embeddings
faiss-cpu               # Vector similarity search
PyPDF2                  # PDF text extraction
langchain-experimental  # CSV agent
python-dotenv           # .env file loading
tiktoken                # Token counting
pandas                  # Tabular data handling
```

---

<div align="center">
Built with ❤️ using Streamlit + LangChain + OpenAI
</div>
