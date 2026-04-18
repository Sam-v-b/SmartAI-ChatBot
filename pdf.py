from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from datetime import datetime
import json
import io
import csv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SmartAI ChatBot - Ask your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Global Styles */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #00d4ff;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --light-bg: #f8fafc;
        --dark-bg: #1e293b;
    }
    
    /* Main Container */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .header-container h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-container p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Card Styling */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Message Styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        color: #1e293b;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stSidebar [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Badge Styling */
    .source-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        margin: 0.3rem;
        display: inline-block;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }
    
    /* Input Styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.8rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #e2e8f0 100%);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #cbd5e1;
    }
    
    /* Metric Styling */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
    }
    
    /* Divider */
    .stDivider {
        border-top: 2px solid #e2e8f0;
        margin: 1.5rem 0;
    }
    
    /* Success Messages */
    .stSuccess {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border-radius: 12px;
        color: white;
    }
    
    /* Warning Messages */
    .stWarning {
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%) !important;
        border-radius: 12px;
        color: white;
    }
    
    /* Info Messages */
    .stInfo {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border-radius: 12px;
        color: white;
    }
    
    /* Error Messages */
    .stError {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border-radius: 12px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")
    
    # Model Settings
    st.markdown("### 🤖 Model Settings")
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help="Higher = more creative answers\nLower = more focused answers"
    )
    
    model = st.selectbox(
        "Select Model",
        ["gpt-3.5-turbo", "gpt-4"],
        help="Choose your AI model:\n• gpt-3.5-turbo: Fast & Affordable\n• gpt-4: Powerful & Accurate"
    )
    
    st.markdown("---")
    
    # RAG Settings
    st.markdown("### 📄 Document Settings")
    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100,
        help="Size of text segments for analysis"
    )
    
    num_sources = st.slider(
        "Number of Sources",
        min_value=1,
        max_value=5,
        value=3,
        help="Relevant text chunks to retrieve"
    )
    
    st.markdown("---")
    
    # Chat history actions
    st.markdown("### 💾 History & Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.knowledge_base = None
            st.session_state.pdf_name = None
            st.rerun()
    
    with col2:
        if st.button("📊 Stats", use_container_width=True):
            with st.expander("View Statistics"):
                st.metric("Total Q&A", len(st.session_state.chat_history))
                if st.session_state.pdf_name:
                    st.metric("Current File", st.session_state.pdf_name[:30])
    
    st.markdown("---")
    st.markdown("**SmartAI ChatBot v1.0**")
    st.caption("Powered by LangChain & OpenAI")

# Main Content Area
st.markdown("""
    <div class="header-container">
        <h1>📄 SmartAI ChatBot</h1>
        <p>Intelligent PDF Analysis & Question Answering System</p>
    </div>
""", unsafe_allow_html=True)

# File uploader
st.markdown("### 📤 Upload Your PDF Document")
uploaded_file = st.file_uploader("Select a PDF file", type="pdf", help="Upload a PDF document to analyze")

# Custom check for file size
max_size = 1024 * 1024 * 1024  # 1GB in bytes
if uploaded_file is not None and uploaded_file.size > max_size:
    st.error(f"❌ File Exceeds Limit: {uploaded_file.size / (1024**3):.2f}GB (Max: 1GB)")

elif uploaded_file is not None:
    # Check if we need to process a new file
    if st.session_state.pdf_name != uploaded_file.name or st.session_state.knowledge_base is None:
        st.session_state.pdf_name = uploaded_file.name
        
        with st.spinner("🔄 Processing PDF..."):
            # Extract text from PDF
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=chunk_size,
                chunk_overlap=200,
                length_function=len
            )
            
            chunks = text_splitter.split_text(text)
            
            # Create embeddings
            embeddings = OpenAIEmbeddings()
            st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            st.success(f"✅ PDF processed! ({len(chunks)} chunks created)")
    
    # Display current PDF info
    st.markdown("### 📋 Document Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <strong>📄 File Name</strong><br>
            {uploaded_file.name[:30]}...
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <strong>📊 File Size</strong><br>
            {uploaded_file.size / 1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)
    with col3:
        status = "✓ Ready" if st.session_state.knowledge_base else "⟳ Processing..."
        st.markdown(f"""
        <div class="metric-card">
            <strong>🔖 Status</strong><br>
            {status}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Question input
    st.markdown("### 💬 Ask Your Question")
    user_question = st.text_input(
        "Enter your question about the PDF",
        placeholder="e.g., What is the main topic of this document?",
        help="Ask anything related to your PDF content"
    )
    
    # Validate question
    if user_question and not re.match(r'^[a-zA-Z\s?!.,;:\'-]+$', user_question):
        st.warning("⚠️ Please provide a valid question (letters, spaces, and basic punctuation only).")
    elif user_question:
        # Search for relevant documents
        docs = st.session_state.knowledge_base.similarity_search(user_question, k=num_sources)
        
        if not docs:
            st.warning("❌ No relevant information found in the document.")
        else:
            # Create QA chain using LCEL
            llm = OpenAI(temperature=temperature, model=model)
            
            template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Helpful Answer:"""
            
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            chain = prompt | llm | StrOutputParser()
            
            # Prepare context
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])
            
            with st.spinner("🤔 Generating answer..."):
                response = chain.invoke({
                    "context": context,
                    "question": user_question
                })
            
            # Store in chat history
            chat_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": user_question,
                "answer": response,
                "sources": [doc.page_content[:200] for doc in docs],
                "temperature": temperature,
                "model": model
            }
            st.session_state.chat_history.insert(0, chat_entry)
            
            # Display answer with sources
            st.markdown("### 💡 Answer")
            st.markdown(f"""
            <div class="assistant-message">
                <strong>❓ Question:</strong> {user_question}<br><br>
                <strong>✓ Answer:</strong> {response}
            </div>
            """, unsafe_allow_html=True)
            
            # Display sources with citations
            st.markdown("### 📚 Referenced Sources")
            for i, doc in enumerate(docs, 1):
                with st.expander(f"📖 Source {i} - Preview"):
                    st.markdown(f"""
                    <div class="card">
                        {doc.page_content[:400]}...
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption(f"🤖 Model: {model} | 🌡️ Temperature: {temperature}")
    
    # Display Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📖 Chat History")
        
        # Export options
        st.markdown("**💾 Export Your Conversation**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Export as JSON"):
                json_str = json.dumps(st.session_state.chat_history, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📥 Export as TXT"):
                txt_content = ""
                for i, chat in enumerate(st.session_state.chat_history, 1):
                    txt_content += f"Question {i}: {chat['question']}\n"
                    txt_content += f"Answer: {chat['answer']}\n"
                    txt_content += f"Time: {chat['timestamp']}\n"
                    txt_content += "-" * 80 + "\n\n"
                
                st.download_button(
                    label="Download TXT",
                    data=txt_content,
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col3:
            if st.button("📥 Export as CSV"):
                csv_buffer = io.StringIO()
                writer = csv.writer(csv_buffer)
                writer.writerow(["#", "Timestamp", "Question", "Answer", "Model", "Temperature"])
                
                for i, chat in enumerate(st.session_state.chat_history, 1):
                    writer.writerow([
                        i,
                        chat['timestamp'],
                        chat['question'],
                        chat['answer'][:200],
                        chat['model'],
                        chat['temperature']
                    ])
                
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Display history with collapsible items
        st.markdown("**📝 Conversation Records**")
        st.markdown("")
        for i, chat in enumerate(st.session_state.chat_history, 1):
            with st.expander(f"**💬 Q{i}:** {chat['question'][:60]}...", expanded=False):
                st.markdown(f"""
                <div class="card">
                    <strong>❓ Question:</strong><br>
                    {chat['question']}<br><br>
                    <strong>✓ Answer:</strong><br>
                    {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"🕐 {chat['timestamp']}")
                with col2:
                    st.caption(f"🤖 {chat['model']} | 🌡️ {chat['temperature']}")

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h2>👆 Get Started</h2>
        <p>Upload a PDF document above to begin analyzing with AI</p>
    </div>
    """, unsafe_allow_html=True)

# Professional Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("📚 Powered by OpenAI GPT")
with col2:
    st.caption("🔐 Secure & Private")
with col3:
    st.caption("⚡ Built with Streamlit & LangChain")
