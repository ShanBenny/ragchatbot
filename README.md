C-CARES RAG Chatbot (Ollama + LangChain + Streamlit)

This project is an intelligent chatbot that answers questions based on the C-CARES User Manual.
It uses Retrieval-Augmented Generation (RAG) with LangChain, Chroma Vector Database, and Ollama (LLaMA-3) for efficient document retrieval and contextual answering.
The frontend is built with Streamlit, offering a clean and professional chat interface.

🧠 Features

Conversational chatbot with context memory

Accurate, document-based answers using RAG

Local embeddings and vector storage with ChromaDB

Integration with Ollama LLaMA-3 model

Professional Streamlit UI with CDAC branding

Displays previous chat history

Persistent vector store for offline use

📁 Project Structure
ragchatbot/
├── app.py                 # Streamlit frontend
├── rag_pipeline.py        # Core RAG backend logic
├── data/
│   └── C-CARES_User_Manual_VN1.0.docx  # Input document
├── vectorstore/           # Chroma database files
├── requirements.txt       # Dependencies
└── README.md              # Project documentation

⚙️ Installation Steps

Clone the Repository

git clone https://github.com/yourusername/ragchatbot.git
cd ragchatbot


Create and Activate Virtual Environment

python -m venv venv
venv\Scripts\activate      # on Windows
source venv/bin/activate   # on Linux/Mac


Install Dependencies

pip install -r requirements.txt


Start Ollama Server

Install Ollama

Run:

ollama run llama3


Index the Document

python -c "from rag_pipeline import load_and_index; load_and_index()"


Run the Streamlit App

streamlit run app.py

🧩 How It Works

The C-CARES user manual (.docx) is split into smaller text chunks.

Each chunk is converted into vector embeddings using the OllamaEmbeddings model.

These embeddings are stored in a Chroma vector database for fast retrieval.

When a user asks a question, the system retrieves the most relevant document chunks.

The LLaMA-3 model (via Ollama) generates an answer based on both the question and retrieved context.

The conversation history is preserved for contextual follow-up questions.

📊 Architecture Overview
User → Streamlit UI → LangChain Pipeline → ChromaDB Retriever → Ollama LLaMA-3 → Response


For a visual overview, see the architecture diagram in /docs/architecture.png.

🧰 Tech Stack

Component	Tool
Frontend	Streamlit
Backend	LangChain
Embeddings	OllamaEmbeddings
Vector Store	ChromaDB
LLM	LLaMA-3 (via Ollama)
Document Loader	Docx2txtLoader
🧑‍💻 Author
Developed by Shan Benny
Apprenticeship Project – C-DAC Bangalore
Group: Big Data and Machine Learning

📄 License

This project is developed for internal learning and research use under C-DAC guidelines.
