from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

DOC_PATH = r"C:\Users\Hp\Desktop\ragchatbot\data\C-CARES_User_Manual_VN1.0.docx"
CHROMA_PATH = "vectorstore"

def load_and_index():
    """Load DOCX manual, split into chunks, and persist embeddings in Chroma."""
    loader = Docx2txtLoader(DOC_PATH)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    vectordb.persist()

def get_qa_chain():
    embeddings = OllamaEmbeddings(model="llama3")
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    llm = OllamaLLM(model="llama3")

    # Memory for conversation context
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"   # 👈 specify which output to store
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        return_source_documents=True,  # keep docs if you need them
        output_key="answer"            # 👈 must match memory
    )
    return qa
