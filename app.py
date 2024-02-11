import os
import pinecone

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st
from docx import Document
import textract

st.set_page_config(page_title="chatbot")
st.title("Chat with Documents")


num_of_top_selection = 3
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
embedding_dim = 768

# Initialize Pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "qp-ai-assessment"


def recreate_index():
    # Check if the index exists, and delete it if it does
    existing_indexes = pc.list_indexes().names()
    print(existing_indexes)
    if index_name in existing_indexes:
        pc.delete_index(index_name)
        print(f"Deleted existing index: {index_name}")

    # Create a new index
    pc.create_index(
        name=index_name,
        metric='cosine',
        dimension=embedding_dim,
        spec=pinecone.PodSpec(os.getenv("PINECONE_ENV"))  # 1536 dim of text-embedding-ada-002
    )
    print(f"Created new index: {index_name}")

def get_text_from_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_from_docx(docx):
    doc = Document(docx)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def get_text_from_text_file(text_file):
    with open(text_file, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def get_text_from_other_file(file_path):
    try:
        text = textract.process(file_path, method='pdftotext').decode('utf-8')
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def load_documents(docs):
    text = ""
    for doc in docs:
        if doc.name.lower().endswith('.pdf'):
            text += get_text_from_pdf(doc)
        elif doc.name.lower().endswith('.docx'):
            text += get_text_from_docx(doc)
        elif doc.name.lower().endswith(('.txt', '.md')):
            text += get_text_from_text_file(doc)
        else:
            # Handle other file types, you can extend this as needed
            text += get_text_from_other_file(doc)

    return text


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_text(documents)
    return text_splitter.create_documents(texts)


def embeddings_on_pinecone(texts):
    # Use HuggingFace embeddings for transforming text into numerical vectors
    embeddings = HuggingFaceEmbeddings()
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=st.session_state.pinecone_index)
    retriever = vectordb.as_retriever(search_kwargs={'k': num_of_top_selection})
    return retriever

def query_llm(retriever, query):
    #llm = OpenAIChat(openai_api_key=st.session_state.openai_api_key)
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def input_fields():
    #
    with st.sidebar:
        
        st.session_state.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        # st.text_input("Pinecone API key", type="password")
        st.session_state.pinecone_env = os.getenv("PINECONE_ENV")
        # st.text_input("Pinecone environment")
        st.session_state.pinecone_index = index_name
        # st.text_input("Pinecone index name")
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    #


def process_documents():

    if not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            # for source_doc in st.session_state.source_docs:
            if st.session_state.source_docs:
                #
                # recreate_index()

                documents = load_documents(st.session_state.source_docs)

                #
                texts = split_documents(documents)
                #
                st.session_state.retriever = embeddings_on_pinecone(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    #
    input_fields()
    #
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()
    
