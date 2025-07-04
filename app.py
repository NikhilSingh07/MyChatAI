from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template

# extracting raw text from the pdf
def get_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text  


# Dividing the text into chunks
def get_text_chunks(raw_text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_text(raw_text)

    return texts


# convert the chunks into embeddings and store them in vector store

# 1. Create an embedding model instance:
# 2. Find the embedding vector dimension:
# 3. Initialize index:
# 4. Create the vector store:
# 5. Embedding the chunks and adding to the store

def get_vectorstore(text_chunks):

    try:
        # Open ai returns the embedding model object
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key = os.getenv("OPEN_AI_API_KEY"))

        # creating embeddings from text_chunks for testing
        #embeddings = embedding_model.embed_documents(text_chunks)
        
        # The embedding model always outputs vectors of the same length (dimension), no matter what input text you give it.
        embedding_dim = len(embedding_model.embed_query("hello world"))

        index = faiss.IndexFlatL2(embedding_dim)

        vector_store = FAISS(
            embedding_function=embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

        # add texts method internally calls the embedding model to convert the chunks into embeddings and store them in the VS
        vector_store.add_texts(text_chunks)
        print("vector store database has been created and intialised successfully")
        return vector_store

    except Exception as e:  # catch all errors
        print(f"Error while getting embeddings: {e}")
        return None


# a conversation chain, where user input from memory + context will be send to the llm
def get_conversation_chain(vector_store):

    llm = ChatOpenAI( model_name="gpt-4.1-nano-2025-04-14",  openai_api_key= os.getenv("OPEN_AI_API_KEY"))

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain


def handle_user_input(user_input):

    if(st.session_state.conversation is not None):

        response = st.session_state.conversation({'question':user_input})
        st.session_state.chat_history = response['chat_history']

        for index, msg in enumerate(st.session_state.chat_history):
            if(index % 2 == 0):
                #st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html = True)
                message(msg.content, is_user=True, avatar_style="personas")
            else:
                #st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html = True)
                message(msg.content, avatar_style="micah")

    else: message("Hi I'm here to help you but first please upload your documents, thanks! - Your neighbourhood friendly AI",avatar_style="micah")

def main():

    # Load variables from .env into os.environ
    load_dotenv()

    st.markdown(css, unsafe_allow_html=True)

    # creating a session state for conversation chain
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_histiry = None    

    st.set_page_config(page_title="Chat with PDFs",page_icon=":books:")

    with st.sidebar:
        st.subheader("Your documents")

        pdf_docs = st.file_uploader("Upload your pdfs here and click on 'Process' button", accept_multiple_files=True)

        if(st.button("Process")):
           if(pdf_docs):
               with st.spinner("Processing"):

                    # get pdf text
                    raw_text = get_text_from_pdf(pdf_docs)

                    # Divide them into chunks
                    text_chunks = get_text_chunks(raw_text)
                    #st.write(text_chunks)

                    # convert them in embeddings and store in the vector store
                    vector_store= get_vectorstore(text_chunks)

                    # creating a conversation chain between human and ai
                    if vector_store is not None:
                        st.session_state.conversation = get_conversation_chain(vector_store)
                        message("Hey, I'm here to help you. Ask me anything from your documents", avatar_style="micah")
                    else:
                        print("No vector store present")    
           else:  
               st.write("Please upload atleast 1 pdf");  
        

    # Input form fixed at bottom
    with st.form("chat_input_form", clear_on_submit=True):
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([10, 1])
        user_input = col1.text_input("Your message", label_visibility="collapsed", placeholder="Ask anything from the document...")
        send_clicked = col2.form_submit_button("➤")
        st.markdown('</div>', unsafe_allow_html=True)

    if user_input and send_clicked:
        handle_user_input(user_input)

if __name__ == '__main__':
    main()
