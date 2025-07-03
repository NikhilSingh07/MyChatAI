import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

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

    except Exception as e:  # catch all errors
        print(f"Error while getting embeddings: {e}")
        return None
       

def main():

    # Load variables from .env into os.environ
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs",page_icon=":books:")
    st.header("Upload your PDF docs and ask anything :books:!")
    st.text_input("Ask anything from your documents")

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
                    st.write(text_chunks)


                    # convert them in embeddings and store in the vector store
                    get_vectorstore(text_chunks)

           else:  
               st.write("Please upload atleast 1 pdf");  
    
    

if __name__ == '__main__':
    main()
