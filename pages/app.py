import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback


def process_text(text):
    # Use Langchain's CharacterTextSplitter to Split the text into chunks
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)

    # Create embeddings for the chunks using a Hugging Face sentence transformer
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    # Create a FAISS vector store (knowledge base) from the embeddings
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase

def main():
    st.title("Summarize PDF by Umme Shafiya Mubeen")
    st.divider()

    try:
        dotenv_path = "openai.env"
        load_dotenv(dotenv_path)
        openApi_key = os.getenv("openApi_key")
        if not openApi_key:
            raise ValueError(f"Please Provide open API key {dotenv_path}")
        os.environ["openApi_key"] = openApi_key
    except ValueError as e:
        st.error(str(e))
        return

    pdf = st.file_uploader('Please Upload your Document', type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # call function to generate knowledge base of provided text pdf
        knowledgeBase = process_text(text)

        query = "Summarize the content of the uploaded PDF in 3-5 sentences, capturing the core arguments, objectives, and any significant findings or recommendations made by the author."

        if query:
            docs = knowledgeBase.similarity_search(query)
            OpenAIModel = "gpt-3.5-turbo-16k"
            llm = ChatOpenAI(model=OpenAIModel, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')

            with get_openai_callback() as cost:
                response = chain.run(input_documents=docs, question=query)
                print(cost)

            st.subheader('Summary Of PDF:')
            st.write(response)

if __name__ == '__main__':
    main()

