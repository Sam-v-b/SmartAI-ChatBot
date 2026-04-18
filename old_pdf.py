from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback   #This function is used to get a callback for interacting with OpenAI models.

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your PDF")    # configures the Streamlit page with a specific title.
    st.header("Ask your PDF 💬")

    # uploading file
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    # custom check for file size
    max_size = 1024 * 1024 * 1024  # 1GB in bytes
    if uploaded_file is not None and uploaded_file.size > max_size:
        st.warning(f"File size exceeds 1GB limit. Please upload a file smaller than 1GB.")
        return

    # reading the contents ( extracts the data page wise )
    if uploaded_file is not None:
        st.write(uploaded_file)
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split input data into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,  # character
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        # create embeddings -> encode of info for effective search
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # create input field for asking questions
        response = ""
        user_question = st.text_input("Ask a question about your PDF: ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()  # large language model - openai in our case
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
        elif user_question == "":
            st.write("Please enter a valid question.")

        # If no answer found, provide a link to search for the information
        if not response:
            st.warning("No answer found in the PDF.")
            st.write("You can search for more information at https://www.google.com")

        st.write(response)

if __name__ == "__main__":
    main()




'''from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback   #This function is used to get a callback for interacting with OpenAI models.

def main():
    load_dotenv()

    st.set_page_config(page_title="Ask your PDF")    # configures the Streamlit page with a specific title.
    st.header("Ask your PDF 💬")

    # uploading file
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

    # custom check for file size
    max_size = 1024 * 1024 * 1024  # 1GB in bytes
    if uploaded_file is not None and uploaded_file.size > max_size:
        st.warning(f"File size exceeds 1GB limit. Please upload a file smaller than 1GB.")
        return

    # Ask for a valid question after uploading the PDF
    if uploaded_file is not None:
        st.write(uploaded_file)
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split input data into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,  # character
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        # create embeddings -> encode of info for effective search
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # create input field for asking questions
        response = ""
        user_question = st.text_input("Ask a question about PDF ")
        if user_question == "":
            st.warning("Please provide a valid question.")
            return

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI()  # large language model - openai in our case
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

        # If no answer found, provide a message and Google link
        if not response:
            st.warning("No answer found in the PDF.")
            st.markdown("You can search for more information on [Google](https://www.google.com)")

        st.write(response)

if __name__ == "__main__":
    main()
'''