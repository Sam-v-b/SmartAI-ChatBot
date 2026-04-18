import streamlit as st
from langchain_experimental.agents import create_csv_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
def main():
    load_dotenv()
    print(os.getenv("OPENAI_API_KEY"))

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📊")

    # uploading csv file
    user_csv = st.file_uploader("Upload your CSV file", type="csv")

    if user_csv is not None:
        st.write(user_csv)
        # taking user input
        user_question = st.text_input("Ask a question about your CSV: ")
        
        llm = OpenAI(temperature=0)
        agent = create_csv_agent(llm, user_csv, verbose=True)

        # getting the answer and display it to output
        if user_question is not None and user_question != "":
            response = agent.run(user_question)
            st.write(response)


if __name__ == "__main__":
    main()