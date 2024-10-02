from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI  # Updated for chat-based models
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    st.set_page_config(page_title="연세대학교 행정혁신")
    st.header("연세대학교 행정혁신 예제챗봇💬")

    # Initialize session state for processing flag
    if 'processing' not in st.session_state:
        st.session_state['processing'] = False

    # upload file
    pdf = st.file_uploader("PDF를 업로드해주세요.", type="pdf")
    
    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Input box with disabled option based on processing state
        user_question = st.text_input(
            "PDF에 대해서 질문해주세요:",
            disabled=st.session_state['processing']
        )

        if user_question and not st.session_state['processing']:
            # Set processing flag to True to disable input
            st.session_state['processing'] = True

            # Process the question in the background
            with st.spinner("답변을 생성하는 중입니다..."):
                docs = knowledge_base.similarity_search(user_question)
                
                # Use chat-based LLM (e.g., gpt-3.5-turbo)
                llm = ChatOpenAI(model="gpt-3.5-turbo")  # Changed to gpt-3.5-turbo
                chain = load_qa_chain(llm, chain_type="stuff")
                
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)
                
                st.write(response)
            
            # After processing is done, reset processing flag to False
            st.session_state['processing'] = False

if __name__ == '__main__':
    main()
