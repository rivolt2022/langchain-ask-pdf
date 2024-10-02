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
    
    # Add a flag for processing state
    processing = st.session_state.get('processing', False)

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
        
        # Check if we're currently processing a question
        if processing:
            st.write("질문 처리중입니다. 잠시만 기다려주세요...")
        else:
            user_question = st.text_input("PDF에 대해서 질문해주세요:")

            if user_question:
                # Set the processing flag to True
                st.session_state['processing'] = True

                # Process the question in background
                with st.spinner("답변을 생성하는 중입니다..."):
                    docs = knowledge_base.similarity_search(user_question)
                    
                    # Updated to use chat-based LLM
                    llm = ChatOpenAI(model="gpt-4")  # You can use "gpt-4" or "gpt-3.5-turbo"
                    chain = load_qa_chain(llm, chain_type="stuff")
                    
                    with get_openai_callback() as cb:
                        response = chain.run(input_documents=docs, question=user_question)
                        print(cb)
                    
                    st.write(response)
                
                # Set the processing flag back to False
                st.session_state['processing'] = False

if __name__ == '__main__':
    main()
