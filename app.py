from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI  # 대화 기반 모델 사용을 위한 업데이트
from langchain.callbacks import get_openai_callback

def main():
    # 환경 변수를 불러옵니다 (예: OpenAI API 키)
    load_dotenv()

    # Streamlit 페이지 설정 (페이지 제목 설정)
    st.set_page_config(page_title="연세대학교 행정 혁신")
    st.header("연세대학교 행정 혁신 예제 챗봇💬")  # 페이지 헤더 설정

    # 세션 상태 초기화 (처리 중 여부를 확인하기 위한 플래그)
    if 'processing' not in st.session_state:
        st.session_state['processing'] = False

    # 파일 업로드 섹션 (PDF 파일 업로드)
    pdf = st.file_uploader("행정 매뉴얼을 PDF로 업로드해주세요.", type="pdf")
    
    # PDF 파일이 업로드된 경우
    if pdf is not None:
        pdf_reader = PdfReader(pdf)  # PDF 리더를 사용하여 PDF 파일 읽기
        text = ""
        # 각 페이지의 텍스트를 추출
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # 텍스트를 청크 단위로 분할
        text_splitter = CharacterTextSplitter(
            separator="\n",  # 줄바꿈을 기준으로 분할
            chunk_size=1000,  # 각 청크의 최대 크기 설정
            chunk_overlap=200,  # 청크 간 중첩 설정
            length_function=len  # 텍스트 길이 계산 함수
        )
        chunks = text_splitter.split_text(text)  # 텍스트를 청크로 분할
        
        # 임베딩 생성 (OpenAI 임베딩 사용)
        embeddings = OpenAIEmbeddings()
        # FAISS를 사용하여 청크로부터 벡터 데이터베이스(knowledge base) 생성
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # 사용자 질문을 입력받는 폼 생성
        with st.form(key="user_question_form", clear_on_submit=True):
            user_question = st.text_input(
                "PDF에 대해서 질문해주세요:",  # 질문 입력창 레이블
                disabled=st.session_state['processing']  # 처리 중일 때 입력 비활성화
            )
            # 폼 제출 버튼
            submit_button = st.form_submit_button(label="질문 제출")

        # 폼이 제출되었고, 질문이 입력되었으며 현재 처리가 진행 중이 아닌 경우
        if submit_button and user_question and not st.session_state['processing']:
            # 처리 중 플래그를 True로 설정하여 입력 비활성화
            st.session_state['processing'] = True

            # 질문에 대한 답변을 생성하는 중 스피너 표시
            with st.spinner("답변을 생성하는 중입니다. 답변 생성중 '질문 제출' 하지 말아주세요."):
                # 질문에 대한 유사도 검색 수행
                docs = knowledge_base.similarity_search(user_question)
                
                # 대화 기반 LLM 모델 사용 (예: GPT-4-turbo)
                llm = ChatOpenAI(model="gpt-4-turbo")  # GPT-4-turbo 모델 사용
                chain = load_qa_chain(llm, chain_type="stuff")  # 질문-응답 체인 생성
                
                # OpenAI API 호출에 대한 사용량 콜백 처리
                with get_openai_callback() as cb:
                    # 유사한 문서에 기반하여 질문에 답변 생성
                    response = chain.run(input_documents=docs, question=user_question)
                    print(cb)  # API 호출 로그 출력
                
                # 생성된 답변을 화면에 출력
                st.write(response)
            
            # 처리 완료 후 플래그를 False로 다시 설정하여 입력 활성화
            st.session_state['processing'] = False

# 메인 함수 실행
if __name__ == '__main__':
    main()
