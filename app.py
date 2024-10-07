from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

def main():
    # 환경 변수를 불러옵니다 (예: OpenAI API 키)
    load_dotenv()

    # Streamlit 페이지 설정 (페이지 제목 설정)
    st.set_page_config(page_title="연세대학교 행정 혁신")
    st.header("연세대학교 행정 혁신 챗봇(업로드 PDF 기반)")  # 페이지 헤더 설정

    # 세션 상태 초기화 (처리 중 여부를 확인하기 위한 플래그)
    if 'processing' not in st.session_state:
        st.session_state['processing'] = False

    # 파일 업로드 섹션 (최대 3개의 PDF 파일 업로드, 각 10MB로 제한)
    # 파일 업로드 섹션 (최대 3개의 PDF 파일 업로드, 각 10MB로 제한)
    uploaded_pdfs = st.file_uploader("최대 3개의 행정 매뉴얼을 PDF로 업로드해주세요. 업로드된 PDF 파일을 분석하여 답변을 생성합니다. (각 파일 최대 20MB)", 
                                      type="pdf", 
                                      accept_multiple_files=True)

    # 업로드된 파일이 있는지 확인하고, 최대 3개까지만 처리
    if uploaded_pdfs is not None and len(uploaded_pdfs) <= 3:
        all_text = ""
        valid_pdfs = True

        # 각 파일에 대해 파일 크기 확인
        for pdf in uploaded_pdfs:
            if pdf.size > 10 * 1024 * 1024 * 2:  # 10MB로 제한
                st.error(f"'{pdf.name}' 파일의 크기가 20MB를 초과합니다.")
                valid_pdfs = False
        
        if valid_pdfs:
            # 각 PDF에서 텍스트 추출
            for pdf in uploaded_pdfs:
                pdf_reader = PdfReader(pdf)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()

                # 텍스트 추출 확인
                if not text:
                    st.error(f"'{pdf.name}'에서 텍스트를 추출할 수 없습니다.")
                    return

                all_text += text

            # 텍스트를 청크 단위로 분할
            text_splitter = CharacterTextSplitter(
                separator="\n",  # 줄바꿈을 기준으로 분할
                chunk_size=1000,  # 각 청크의 최대 크기 설정
                chunk_overlap=200,  # 청크 간 중첩 설정
                length_function=len  # 텍스트 길이 계산 함수
            )
            chunks = text_splitter.split_text(all_text)

            # 청크가 유효한지 확인
            if len(chunks) == 0:
                #st.error("유효한 텍스트 청크가 생성되지 않았습니다. PDF 파일의 내용을 확인하세요.")
                return
            
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
