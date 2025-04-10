
import streamlit as st
import os
import time # Para simular processamento
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import traceback # Para mostrar erros no app

# --- Configurações e Carregamento de Recursos (com cache) ---

# Carregar Chave API (Prioriza st.secrets)
OPENAI_API_KEY = None
try:
    # Tenta carregar do secrets.toml (para deploy no Streamlit Cloud)
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    print("OpenAI Key carregada do st.secrets") # Log interno
except (FileNotFoundError, KeyError):
    print("Secret 'OPENAI_API_KEY' não encontrado no st.secrets. Tentando variável de ambiente...")
    # Tenta carregar de variáveis de ambiente
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    if OPENAI_API_KEY:
        print("OpenAI Key carregada da variável de ambiente.")
    else:
        st.error("Chave API da OpenAI ('OPENAI_API_KEY') não configurada! Configure em st.secrets ou variável de ambiente.")
        st.stop() # Para a execução se não tiver a chave

# Define a variável de ambiente para Langchain usar
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Caminho para o índice FAISS salvo
FAISS_INDEX_PATH = "faiss_index_anpd_acts"

@st.cache_resource(show_spinner="Carregando base de conhecimento (índice FAISS)...")
def load_faiss_index(index_path):
    if not os.path.exists(index_path):
         # CORREÇÃO: Escapar chaves ao redor de index_path
         st.error(f"Pasta do índice FAISS não encontrada em '{index_path}'. Verifique se a pasta existe no repositório junto com app.py.")
         st.stop()
    try:
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    except Exception as e:
        # CORREÇÃO: Escapar chaves ao redor de e
        st.error(f"Erro ao carregar índice FAISS: {e}")
        st.error(traceback.format_exc())
        st.stop()

@st.cache_resource(show_spinner="Preparando o assistente (chain QA)...")
def create_qa_chain(_vector_store):
    try:
        # Usa variáveis definidas no início do script app.py (não precisa escapar)
        llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.3)
        retriever = _vector_store.as_retriever(search_kwargs={'k': 4})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
         # CORREÇÃO: Escapar chaves ao redor de e
         st.error(f"Erro ao criar a chain QA: {e}")
         st.error(traceback.format_exc())
         st.stop()

# --- Carregar Recursos ---
vector_store_app = load_faiss_index(FAISS_INDEX_PATH)
qa_chain_app = create_qa_chain(vector_store_app)
st.success("Assistente pronto!") # Mensagem de sucesso após carregar tudo

# --- Interface do Usuário Streamlit ---
st.title("🖊📑 Chatbot Consulta ACTs ANPD")
st.markdown("Consulte informações sobre Acordos de Cooperação Técnica da ANPD")
st.divider()

with st.form("input_form"):
    user_question = st.text_area("Digite sua pergunta sobre os ACTs:", key="user_input", height=100)
    submitted = st.form_submit_button("Buscar Resposta")

    if submitted:
        if user_question:
            with st.spinner("Buscando informações nos documentos..."):
                try:
                    start_time = time.time()
                    resposta = qa_chain_app.invoke({"query": user_question})
                    end_time = time.time()

                    st.markdown("### Resposta:")
                    st.info(resposta.get('result', "Não foi possível obter uma resposta."))
                    st.caption(f"Tempo de resposta: {end_time - start_time:.2f} segundos") # Escapar aqui também

                    with st.expander("Ver Documentos Fonte Utilizados"):
                        if resposta.get('source_documents'):
                            fontes_usadas = set()
                            for doc in resposta['source_documents']:
                                source_file = os.path.basename(doc.metadata.get('source', 'N/A'))
                                if source_file not in fontes_usadas:
                                    # Usar st.markdown para formatar como código
                                    st.markdown(f"- `compress ocr act-senacon_ocultado (1)_compressed.pdf`")
                                    fontes_usadas.add(source_file)
                            if not fontes_usadas:
                                 st.write("Nenhuma fonte específica identificada nos metadados.")
                        else:
                            st.write("Nenhum documento fonte foi retornado pela chain.")

                except Exception as e:
                    # CORREÇÃO: Escapar chaves ao redor de e
                    st.error(f"Ocorreu um erro ao processar sua pergunta:")
                    st.exception(e) # Mostra o erro detalhado no app
        else:
            st.warning("Por favor, digite uma pergunta.")

st.divider()
st.caption("MBA IA & Big Data - Projeto GenAI RAG - Camila Falchetto")

