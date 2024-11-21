import os
import numpy as np
import faiss
import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

# Baixa os pacotes necessários para a tokenização de sentenças
nltk.download('punkt')
nltk.download('punkt_tab')

# Configura o cliente OpenAI usando a chave de API do ambiente
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

def preprocess_text(text):
    """
    Pré-processa o texto dividindo-o em sentenças.

    Args:
        text (str): Texto a ser pré-processado.

    Returns:
        list: Lista de sentenças extraídas do texto.
    """
    sentences = sent_tokenize(text)
    return sentences

def split_text(text):
    """
    Divide o texto em chunks menores para facilitar o processamento.

    Args:
        text (str): Texto a ser dividido.

    Returns:
        list: Lista de chunks de texto.
    """
    text_splitter = CharacterTextSplitter(
        separator='\n',       # Separador entre os chunks
        chunk_size=8000,      # Tamanho máximo de cada chunk
        chunk_overlap=1600    # Sobreposição entre os chunks
    )
    texts = text_splitter.split_text(text)
    return texts

def create_embeddings(texts):
    """
    Cria embeddings para os textos e constrói um índice FAISS para busca eficiente.

    Args:
        texts (list): Lista de textos para os quais serão criados embeddings.

    Returns:
        tuple: Objeto de embeddings e o índice FAISS criado.
    """
    embeddings = OpenAIEmbeddings()
    doc_embeddings = embeddings.embed_documents(texts)    # Cria embeddings para os documentos
    dimension = len(doc_embeddings[0])                    # Obtém a dimensão dos embeddings
    index = faiss.IndexFlatL2(dimension)                  # Cria um índice FAISS usando distância L2
    index.add(np.array(doc_embeddings))                   # Adiciona os embeddings ao índice
    return embeddings, index

def search_docs(query, embeddings, index, texts, k=3):
    """
    Busca os documentos mais relevantes para a consulta fornecida.

    Args:
        query (str): Consulta de busca.
        embeddings (OpenAIEmbeddings): Objeto de embeddings para criar o embedding da consulta.
        index (faiss.IndexFlatL2): Índice FAISS para realizar a busca.
        texts (list): Lista de textos originais.
        k (int, opcional): Número de resultados a retornar. Padrão é 3.

    Returns:
        list: Lista dos textos mais relevantes encontrados.
    """
    query_embedding = embeddings.embed_query(query)       # Cria o embedding da consulta
    distances, indices = index.search(np.array([query_embedding]), k)  # Realiza a busca no índice
    results = [texts[i] for i in indices[0]]              # Recupera os textos correspondentes aos índices
    return results

def generate_answer(messages, embeddings, index, texts):
    """
    Gera a resposta do chatbot usando o modelo de linguagem, considerando o contexto.

    Args:
        messages (list): Histórico de mensagens da conversa.
        embeddings (OpenAIEmbeddings): Objeto de embeddings para busca.
        index (faiss.IndexFlatL2): Índice FAISS para busca.
        texts (list): Lista de textos originais para contexto.

    Returns:
        str: Resposta gerada pelo modelo.
    """
    question = messages[-1]["content"]                    # Obtém a última pergunta do usuário
    context = search_docs(question, embeddings, index, texts)  # Busca o contexto relevante
    context_str = "\n\n".join(context)                    # Combina os textos do contexto em uma string

    # Configura a mensagem inicial para o modelo
    api_messages = [
        {"role": "system", "content": "Você é um assistente especializado no Vestibular da Unicamp 2025. Responda às perguntas dos usuários usando apenas as informações do edital oficial fornecido. Seja preciso e direto. Se não encontrar a resposta no contexto, indique que não possui essa informação."},
    ]

    # Seleciona as últimas três mensagens do histórico, se existirem
    previous_messages = messages[-3:] if len(messages) >= 3 else messages
    for msg in previous_messages[:-1]:
        api_messages.append(msg)

    # Adiciona a última mensagem do usuário com o contexto encontrado
    last_user_message = messages[-1]
    last_user_message_with_context = {
        "role": last_user_message["role"],
        "content": f"Contexto:\n{context_str}\n\nPergunta:\n{last_user_message['content']}"
    }
    api_messages.append(last_user_message_with_context)

    # Realiza a chamada à API de completions do OpenAI
    chat_completion = client.chat.completions.create(
        model="gpt-3.5-turbo",            # Modelo a ser utilizado
        messages=api_messages,            # Histórico de mensagens
        max_tokens=400,                   # Máximo de tokens na resposta
        temperature=0.3                   # Controla a aleatoriedade da resposta
    )
    answer = chat_completion.choices[0].message.content.strip()  # Extrai e limpa a resposta gerada
    return answer

def main():
    """
    Função principal que executa o aplicativo Streamlit.
    """

    # Configurações iniciais da página do Streamlit
    st.set_page_config(page_title="Chatbot Vestibular Unicamp 2025", page_icon="🎓", layout="wide")
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Adiciona estilo customizado para melhorar a interface
    st.markdown("""
        <style>
            .stChatMessage {
                font-size: 1.1em;
            }
            .user {
                background-color: #DCF8C6;
            }
            .assistant {
                background-color: #F1F0F0;
            }
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

    # Cria a barra lateral com informações adicionais
    with st.sidebar:
        image_path = os.path.join(current_dir, '..', 'data', 'UNICAMP.png')
        st.image(image_path, use_container_width=True)
        st.markdown("## Sobre")
        st.write("Este chatbot foi desenvolvido para ajudar candidatos a esclarecer dúvidas sobre o Vestibular da Unicamp 2025.")
        st.markdown("---")
        st.markdown("### Desenvolvido por")
        st.write("Augusto Zolet")
        st.write("[LinkedIn](www.linkedin.com/in/augusto-zolet) | [GitHub](https://github.com/Augusto-Zolet)")

    # Título e descrição do aplicativo
    st.title("🎓 Chatbot Vestibular Unicamp 2025")
    st.write("Bem-vindo ao chatbot que responde suas dúvidas sobre o Vestibular da Unicamp 2025! Digite sua pergunta abaixo e aguarde a resposta.")

    # Processa o edital e cria os embeddings na primeira execução
    if 'texts' not in st.session_state:
        with st.spinner('Processando o edital, por favor aguarde...'):
            text_path = os.path.join(current_dir, '..', 'data', 'Normas_Vestibular_2025.txt')
            with open(text_path, 'r', encoding='utf-8') as file:
                edital_text = file.read()
            sentences = preprocess_text(edital_text)               # Pré-processa o texto do edital
            texts = split_text(edital_text)                        # Divide o texto em chunks
            embeddings, index = create_embeddings(texts)           # Cria embeddings e índice FAISS
            st.session_state.texts = texts                         # Armazena os textos no estado da sessão
            st.session_state.embeddings = embeddings               # Armazena os embeddings
            st.session_state.index = index                         # Armazena o índice FAISS
    else:
        # Recupera dados do estado da sessão para evitar reprocessamento
        texts = st.session_state.texts
        embeddings = st.session_state.embeddings
        index = st.session_state.index

    # Inicializa o histórico de mensagens se ainda não existir
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Exibe as mensagens anteriores na interface de chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Campo de entrada para a pergunta do usuário
    question = st.chat_input("Digite sua pergunta sobre o vestibular:")

    if question:
        # Adiciona a mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(f"**Você:** {question}")

        # Gera a resposta do chatbot com base na pergunta
        with st.spinner('Gerando resposta...'):
            try:
                answer = generate_answer(st.session_state.messages, embeddings, index, texts)
                # Adiciona a resposta ao histórico de mensagens
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(f"**Chatbot:** {answer}")
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")

# Executa a função principal quando o script é executado
if __name__ == '__main__':
    main()
