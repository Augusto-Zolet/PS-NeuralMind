# PS-NeuralMind: Chatbot com RAG para dúvidas do vestibular 2025 da Unicamp 

Este projeto consiste em um chatbot que utiliza o modelo GPT-3.5-turbo combinado com a técnica RAG (Retrieval-Augmented Generation) para responder dúvidas relacionadas ao vestibular 2025 da Unicamp.

## Pré-requisitos

Para execução do script, previamente deve-se ter instalado:

- **Python 3.12.5** ou superior
- **pip 24.3.1** ou superior

## Instalação

Siga os passos abaixo para configurar o ambiente:

1. **Clonar o repositório**

   Abra o terminal e execute o comando:

   ```bash
   git clone https://github.com/Augusto-Zolet/PS-NeuralMind.git
   ```

2. **Acessar o diretório do projeto**

   Navegue até o diretório do projeto:

   ```bash
   cd PS-NeuralMind
   ```

3. **Criar um ambiente virtual**

   Recomenda-se a criação de um ambiente virtual para isolar as dependências do projeto:

   ```bash
   python -m venv venv
   ```


4. **Ativar o ambiente virtual**

   Ative o ambiente virtual:

   - **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

5. **Instalar as dependências**

   Com o ambiente virtual ativado, instale as dependências necessárias:

   ```bash
   pip install -r requirements.txt
   ```

## Execução

Siga os passos abaixo para executar o chatbot:

1. **Definir a chave de API do OpenAI**

   O chatbot utiliza a API do OpenAI. É necessário definir a variável de ambiente `OPENAI_API_KEY` com a sua chave de API.

   - **Windows (Prompt de Comando)**:

     ```bash
     set OPENAI_API_KEY=sua-chave-api
     ```

   - **PowerShell**:

     ```bash
     $env:OPENAI_API_KEY="sua-chave-api"
     ```

   - **macOS/Linux**:

     ```bash
     export OPENAI_API_KEY='sua-chave-api'
     ```

   > *Substitua `'sua-chave-api'` pela sua chave de API real.*

2. **Executar o aplicativo**

   Inicie o aplicativo Streamlit:

   ```bash
   streamlit run src/chatbot.py
   ```

   O Streamlit abrirá uma nova janela no seu navegador padrão com a interface do chatbot.

---

**Desenvolvido por**: Augusto Zolet  
[LinkedIn](https://www.linkedin.com/in/augusto-zolet)

---