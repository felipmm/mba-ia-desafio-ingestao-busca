# Desafio MBA Engenharia de Software com IA - Full Cycle

Sistema RAG (Retrieval-Augmented Generation) para busca e chat sobre documentos PDF utilizando LangChain, PostgreSQL com pgvector e Google Gemini.

## 📋 Pré-requisitos

- Python 3.9+ (recomendado 3.11)
- Docker e Docker Compose
- Conta no Google AI Studio para obter API Key do Gemini

## 🚀 Como executar a aplicação

### 1. Clone o repositório e navegue até o diretório
```bash
git clone https://github.com/felipmm/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca
```

### 2. Configure as variáveis de ambiente
Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# Google AI Configuration
GOOGLE_API_KEY=sua_api_key_do_google_ai_studio
GOOGLE_EMBEDDING_MODEL=models/embedding-001
GOOGLE_CHAT_MODEL=gemini-2.0-flash-exp

# Database Configuration
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documents
```

**⚠️ Importante:** Obtenha sua API Key gratuita em [Google AI Studio](https://aistudio.google.com/app/apikey)

### 3. Inicie o banco de dados PostgreSQL com pgvector
```bash
docker-compose up -d
```

Aguarde alguns segundos para que o banco seja inicializado e a extensão pgvector seja criada.

### 4. Configure o ambiente Python
```bash
# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# No macOS/Linux:
source venv/bin/activate
# No Windows:
# venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### 5. Execute a ingestão dos documentos PDF
```bash
cd src
python ingest.py
```

Este comando irá:
- Carregar os PDFs da pasta `src/pdf/` (Iron Maiden e Receitas Tradicionais)
- Dividir os documentos em chunks
- Gerar embeddings usando Google Gemini
- Armazenar no PostgreSQL com pgvector

### 6. Inicie o chat interativo
```bash
python chat.py
```

## 💬 Como usar o chat

Após executar o chat, você pode fazer perguntas sobre o conteúdo dos PDFs:

**Exemplos de perguntas:**
- "Quem são os membros do Iron Maiden?"
- "Qual é a história da banda Iron Maiden?"
- "Como fazer uma receita tradicional?"
- "Quais ingredientes são usados nas receitas?"

Para encerrar o chat, digite `quit`.

## 🏗️ Arquitetura da aplicação

- **ingest.py**: Processa e armazena documentos PDF no banco vetorial
- **chat.py**: Interface de chat com histórico de conversação
- **search.py**: Templates de prompt para respostas baseadas em contexto
- **docker-compose.yml**: Configuração do PostgreSQL com pgvector

## 🔧 Solução de problemas

### Erro de dependências
Se encontrar problemas com numpy no Python 3.9, use:
```bash
pip install numpy==1.26.4
```

### Banco de dados não conecta
Verifique se o Docker está rodando:
```bash
docker-compose ps
```

### API Key inválida
Confirme se a variável `GOOGLE_API_KEY` está correta no arquivo `.env`

## 🛑 Para parar a aplicação

```bash
# Parar o banco de dados
docker-compose down

# Desativar o ambiente virtual
deactivate