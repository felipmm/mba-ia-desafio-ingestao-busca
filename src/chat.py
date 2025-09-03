import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from search import search_prompt
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import chain, RunnableLambda

load_dotenv()
for k in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL","models/embedding-001"))

def main():
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
    )

    chain_prompt = search_prompt("pergunta?")

    if not chain_prompt:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    results = sorted(
       store.similarity_search_with_score(
           search_prompt().format(contexto="RECEITAS", pergunta="O que é brigadeiro?"),
           k=10),
       key=lambda x: (x[0].metadata.get("page", 0), x[1]),
       reverse=False
    )

    for i, (doc, score) in enumerate(results, start=1):
       print("=" * 50)
       print(f"Resultado {i} (score: {score:.2f}):")
       print(doc.page_content.strip().replace('\n', ' '))

if __name__ == "__main__":
    main()