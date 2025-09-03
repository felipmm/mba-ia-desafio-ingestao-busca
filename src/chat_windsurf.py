import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

from search import search_prompt
from langchain_postgres import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.runnables import chain, RunnableLambda

load_dotenv()
for k in ("DATABASE_URL", "PG_VECTOR_COLLECTION_NAME", "GOOGLE_CHAT_MODEL", "GOOGLE_API_KEY"):
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

    user_question = input("Digite sua pergunta: ").strip()

    results = store.similarity_search_with_score(user_question, k=10)

    context_parts = []
    for doc, score in results:
        context_parts.append(doc.page_content.strip().replace('\n', ' '))

    combined_context = "\n\n".join(context_parts)

    prompt_templates = search_prompt()
    formatted_prompt = prompt_templates.format(
        contexto=combined_context,
        pergunta=user_question
    )

    llm = ChatGoogleGenerativeAI(model=os.getenv("GOOGLE_CHAT_MODEL","gemini-2.5-flash"), temperature=0.5)

    response = llm.invoke(formatted_prompt)

    print(response)

if __name__ == "__main__":
    main()