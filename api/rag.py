from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

knowledge_base = [
    "RAG stands for Retrieval-Augmented Generation.",
    "Vercel supports Python deployments.",
    "FAISS is used for vector similarity search.",
    "LangChain helps build LLM apps."
]

def retrieve_context(query):
    q = query.lower()
    matches = [x for x in knowledge_base if any(word in x.lower() for word in q.split())]
    return "\n".join(matches[:3]) if matches else "No relevant context found."

@app.route("/api/rag", methods=["POST"])
def rag():
    data = request.get_json()
    query = data.get("query", "")

    context = retrieve_context(query)

    prompt = f"""
Use this context to answer the question.

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return jsonify({
        "query": query,
        "answer": answer,
        "context": context
    })