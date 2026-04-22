from flask import Flask, request, jsonify
import os
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route("/api/rag", methods=["GET", "POST"])
def rag():

    # Browser access
    if request.method == "GET":
        return jsonify({
            "message": "RAG API is live. Use POST with JSON body."
        })

    # POST request
    data = request.get_json()
    query = data.get("query", "")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": query}
        ]
    )

    answer = response.choices[0].message.content

    return jsonify({
        "query": query,
        "answer": answer
    })