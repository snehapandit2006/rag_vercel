from flask import Flask, request, jsonify, render_template
from utils.loader import load_documents
from utils.vectorstore import build_vectorstore, ask_rag

app = Flask(__name__, template_folder="../templates")

vector_db = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    global vector_db

    try:
        files = request.files.getlist("files")

        if not files:
            return jsonify({"message": "No files selected"}), 400

        docs = load_documents(files)

        vector_db = build_vectorstore(docs)

        return jsonify({
            "message": "Files uploaded successfully"
        })

    except Exception as e:
        print("UPLOAD ERROR:", str(e))   # terminal log
        return jsonify({
            "message": str(e)
        }), 500


@app.route("/ask", methods=["POST"])
def ask():

    global vector_db

    data = request.get_json()
    query = data["query"]

    answer = ask_rag(vector_db, query)

    return jsonify(answer)



if __name__ == "__main__":
    app.run(debug=True)

