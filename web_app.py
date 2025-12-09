from flask import Flask, render_template, request
from graph.workflow import create_workflow

app = Flask(__name__)
workflow = create_workflow()

@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    answer = None
    evaluation = None
    error = None

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            try:
                result = workflow.invoke({"query": query})
                answer = result.get("answer")
                evaluation = result.get("evaluation")
            except Exception as e:
                error = str(e)

    return render_template(
        "index.html",
        query=query,
        answer=answer,
        evaluation=evaluation,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)
