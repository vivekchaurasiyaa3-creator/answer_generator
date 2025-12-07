from graph.workflow import create_workflow

if __name__ == "__main__":
    workflow = create_workflow()

    query = "Explain Deep Learining in university format"
    result = workflow.invoke({"query": query})

    print("\n=== FINAL OUTPUT ===\n")
    print(result["answer"])
    print("\n=== CRITIC REVIEW ===\n")
    print(result["evaluation"])
