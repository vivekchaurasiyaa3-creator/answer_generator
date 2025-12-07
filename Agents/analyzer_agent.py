from .llm_config import get_llm

llm = get_llm(temperature=0.0)

def analyzer(state):
    text = "\n".join([d.page_content for d in state["retrieved"]])
    summary = llm.predict(f"Summarize this:\n{text}")
    return {"summary": summary}


