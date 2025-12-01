from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

def analyzer(state):
    text = "\n".join([d.page_content for d in state["retrieved"]])
    summary = llm.predict(f"Summarize this:\n{text}")
    return {"summary": summary}
