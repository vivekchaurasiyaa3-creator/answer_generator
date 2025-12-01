from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")

def critic(state):
    critique = llm.predict(
        f"Review this answer for correctness and clarity:\n\n{state['answer']}"
    )
    return {"evaluation": critique}
