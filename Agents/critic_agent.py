from .llm_config import get_llm

llm = get_llm(temperature=0.0)

def critic(state):
    critique = llm.predict(
        f"Review this answer for correctness and clarity:\n\n{state['answer']}"
    )
    return {"evaluation": critique}
