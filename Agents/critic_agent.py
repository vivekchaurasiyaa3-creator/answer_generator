from .llm_config import get_llm

llm = get_llm(temperature=0.0)

def critic(state):
    # We assume generator stored the answer under "answer"
    answer_text = state.get("answer", "")

    prompt = f"""
You are a strict but helpful university examiner.

Review the following answer and provide feedback in clear bullets.

REVIEW RULES:
- Focus on correctness, completeness, clarity, and structure.
- Mention if Definition / Explanation / Diagram / Applications / Advantages / Disadvantages are missing or weak.
- Be concise but specific.
- Do NOT repeat the full original answer, only comment on it.

Answer to review:
{answer_text}
"""

    # llm is a _PredictWrapper → .predict(...) is available
    raw = llm.predict(prompt)

    # Handle both AIMessage and plain string
    if hasattr(raw, "content"):
        critique = raw.content
    else:
        critique = raw

    # Normalize newlines, remove triple blanks
    critique = critique.replace("\r\n", "\n")
    while "\n\n\n" in critique:
        critique = critique.replace("\n\n\n", "\n\n")

    return {"evaluation": critique}

