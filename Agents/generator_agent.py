from .llm_config import get_llm

llm = get_llm(temperature=0.2)

def generator(state):
    prompt = f"""
You are preparing a university-style answer.

IMPORTANT FORMATTING RULES:
- Each section MUST start on a NEW LINE.
- There MUST be ONE BLANK LINE between every section.
- Use the exact section titles given below.
- Do NOT merge sections together.

Use EXACTLY this output format:

Definition:
<write definition here>

Explanation:
<write explanation here>

Diagram:
<write diagram description here>

Applications:
<write applications here>

Advantages:
<write advantages here>

Disadvantages:
<write disadvantages here>

Topic: {state['query']}
Context Summary: {state['summary']}
"""

    # llm is a _PredictWrapper → .predict(...) works
    raw = llm.predict(prompt)

    # raw can be either a string or an AIMessage; handle both
    if hasattr(raw, "content"):
        answer = raw.content
    else:
        answer = raw

    # Normalize newlines, clean up extra blank lines
    answer = answer.replace("\r\n", "\n")
    while "\n\n\n" in answer:
        answer = answer.replace("\n\n\n", "\n\n")

    return {"answer": answer}
