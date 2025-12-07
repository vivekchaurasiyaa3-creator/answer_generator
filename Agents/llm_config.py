# llm_config.py
import os

# Try to use Groq provider first, fall back to OpenAI if available.
try:
    from langchain_groq import ChatGroq  # type: ignore
    _HAS_GROQ = True
except Exception:
    ChatGroq = None
    _HAS_GROQ = False

try:
    from langchain.chat_models import ChatOpenAI  # type: ignore
    _HAS_OPENAI = True
except Exception:
    ChatOpenAI = None
    _HAS_OPENAI = False


class _PredictWrapper:
    def __init__(self, model):
        self._m = model

    def predict(self, prompt: str):
        # Try common LangChain method names in order of likelihood.
        if hasattr(self._m, "predict"):
            return self._m.predict(prompt)
        if hasattr(self._m, "invoke"):
            return self._m.invoke(prompt)
        if hasattr(self._m, "generate"):
            # `generate` may accept an iterable of prompts.
            gen = self._m.generate([prompt])
            try:
                # langchain-style GenerateResult -> generations[0][0].text
                return gen.generations[0][0].text
            except Exception:
                return str(gen)
        if callable(self._m):
            return self._m(prompt)
        raise RuntimeError("Wrapped model has no callable predict/invoke/generate")


def get_llm(temperature: float = 0.0):
    if _HAS_GROQ:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment for Groq")
        model = ChatGroq(
            groq_api_key=api_key,
            model_name=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )
        return _PredictWrapper(model)

    if _HAS_OPENAI:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment for OpenAI fallback")
        model = ChatOpenAI(temperature=temperature, openai_api_key=api_key)
        return _PredictWrapper(model)

    raise ImportError(
        "No supported chat model available. Install `langchain_groq` or set up `ChatOpenAI` with `OPENAI_API_KEY`."
    )

