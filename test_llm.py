from app.llm.client import LLMClient


print("\n")
print("=" * 80)
print("LLM TEST")
print("=" * 80)


llm = LLMClient()


response = llm.generate(

    instructions=(
        "You are a research assistant. "
        "Answer clearly and concisely."
    ),

    input_text=(
        "Explain in one paragraph what "
        "Virtual Try-On technology is."
    )
)


print("\n")
print("LLM RESPONSE")
print("-" * 80)
print(response)
