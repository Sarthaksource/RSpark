# llm.py
from config import state

# ─────────────────────────────────────────────
# LLM GENERATION
# ─────────────────────────────────────────────
def generate_answer(query: str, context_items: list[dict],
                    temperature: float = 0.2, max_new_tokens: int = 256) -> str:
    
    llm_model = state["llm_model"]

    context = "- " + "\n- ".join([item["text"] for item in context_items])

    base_prompt = (
        "You are a helpful assistant. Use the provided context to answer the user's query.\n"
        "Be detailed, explanatory, and directly address the question without filler. "
        "Keep your answer concise.\n\n"
        f"Context:\n{context}"
    )

    response = llm_model.create_chat_completion(
        messages=[
            {"role": "system", "content": base_prompt},
            {"role": "user", "content": query}
        ],
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    # Extract the text from the response dictionary
    output_text = response["choices"][0]["message"]["content"].strip()

    return output_text