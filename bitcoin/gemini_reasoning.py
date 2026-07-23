"""
Gemini reasoning layer -- the "generation" half of the RAG pattern.

The retrieval step (retrieval.py) finds historical windows whose learned
embedding is close to the current one; this module turns those retrieved
analogs plus the model's numeric outputs into a natural-language market
note, in the same spirit as a document-grounded RAG answer -- except the
"documents" are historical market windows instead of text.

Explicitly NOT financial advice. The prompt says so, and so should
anything built on top of this output.

Requires: pip install google-genai
Requires: a GEMINI_API_KEY environment variable (never hardcode the key).
"""
import os

from retrieval import RetrievedNeighbor


def build_prompt(current_indicators: dict, raw_pred: float, blended_pred: float,
                  neighbors: list[RetrievedNeighbor]) -> str:
    lines = [
        "You are assisting with interpretation of a quantitative BTC/USD "
        "next-step return forecast. You are NOT giving financial advice; "
        "your job is to explain the model's output and the historical "
        "analogs it retrieved, clearly and with appropriate uncertainty.",
        "",
        "Current market indicators:",
    ]
    for k, v in current_indicators.items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")

    lines += [
        "",
        f"Base model (GRU/LSTM) raw predicted next-step return: {raw_pred:+.4%}",
        f"Retrieval-blended final prediction: {blended_pred:+.4%}",
        "",
        f"Top {len(neighbors)} historical windows with the most similar learned "
        "market pattern (by embedding distance), and what actually happened next:",
    ]
    for n in neighbors:
        lines.append(f"- {n.timestamp}: realized next-step return {n.outcome_return:+.4%} (distance {n.distance:.3f})")

    lines += [
        "",
        "Write a short note (4-6 sentences) that:",
        "1. States the model's prediction in plain language.",
        "2. Says whether the retrieved historical analogs mostly agree or "
        "disagree with each other and with the model's prediction (this is "
        "a signal of confidence, not a guarantee).",
        "3. Explicitly states this is a backtested statistical model output, "
        "not financial advice, and past patterns do not guarantee future results.",
        "Do not invent numbers not given above.",
    ]
    return "\n".join(lines)


def get_market_commentary(current_indicators: dict, raw_pred: float, blended_pred: float,
                           neighbors: list[RetrievedNeighbor], model_name: str = "gemini-2.5-flash") -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. In Colab: os.environ['GEMINI_API_KEY'] = "
            "userdata.get('GEMINI_API_KEY') (store it in Colab Secrets, don't paste it in a cell)."
        )

    from google import genai

    prompt = build_prompt(current_indicators, raw_pred, blended_pred, neighbors)
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


if __name__ == "__main__":
    fake_indicators = {"rsi": 62.3, "macd": 12.7, "ema_20": 42150.0}
    fake_neighbors = [
        RetrievedNeighbor(distance=0.12, outcome_return=0.004, timestamp="2024-03-01 14:00"),
        RetrievedNeighbor(distance=0.15, outcome_return=0.006, timestamp="2024-05-14 09:00"),
        RetrievedNeighbor(distance=0.21, outcome_return=-0.001, timestamp="2024-02-19 22:00"),
    ]

    prompt = build_prompt(fake_indicators, raw_pred=0.0032, blended_pred=0.0028, neighbors=fake_neighbors)
    assert "not financial advice" in prompt.lower()
    assert "0.32%" in prompt or "0.0032" in prompt or "+0.3200%" in prompt
    assert len(fake_neighbors) == prompt.count("distance ")

    print(f"[ok] prompt built, {len(prompt.splitlines())} lines")
    print("[skip] live Gemini API call (no key in this sandbox / not on network allowlist) -- test in Colab")
