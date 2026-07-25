import os
from pathlib import Path
import pandas as pd 
from dotenv import load_dotenv
from retrieval import RetrievedNeighbor


_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH, override=False)


def build_prompt(
    current_indicators: dict,
    raw_pred: float,
    blended_pred: float,
    neighbors: list[RetrievedNeighbor],
) -> str:
    lines = [
        "You are assisting with interpretation of a quantitative BTC/USD "
        "next-step return forecast. You are NOT giving financial advice; "
        "your job is to explain the model's output and the historical "
        "analogs it retrieved, clearly and with appropriate uncertainty.",
        "",
        "Current market indicators:",
    ]

    for key, value in current_indicators.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")

    lines += [
        "",
        f"Base model raw predicted next-step return: {raw_pred:+.4%}",
        f"Retrieval-blended final prediction: {blended_pred:+.4%}",
        "",
        f"Top {len(neighbors)} historical windows with the most similar "
        "learned market pattern, and what happened next:",
    ]

    for neighbor in neighbors:
        lines.append(
            f"- {neighbor.timestamp}: realized next-step return "
            f"{neighbor.outcome_return:+.4%} "
            f"(distance {neighbor.distance:.3f})"
        )

    lines += [
        "",
        "Write a short note of 4-6 sentences that:",
        "1. States the model's prediction in plain language.",
        "2. Explains whether the historical analogs mostly agree or disagree "
        "with each other and with the model prediction.",
        "3. Explains that agreement is only a confidence signal, not a guarantee.",
        "4. Explicitly states this is a backtested statistical model output, "
        "not financial advice, and past patterns do not guarantee future results.",
        "Do not invent numbers that are not provided above.",
    ]

    return "\n".join(lines)

def get_market_commentary(
    current_indicators: dict,
    raw_pred: float,
    blended_pred: float,
    neighbors: list[RetrievedNeighbor],
    model_name: str | None = None,
) -> str:
    """Generate Gemini commentary for the model prediction."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"GEMINI_API_KEY was not found. Check this file:\n{_ENV_PATH}"
        )

    model_name = model_name or os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash-lite",
    )

    from google import genai

    prompt = build_prompt(
        current_indicators,
        raw_pred,
        blended_pred,
        neighbors,
    )

    client = genai.Client(api_key=api_key)

    try:
        response = client.models.generate_content(
            model=model_name.removeprefix("models/"),
            contents=prompt,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gemini request failed using model '{model_name}'. "
            "Set GEMINI_MODEL in .env to a model available to your API key."
        ) from exc

    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    return text

if __name__ == "__main__":
    fake_indicators = {
        "rsi": 62.3,
        "macd": 12.7,
        "ema_20": 42150.0,
    }

    fake_neighbors = [
        RetrievedNeighbor(
            distance=0.12,
            outcome_return=0.004,
            timestamp=pd.Timestamp("2024-03-01 15:00"),
        ),
        RetrievedNeighbor(
            distance=0.15,
            outcome_return=0.006,
            timestamp=pd.Timestamp("2024-05-14 09:00"),
        ),
        RetrievedNeighbor(
            distance=0.21,
            outcome_return=-0.001,
            timestamp=pd.Timestamp("2024-02-19 22:00"),
        ),
    ]

    prompt = build_prompt(
        fake_indicators,
        raw_pred=0.0032,
        blended_pred=0.0028,
        neighbors=fake_neighbors,
    )

    assert "not financial advice" in prompt.lower()
    assert len(fake_neighbors) == prompt.count("distance ")

    print(f"[ok] prompt built, {len(prompt.splitlines())} lines")
    print("[skip] live Gemini API call")