"""
Gemini-based clinical decision support layer.

The CNN outputs calibrated-ish probabilities per finding; this module
turns those numbers into a structured, readable summary by prompting
Gemini with the model's own outputs (not the raw image -- the CNN has
already done the visual work, Gemini's job here is reasoning/communication
over structured findings, which is a more honest use of an LLM than
asking it to "read" an X-ray it never saw pixels of).

Explicitly NOT a diagnostic tool. Output is framed as decision *support*
for a clinician, not a diagnosis, and the prompt says so.

Requires: pip install google-genai
Requires: a GEMINI_API_KEY environment variable (never hardcode the key).
"""
import os
from dataclasses import dataclass

from model import NIH_CLASSES


@dataclass
class Finding:
    label: str
    probability: float


def top_findings(probs, threshold: float = 0.5, top_k: int = 5) -> list[Finding]:
    """
    probs: iterable of length len(NIH_CLASSES), sigmoid outputs for one image.
    Returns findings above `threshold`, sorted by probability descending,
    capped at top_k. If nothing clears the threshold, returns the single
    highest-probability finding so the report is never empty.
    """
    paired = sorted(zip(NIH_CLASSES, probs), key=lambda x: x[1], reverse=True)
    above = [Finding(label, float(p)) for label, p in paired if p >= threshold][:top_k]
    if not above:
        label, p = paired[0]
        above = [Finding(label, float(p))]
    return above


def build_prompt(findings: list[Finding], patient_context: str | None = None) -> str:
    lines = [
        "You are assisting a radiologist by summarizing the output of a CNN "
        "trained on NIH ChestX-ray14. You are NOT diagnosing the patient; "
        "the CNN's numbers are the input, your job is to explain them "
        "clearly and note their limitations.",
        "",
        "Model output (finding: probability):",
    ]
    for f in findings:
        lines.append(f"- {f.label}: {f.probability:.2f}")
    if patient_context:
        lines.append(f"\nPatient context provided by clinician: {patient_context}")
    lines += [
        "",
        "Write a short clinical decision-support note (4-6 sentences) that:",
        "1. States the model's top finding(s) and confidence in plain language.",
        "2. Notes any clinically relevant co-occurrence (e.g. Effusion + Cardiomegaly).",
        "3. Explicitly states this is a model output for review, not a diagnosis, "
        "and recommends radiologist confirmation.",
        "Do not invent findings that are not in the list above.",
    ]
    return "\n".join(lines)


def get_clinical_summary(probs, patient_context: str | None = None,
                          threshold: float = 0.5, model_name: str = "gemini-2.5-flash") -> str:
    """
    Calls the Gemini API. Raises a clear error if the API key is missing
    rather than silently failing, since this is a support-layer output
    that shouldn't be presented if it wasn't actually generated.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. In Colab: os.environ['GEMINI_API_KEY'] = "
            "userdata.get('GEMINI_API_KEY') (store it in Colab Secrets, don't paste it in a cell)."
        )

    from google import genai

    findings = top_findings(probs, threshold=threshold)
    prompt = build_prompt(findings, patient_context)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text


if __name__ == "__main__":
    # sanity test the parts that don't need a live API call / key
    fake_probs = [0.1] * len(NIH_CLASSES)
    fake_probs[NIH_CLASSES.index("Cardiomegaly")] = 0.87
    fake_probs[NIH_CLASSES.index("Effusion")] = 0.62

    findings = top_findings(fake_probs, threshold=0.5)
    assert len(findings) == 2
    assert findings[0].label == "Cardiomegaly"

    prompt = build_prompt(findings, patient_context="62F, presenting with dyspnea")
    assert "Cardiomegaly: 0.87" in prompt
    assert "not a diagnosis" in prompt.lower() or "NOT diagnosing" in prompt

    empty_case = top_findings([0.05] * len(NIH_CLASSES), threshold=0.5)
    assert len(empty_case) == 1  # falls back to top-1 instead of empty report

    print(f"[ok] top_findings={[ (f.label, round(f.probability,2)) for f in findings]}")
    print(f"[ok] prompt built, {len(prompt.splitlines())} lines")
    print("[skip] live Gemini API call (no key in this sandbox / not on network allowlist) -- test in Colab")
