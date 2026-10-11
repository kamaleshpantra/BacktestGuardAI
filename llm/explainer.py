"""
llm/explainer.py
----------------
Local Ollama-powered explanation engine for BacktestGuard AI.

Architectural Safeguards:
1. Grounded Prompts:
   - The LLM receives deterministically validated findings and metrics.
   - It is strictly forbidden to invent findings, alter severity, or claim profitability.
2. Graceful Fallback:
   - If Ollama is offline or model call fails/times out, the system returns
     deterministic markdown without crashing or hallucinating.
"""

from __future__ import annotations

import httpx

from reporting.report_generator import ValidationReport


class OllamaExplainer:
    """
    Connects to local Ollama instance to generate human-readable explanations.
    """

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: float = 15.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    def explain(self, report: ValidationReport) -> str:
        """
        Produce clear, beginner-friendly explanation of validation findings.
        Falls back to deterministic summary if Ollama is unreachable.
        """
        if not self.is_available():
            return (
                "### ℹ️ Local AI Explanation Offline\n"
                "*Ollama daemon is currently not reachable at localhost:11434.* "
                "Review the deterministic findings directly above."
            )

        prompt = self._build_prompt(report)
        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "").strip()
                else:
                    return f"Ollama error {response.status_code}: {response.text}"
        except httpx.TimeoutException:
            return "Local AI model timed out. Deterministic report remains fully valid."
        except Exception as e:
            return f"Failed to generate AI explanation: {str(e)}"

    def _build_prompt(self, report: ValidationReport) -> str:
        return f"""You are BacktestGuard AI's technical mentor explaining backtesting quality.
Explain these specific automated findings to a beginner developer.

RULES:
1. Explain WHAT was detected and WHY it matters in trading simulations.
2. Do not invent new findings or contradict the report.
3. Keep it clear, concise, and structured.
4. Never state or imply that any strategy is profitable or guaranteed to work.

REPORT FINDINGS JSON:
{report.to_json()}

Produce your explanation now in clean markdown:"""
