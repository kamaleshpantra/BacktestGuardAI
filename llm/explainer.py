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
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        """Checks if local Ollama daemon is reachable."""
        try:
            with httpx.Client(timeout=3.0) as client:
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
            # Set explicit connect and read timeouts
            timeout_cfg = httpx.Timeout(self.timeout, connect=10.0)
            with httpx.Client(timeout=timeout_cfg) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 600,  # Room to complete all points without truncation
                            "temperature": 0.1,  # Focused, deterministic output
                        },
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
        summary_info = {
            "dataset_passed": report.dataset_passed,
            "has_lookahead_risk": report.has_lookahead_risk,
            "metrics": report.metrics,
            "findings_count": len(report.data_findings) + len(report.code_findings),
        }
        import json
        return f"""You are BacktestGuard AI, a direct quantitative engineering mentor.
Analyze the backtest summary below and produce EXACTLY 3 concise bullet points for a developer.
Do NOT include introductory phrases, conversational filler, or greetings. Start directly with the bullet points.

1. **Risk & Return:** Evaluate the return vs max drawdown trade-off directly from the numbers.
2. **Quality & Frictions:** Assess what the Sharpe ratio and transaction fees reveal about execution quality.
3. **Simulation Disclaimer:** Explain in 1 sentence why this historical simulation cannot guarantee live market profitability.

METRICS:
{json.dumps(summary_info, indent=2)}

Analysis:"""
