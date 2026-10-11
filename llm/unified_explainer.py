"""
llm/unified_explainer.py
------------------------
Unified LLM client supporting both:
1. Local Ollama (default on local machine, zero cost)
2. Cloud OpenAI-compatible API (e.g. free Groq llama-3.1-8b or OpenRouter)
"""

from __future__ import annotations

import json

import httpx

from llm.explainer import OllamaExplainer
from reporting.report_generator import ValidationReport


class UnifiedExplainer:
    """
    Intelligently routes AI requests between local Ollama and Cloud APIs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        cloud_base_url: str = "https://api.groq.com/openai/v1",
        cloud_model: str = "llama-3.1-8b-instant",
        ollama_model: str = "qwen2.5:1.5b",
    ) -> None:
        self.api_key = api_key.strip() if api_key else None
        self.cloud_base_url = cloud_base_url.rstrip("/")
        self.cloud_model = cloud_model
        self.ollama = OllamaExplainer(model=ollama_model)

    def explain(self, report: ValidationReport) -> str:
        """
        Produce AI explanation. Uses Cloud API if an API key is provided;
        otherwise defaults to local Ollama.
        """
        if self.api_key:
            cloud_res = self._explain_cloud(report)
            if cloud_res:
                return cloud_res

        # Fallback to local Ollama
        return self.ollama.explain(report)

    def _explain_cloud(self, report: ValidationReport) -> str | None:
        """Call Groq / OpenAI compatible cloud endpoints."""
        summary_info = {
            "dataset_passed": report.dataset_passed,
            "has_lookahead_risk": report.has_lookahead_risk,
            "metrics": report.metrics,
            "findings_count": len(report.data_findings) + len(report.code_findings),
        }

        prompt = f"""You are BacktestGuard AI, a quant software engineering mentor.
Analyze the backtest summary below and produce EXACTLY 3 concise bullet points:
1. **Risk & Return:** Evaluate the return vs max drawdown trade-off.
2. **Quality & Frictions:** Assess what the Sharpe ratio and transaction fees reveal about execution quality.
3. **Simulation Disclaimer:** Remind the user why this historical simulation does not guarantee future live performance.

METRICS:
{json.dumps(summary_info, indent=2)}

Analysis:"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.cloud_model,
            "messages": [
                {"role": "system", "content": "You are a quantitative finance tutor. Be concise, direct, and factual."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 500,
            "temperature": 0.1,
        }

        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.post(f"{self.cloud_base_url}/chat/completions", headers=headers, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
        return None
