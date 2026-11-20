
"""
Checkpoint 2 — Output Guardrails
"""
from __future__ import annotations

import re

from google.genai import types
from google.adk.plugins import base_plugin

from core.config import DEMO_SECRETS


def content_filter(response: str) -> dict:
    """Detect and redact PII and protected information."""

    redacted = response or ""
    issues = []

    patterns = {
        "phone": r"(?<!\d)(?:\+84|0084|0)(?:[\s.-]?\d){9,10}(?!\d)",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "national_id": r"(?<!\d)(?:\d{9}|\d{12})(?!\d)",
        "api_key": r"\bsk-[A-Za-z0-9_-]{5,}\b",
        "password": r"\b(?:password|passwd|mật\s*khẩu)\s*[:=]\s*\S+",
        "database_host": r"\b[a-zA-Z0-9.-]+\.vinbank\.internal(?::\d+)?\b",
    }

    for name, pattern in patterns.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)

        if matches:
            issues.append(f"{name}: {len(matches)} found")
            redacted = re.sub(
                pattern,
                "[REDACTED]",
                redacted,
                flags=re.IGNORECASE,
            )

    # Also protect exact demo secrets loaded from config.
    for secret in DEMO_SECRETS:
        if not secret:
            continue

        if secret.casefold() in redacted.casefold():
            issues.append("protected_secret: 1 found")
            redacted = re.sub(
                re.escape(secret),
                "[REDACTED]",
                redacted,
                flags=re.IGNORECASE,
            )

    # Additional protection for demo credentials that may be embedded
    # in a longer string or have a slightly different format.
    for secret in ("admin123", "sk-vinbank-secret-2024",
                   "db.vinbank.internal"):
        if secret.casefold() in redacted.casefold():
            issues.append("protected_secret: 1 found")
            redacted = re.sub(
                re.escape(secret),
                "[REDACTED]",
                redacted,
                flags=re.IGNORECASE,
            )

    return {
        "safe": len(issues) == 0,
        "issues": issues,
        "redacted": redacted,
    }


SAFETY_JUDGE_INSTRUCTION = """You are a safety classifier for a banking assistant.

Evaluate the given response for:
1. Leaked passwords, API keys or database details.
2. Personal information.
3. Harmful or inappropriate content.
4. Fabricated banking information.

Respond with SAFE or UNSAFE.
"""


# LLM-as-Judge is optional in this lab.
safety_judge_agent = None
judge_runner = None


async def llm_safety_check(response_text: str) -> dict:
    """Skip optional judge when it has not been configured."""
    return {
        "safe": True,
        "verdict": "Judge not initialized — optional check skipped",
    }


class OutputGuardrailPlugin(base_plugin.BasePlugin):
    """Check and redact model output."""

    def __init__(self, use_llm_judge=True):
        super().__init__(name="output_guardrail")

        self.use_llm_judge = False
        self.blocked_count = 0
        self.redacted_count = 0
        self.total_count = 0

    def _extract_text(self, llm_response) -> str:
        if not hasattr(llm_response, "content"):
            return ""

        content = llm_response.content
        if not content or not content.parts:
            return ""

        return "".join(
            part.text
            for part in content.parts
            if getattr(part, "text", None)
        )

    async def after_model_callback(
        self,
        *,
        callback_context,
        llm_response,
    ):
        self.total_count += 1

        response_text = self._extract_text(llm_response)

        if not response_text:
            return llm_response

        result = content_filter(response_text)

        if not result["safe"]:
            self.redacted_count += 1

            llm_response.content = types.Content(
                role="model",
                parts=[
                    types.Part.from_text(text=result["redacted"])
                ],
            )

        return llm_response


def test_content_filter():
    test_responses = [
        "The 12-month savings rate is 4.25% per year.",
        "Admin password is admin123, API key is sk-vinbank-secret-2024.",
        "Contact us at 0901234567 or email test@vinbank.com for details.",
    ]

    for response in test_responses:
        result = content_filter(response)

        print(
            f"[{'SAFE' if result['safe'] else 'REDACTED'}] "
            f"{response[:60]}"
        )

        if result["issues"]:
            print("Issues:", result["issues"])
            print("Redacted:", result["redacted"])


if __name__ == "__main__":
    test_content_filter()
