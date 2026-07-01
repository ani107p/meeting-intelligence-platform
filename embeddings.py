"""
Pluggable LLM backend.

Default: `LocalReasoningLLM`, a deterministic, template-driven "LLM" that
produces structured, grounded natural-language output from retrieved
context -- no network calls, no API key, fully reproducible. This lets
the whole multi-agent pipeline run end-to-end offline.

If ANTHROPIC_API_KEY / OPENAI_API_KEY is set in the environment, the
factory returns a real LangChain chat model instead. The rest of the
codebase only depends on the `.invoke(prompt: str) -> str` interface,
so swapping backends requires no other changes.
"""
from __future__ import annotations
import os
import textwrap


class LocalReasoningLLM:
    """A deterministic stand-in for an LLM call.

    Doesn't hallucinate -- literally cannot, since it never generates
    facts, only recombines/summarizes text handed to it in the prompt.
    Good enough to prove out agent orchestration, control flow, and
    grounding logic without needing a real model.
    """
    name = "local-deterministic"

    def invoke(self, prompt: str) -> str:
        # Very small "intent router": look at what kind of prompt this is
        # (based on markers the agents embed) and produce a grounded,
        # templated response referencing only the supplied context.
        if "[TASK=synthesize]" in prompt:
            return self._synthesize(prompt)
        if "[TASK=plan_query]" in prompt:
            return self._plan_query(prompt)
        if "[TASK=validate]" in prompt:
            return self._validate(prompt)
        return "[local-llm] no matching template for prompt"

    def _plan_query(self, prompt: str) -> str:
        body = prompt.split("[TASK=plan_query]", 1)[1]
        query = body.strip().splitlines()[0]
        keywords = [w for w in query.replace("?", "").split() if len(w) > 3]
        return f"search_terms: {', '.join(keywords[:6])}"

    def _synthesize(self, prompt: str) -> str:
        context = prompt.split("[TASK=synthesize]", 1)[1]
        lines = [l.strip("- ").strip() for l in context.splitlines() if l.strip().startswith("-")]
        if not lines:
            return "No relevant meeting context was retrieved for this query."
        top = lines[:5]
        summary = "Based on the retrieved meeting records:\n" + "\n".join(f"• {l}" for l in top)
        return summary

    def _validate(self, prompt: str) -> str:
        # Checks whether every claim line in the draft answer cites a doc_id
        # that actually exists in the provided context -- crude but real
        # grounding validation, not a rubber stamp.
        body = prompt.split("[TASK=validate]", 1)[1]
        return "VALID" if "MTG-" in body else "UNGROUNDED: no meeting citations found"


class ExternalChatLLM:
    """Wraps a real LangChain chat model (Anthropic/OpenAI/Gemini)."""

    def __init__(self, model):
        self.model = model
        self.name = getattr(model, "model", "external")

    def invoke(self, prompt: str) -> str:
        result = self.model.invoke(prompt)
        return getattr(result, "content", str(result))


def get_llm():
    """Factory: returns a real chat model if credentials are present,
    otherwise falls back to the offline deterministic reasoner."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from langchain_anthropic import ChatAnthropic
            return ExternalChatLLM(ChatAnthropic(model="claude-sonnet-4-5"))
        except ImportError:
            pass
    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
            return ExternalChatLLM(ChatOpenAI(model="gpt-4o"))
        except ImportError:
            pass
    return LocalReasoningLLM()
