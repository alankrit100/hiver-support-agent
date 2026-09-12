"""Reply drafter module using RAG (Retrieval-Augmented Generation).

Drafts replies to customer messages grounded in historically resolved threads.
Includes citations to which historical examples informed the draft.
"""

import json
from llm_client import LLMClient


class ReplyDrafter:
    """Drafts replies using retrieved historical resolutions."""

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()
        self._build_system_prompt()

    def _build_system_prompt(self):
        """Build the system prompt for reply drafting."""
        self.system_instruction = """Apple Support agent. Draft concise replies (2-4 sentences) grounded in retrieved examples.

CRITICAL RULE - SPECIFIC CITATIONS:
You MUST cite specific examples by number. Use this exact format:
- "Example 1 showed [solution], so let's try that..."
- "Following Example 2's approach for [issue]..."
- "Like in Example 3, we recommend [solution]..."
- If adapting: "Building on Example 1's solution, we suggest [modification]..."

DO NOT use generic phrases like "Based on similar cases" or "Following the approach that worked."
The customer must see WHICH example informed the reply.

If NO example fits well, say: "I don't have a similar resolved case for this, but [your advice]..."

INTENT GUIDELINES:
- ios_update_issues: Check for patches, force restart, recovery mode
- battery_drain_after_update: Check battery health, disable background refresh
- device_crash_freeze: Force restart, reset settings, DFU restore
- app_malfunction: Force-quit, reinstall, check app updates
- account_access: iforgot.apple.com, check Apple ID email
- unclear: Ask clarifying questions, offer DM support

Structure: Acknowledge → Solution (with specific example citation) → Offer further help."""

    def draft(
        self,
        message: str,
        intent: str,
        retrieved_threads: list,
        confidence: float = 0.5,
    ) -> dict:
        """Draft a reply to a customer message.

        Per the V4 brief: classifier confidence must NEVER suppress drafting.
        Low confidence should still get the best available grounded draft,
        with the human deciding via escalation -- not a dead-end fallback
        message. `confidence` is accepted (kept in the signature for callers
        and future use) but intentionally not used to gate drafting here.
        The only thing that produces a non-LLM fallback reply is a genuine
        cold start: no historical evidence to ground a draft in at all.

        Args:
            message: The customer's message.
            intent: Classified intent.
            retrieved_threads: List of similar resolved threads from retrieval.
            confidence: Classification confidence (NOT used to gate drafting;
                escalation.py is where confidence should affect the outcome).

        Returns:
            Dict with keys: reply, citations, grounding_source
        """
        # Cold start: no historical evidence at all to ground a draft in.
        if not retrieved_threads:
            return self._draft_cold_start(message, intent)

        # Build the prompt with retrieved context
        user_prompt = self._build_prompt(message, intent, retrieved_threads)

        # Get reply from LLM (with retry)
        response = self.llm.generate(
            prompt=user_prompt,
            system_instruction=self.system_instruction,
        )

        # Build citations
        citations = self._build_citations(retrieved_threads)

        # Handle None response from LLM (API failure, not a confidence issue)
        if response is None:
            return self._draft_cold_start(message, intent)

        return {
            "reply": response.strip(),
            "citations": citations,
            "grounding_source": "retrieval",
        }

    def _build_prompt(
        self, message: str, intent: str, retrieved_threads: list
    ) -> str:
        """Build the user prompt with retrieved context."""
        prompt = f"""Customer: "{message[:150]}"
Intent: {intent}

HISTORICAL EXAMPLES (cite by number in your reply):
"""
        for i, thread in enumerate(retrieved_threads[:3], 1):
            prompt += f"""
Example {i}:
  Customer: "{thread['customer_query'][:100]}"
  Agent: "{thread['brand_response'][:100]}"
"""

        prompt += """YOUR REPLY (must cite "Example 1", "Example 2", or "Example 3"):"""

        return prompt

    def _draft_cold_start(self, message: str, intent: str) -> dict:
        """Fallback reply for the genuine cold-start case: no historical
        resolution exists to ground a draft in (per the brief's defined
        fallback for this case: generic brand-tone reply + auto-escalate --
        the escalation module is responsible for the auto-escalate part
        based on grounding_source=="fallback_cold_start")."""
        return {
            "reply": "We'd like to help you with this. Could you provide a bit more detail about the issue you're experiencing? You can also DM us for more personalized support.",
            "citations": [],
            "grounding_source": "fallback_cold_start",
        }

    def _build_citations(self, retrieved_threads: list) -> list:
        """Build citation list from retrieved threads."""
        citations = []
        for i, thread in enumerate(retrieved_threads[:3], 1):
            citations.append(
                {
                    "example_number": i,
                    "similarity_score": thread.get("similarity_score", 0),
                    "customer_query_excerpt": thread["customer_query"][:100],
                    "resolution_type": thread.get("resolution_type", "unknown"),
                }
            )
        return citations


if __name__ == "__main__":
    # Test the reply drafter
    drafter = ReplyDrafter()

    # Mock retrieved threads
    mock_threads = [
        {
            "customer_query": "My phone keeps shutting off randomly",
            "brand_response": "We're sorry to hear that. Let's try a force restart: press and hold the Side button and Volume Down button together for 10 seconds.",
            "similarity_score": 0.85,
            "resolution_type": "gratitude_signal",
        },
        {
            "customer_query": "Phone shuts off when it wants no matter the battery life",
            "brand_response": "That sounds frustrating. Have you tried resetting all settings? Go to Settings > General > Transfer or Reset iPhone > Reset > Reset All Settings.",
            "similarity_score": 0.78,
            "resolution_type": "single_turn_with_troubleshooting",
        },
    ]

    result = drafter.draft(
        message="My iPhone keeps turning off by itself",
        intent="device_crash_freeze",
        retrieved_threads=mock_threads,
        confidence=0.85,
    )

    print("Reply:", result["reply"])
    print("\nCitations:", json.dumps(result["citations"], indent=2))
    print("\nGrounding:", result["grounding_source"])
