"""Escalation decision module.

Combines rule-based checks with LLM reasoning to determine if a message
should be escalated to a human agent. Always provides a human-readable reason.
"""

import re
from llm_client import LLMClient


class EscalationDecider:
    """Decides whether to escalate a message to a human agent."""

    # Rule-based escalation triggers
    ANGER_KEYWORDS = [
        "angry", "furious", "frustrated", "unacceptable", "terrible",
        "worst", "horrible", "disgusted", "livid", "outraged",
        "sue", "lawyer", "legal", "report", "bbb", "attorney",
    ]

    SAFETY_KEYWORDS = [
        "hurt", "injury", "dangerous", "fire", "burn", "shock",
        "electrical", "overheat", "explosion", "smoke",
    ]

    REFUND_KEYWORDS = [
        "refund", "money back", "return", "cancel", "reimburse",
        "charge", "billing", "overcharge", "double charge",
    ]

    # NOTE: "still", "keeps", "again", "continues" were removed from this list.
    # They fire on ordinary first-contact bug descriptions ("phone keeps
    # shutting off" describes a recurring *device* symptom, not repeated
    # *contact with support*) and were a documented false-positive source
    # (see PROGRESS.md). Kept only phrases that specifically signal the
    # customer has already contacted support before about this.
    REPEATED_FRUSTRATION = [
        "multiple times", "several times", "already told", "already said",
        "not the first time", "how many times", "for the third time",
        "keep telling you", "told you already",
    ]

    # Below this similarity score, retrieval evidence is considered too weak
    # to trust a grounded draft on its own.
    WEAK_EVIDENCE_SIMILARITY = 0.4

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()

    def should_escalate(
        self,
        message: str,
        intent: str,
        confidence: float,
        reasoning: str,
        reply: str = None,
        retrieved_threads: list = None,
        grounding_source: str = None,
    ) -> dict:
        """Determine if a message should be escalated.

        Args:
            message: The customer's message.
            intent: Classified intent.
            confidence: Classification confidence.
            reasoning: Classification reasoning.
            reply: Drafted reply (optional, for context).
            retrieved_threads: Retrieval results used to ground the reply
                (per the V4 brief: weak/missing evidence should escalate,
                independent of classifier confidence).
            grounding_source: reply_drafter.py's grounding_source
                ("retrieval" or "fallback_cold_start").

        Returns:
            Dict with keys: escalate (bool), reason (str), triggers (list)
        """
        triggers = []
        message_lower = message.lower()
        retrieved_threads = retrieved_threads or []

        # Rule 1: Low classification confidence
        if confidence < 0.4:
            triggers.append({
                "rule": "low_confidence",
                "detail": f"Classification confidence {confidence:.2f} is below threshold 0.4",
            })

        # Rule 1b: No historical resolution to ground the draft in at all
        # (cold start -- brief's defined fallback is generic reply + auto-escalate).
        if grounding_source == "fallback_cold_start":
            triggers.append({
                "rule": "no_historical_resolution",
                "detail": "No historical resolution was found to ground a draft in; "
                          "reply is a generic fallback and needs human handling.",
            })

        # Rule 1c: Retrieval evidence is present but weak (best match below threshold).
        elif retrieved_threads:
            best_similarity = max(t.get("similarity_score", 0) for t in retrieved_threads)
            if best_similarity < self.WEAK_EVIDENCE_SIMILARITY:
                triggers.append({
                    "rule": "weak_retrieval_evidence",
                    "detail": f"Best retrieval similarity {best_similarity:.2f} is below "
                              f"threshold {self.WEAK_EVIDENCE_SIMILARITY} -- reply may not "
                              f"be well grounded.",
                })

            # Rule 1d: Conflicting historical resolutions -- top matches are both
            # reasonably similar but resolved the issue differently, so the draft
            # may be picking one course of action arbitrarily.
            strong_matches = [t for t in retrieved_threads if t.get("similarity_score", 0) >= 0.5]
            resolution_types = {t.get("resolution_type") for t in strong_matches}
            if len(strong_matches) >= 2 and len(resolution_types) >= 2:
                triggers.append({
                    "rule": "conflicting_resolutions",
                    "detail": f"Top retrieved examples disagree on resolution type "
                              f"({', '.join(sorted(r for r in resolution_types if r))}); "
                              f"draft may be picking one arbitrarily.",
                })

        # Rule 2: Anger/frustration keywords
        anger_found = [kw for kw in self.ANGER_KEYWORDS if kw in message_lower]
        if anger_found:
            triggers.append({
                "rule": "anger_keywords",
                "detail": f"Found anger/frustration keywords: {', '.join(anger_found)}",
            })

        # Rule 3: Safety concerns
        safety_found = [kw for kw in self.SAFETY_KEYWORDS if kw in message_lower]
        if safety_found:
            triggers.append({
                "rule": "safety_concern",
                "detail": f"Found safety-related keywords: {', '.join(safety_found)}",
            })

        # Rule 4: Refund/legal requests
        refund_found = [kw for kw in self.REFUND_KEYWORDS if kw in message_lower]
        if refund_found:
            triggers.append({
                "rule": "refund_request",
                "detail": f"Found refund/legal keywords: {', '.join(refund_found)}",
            })

        # Rule 5: Repeated frustration indicators
        repeated_found = [kw for kw in self.REPEATED_FRUSTRATION if kw in message_lower]
        if repeated_found:
            triggers.append({
                "rule": "repeated_frustration",
                "detail": f"Indicates repeated issue: {', '.join(repeated_found)}",
            })

        # Rule 6: ALL CAPS (more than 50% uppercase, excluding short messages)
        if len(message) > 10:
            uppercase_chars = sum(1 for c in message if c.isupper())
            total_alpha = sum(1 for c in message if c.isalpha())
            if total_alpha > 10 and uppercase_chars / total_alpha > 0.7:
                triggers.append({
                    "rule": "all_caps",
                    "detail": "Message is predominantly in ALL CAPS",
                })

        # Rule 7: Multiple exclamation/question marks
        if message.count("!") > 2 or message.count("?") > 2:
            triggers.append({
                "rule": "excessive_punctuation",
                "detail": f"Multiple exclamation/question marks ({message.count('!')}!, {message.count('?')}?)",
            })

        # Rule 8: Profanity check (simple)
        profanity_pattern = r'\b(damn|crap|shit|fuck|ass|hell)\b'
        if re.search(profanity_pattern, message_lower):
            triggers.append({
                "rule": "profanity",
                "detail": "Message contains profanity",
            })

        # Rule 9: Mentions of competitor (switching threat)
        competitor_keywords = ["samsung", "android", "google", "switch", "competitor"]
        competitor_found = [kw for kw in competitor_keywords if kw in message_lower]
        if competitor_found:
            triggers.append({
                "rule": "competitor_mention",
                "detail": f"Mentions competitor/switching: {', '.join(competitor_found)}",
            })

        # Determine escalation
        escalate = len(triggers) > 0

        # Build reason
        if escalate:
            rule_names = [t["rule"] for t in triggers]
            reason = f"Escalation triggered by: {', '.join(rule_names)}. "
            reason += " ".join(t["detail"] for t in triggers)
        else:
            reason = "No escalation triggers detected. Message can be handled by automated system."

        return {
            "escalate": escalate,
            "reason": reason,
            "triggers": triggers,
        }


if __name__ == "__main__":
    # Test the escalation decider
    decider = EscalationDecider()

    test_cases = [
        {
            "message": "This is UNACCEPTABLE! I've told you 5 times already and nothing has changed!",
            "intent": "device_crash_freeze",
            "confidence": 0.85,
            "reasoning": "Clear crash/freeze complaint with frustration",
        },
        {
            "message": "My phone shuts off randomly",
            "intent": "device_crash_freeze",
            "confidence": 0.92,
            "reasoning": "Clear crash complaint",
        },
        {
            "message": "I'm going to switch to Samsung if you don't fix this NOW!!!",
            "intent": "device_crash_freeze",
            "confidence": 0.78,
            "reasoning": "Crash complaint with anger",
        },
        {
            "message": "Thank you so much, that fixed it!",
            "intent": "positive_feedback",
            "confidence": 0.95,
            "reasoning": "Gratitude message",
        },
    ]

    for case in test_cases:
        result = decider.should_escalate(**case)
        print(f"\nMessage: {case['message'][:80]}...")
        print(f"Escalate: {result['escalate']}")
        print(f"Reason: {result['reason'][:150]}")
        print(f"Triggers: {len(result['triggers'])}")
