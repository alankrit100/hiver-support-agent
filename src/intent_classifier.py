"""Intent classifier using LLM with few-shot prompting.

Classifies customer support messages into predefined intent categories
using the approved intent definitions from intent_definitions.json.

This is the LLM-only version - no regex shortcuts, no keyword fallback.
The LLM (Sarvam 105B) handles all classification decisions.
"""

import json
from pathlib import Path

from llm_client import LLMClient


class IntentClassifier:
    """LLM-based intent classifier with confidence scoring."""

    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client or LLMClient()
        self.intent_defs = self._load_intent_definitions()
        self._build_few_shot_prompt()

    def _load_intent_definitions(self) -> dict:
        """Load intent definitions from JSON file."""
        intent_path = Path(__file__).parent / "intent_definitions.json"
        with open(intent_path) as f:
            return json.load(f)

    def _build_few_shot_prompt(self):
        """Build the few-shot classification prompt."""
        intents = self.intent_defs["intents"]

        # Build intent descriptions with examples
        intent_descriptions = []
        for intent in intents:
            examples = "\n".join(f'  - "{ex}"' for ex in intent["examples"])
            intent_descriptions.append(
                f"**{intent['name']}**: {intent['description']}\n"
                f"  Examples:\n{examples}\n"
                f"  Boundary: {intent['boundary_notes']}"
            )

        self.system_instruction = """You are an Apple Support message classifier. Your job is to classify customer messages into one of the predefined intent categories.

CRITICAL RULE - READ THIS FIRST:
You MUST classify messages into one of the 5 intents below. The "unclear" intent is ONLY for messages that are completely unclassifiable (empty, gibberish, or contain NO useful information about the issue). Even short messages with keywords like "ios", "battery", "crash", "app", "login" should be classified into one of the 5 intents.

RULES:
1. Choose EXACTLY ONE intent from the list below
2. Provide a confidence score between 0.0 and 1.0
3. NEVER classify as "unclear" if the message contains ANY of these keywords: ios, update, battery, drain, charge, crash, freeze, app, login, password, icloud
4. When in doubt, choose the MOST LIKELY intent rather than "unclear"

INTENT SELECTION RULES:
- If message mentions iOS version numbers (11, 10, 11.0.3, etc.) OR update/upgrade → ios_update_issues
- If message mentions battery, draining, power, charging, dies, dead → battery_drain_after_update
- If message mentions crash, freeze, shutdown, unresponsive, stuck, slow, sluggish → device_crash_freeze
- If message mentions specific app (Safari, Music, Messages, etc.) OR feature not working → app_malfunction
- If message mentions login, account, password, iCloud, backup, Apple ID → account_access

WHEN MESSAGE MENTIONS MULTIPLE ISSUES:
- Battery + Crash/Freeze → device_crash_freeze (crash is more severe)
- Battery + Update → battery_drain_after_update (battery is the main complaint)
- App + Update → app_malfunction (app is the specific issue)
- App + Account → account_access (account is root cause)

DO NOT CLASSIFY AS UNCLEAR IF:
- Message mentions any iOS version (e.g., "ios11.0.1", "iOS 11", "11.0.3")
- Message mentions battery, drain, charge, power
- Message mentions crash, freeze, shutdown
- Message mentions any app name (Safari, Music, Messages, etc.)
- Message mentions login, password, account, iCloud
- Message is short but contains any of the above keywords

AVAILABLE INTENTS:
"""
        self.system_instruction += "\n\n".join(intent_descriptions)

        self.user_prompt_template = """Classify this customer support message:

"{message}"

Respond in this exact JSON format:
{{
    "intent": "intent_name",
    "confidence": 0.0-1.0
}}"""

    def classify(self, message: str) -> dict:
        """Classify a single customer message using LLM only.

        Args:
            message: The customer support message to classify.

        Returns:
            Dict with keys: intent, confidence
        """
        # Build the full prompt
        user_prompt = self.user_prompt_template.format(message=message)

        # Get classification from LLM (no regex shortcuts)
        response = self.llm.generate_structured(
            prompt=user_prompt,
            system_instruction=self.system_instruction,
        )

        # Validate and normalize response
        return self._validate_response(response)

    def classify_batch(self, messages: list) -> list:
        """Classify multiple messages.

        Args:
            messages: List of customer support messages.

        Returns:
            List of classification dicts.
        """
        return [self.classify(msg) for msg in messages]

    def _validate_response(self, response: dict) -> dict:
        """Validate and normalize the LLM response."""
        # Ensure required fields exist
        intent = response.get("intent", "unclear")
        confidence = response.get("confidence", 0.5)

        # Validate intent name
        valid_intents = [i["name"] for i in self.intent_defs["intents"]]
        
        if intent not in valid_intents:
            # Force to unclear if invalid
            intent = "unclear"
            confidence = min(confidence, 0.3)

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, float(confidence)))

        return {
            "intent": intent,
            "confidence": round(confidence, 2),
        }


if __name__ == "__main__":
    # Test the classifier
    classifier = IntentClassifier()

    test_messages = [
        "My phone keeps shutting off randomly",
        "Thank you so much!",
        "The iOS update failed to install",
        "My battery is draining so fast",
        "??",
        "@AppleSupport",
        "I can't log into my iCloud account",
        "ios11.0.1",
        "battery drain",
        "safari not working",
    ]

    for msg in test_messages:
        result = classifier.classify(msg)
        print(f"\nMessage: {msg}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']}")
