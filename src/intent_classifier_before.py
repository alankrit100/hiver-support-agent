"""Intent classifier using LLM with few-shot prompting.

Classifies customer support messages into predefined intent categories
using the approved intent definitions from intent_definitions.json.
"""

import json
import os
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

IMPORTANT RULES:
1. Choose EXACTLY ONE intent from the list below
2. Provide a confidence score between 0.0 and 1.0
3. Explain your reasoning briefly
4. ONLY classify as "unclear" if the message is truly unclassifiable (empty, gibberish, or contains no useful information about the issue)
5. Even short messages can be classified if they contain keywords (iOS version numbers, battery, crash, specific app names, etc.)
6. Pay attention to boundary notes - they define when to choose one intent over another
7. When in doubt, choose the MOST LIKELY intent rather than "unclear"

EXAMPLES OF SHORT MESSAGES THAT ARE CLASSIFIABLE:
- "ios11.0.1" → ios_update_issues (mentions iOS version)
- "battery drain" → battery_drain_after_update (mentions battery)
- "phone keeps freezing" → device_crash_freeze (mentions freeze)
- "safari not working" → app_malfunction (mentions specific app)
- "can't login" → account_access (mentions login)

AVAILABLE INTENTS:
"""
        self.system_instruction += "\n\n".join(intent_descriptions)

        self.user_prompt_template = """Classify this customer support message:

"{message}"

Respond in this exact JSON format:
{{
    "intent": "intent_name",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    def classify(self, message: str) -> dict:
        """Classify a single customer message.

        Args:
            message: The customer support message to classify.

        Returns:
            Dict with keys: intent, confidence, reasoning
        """
        # Handle edge cases
        edge_case = self._handle_edge_cases(message)
        if edge_case:
            return edge_case

        # Keyword fallback for short messages
        keyword_result = self._keyword_fallback(message)
        if keyword_result:
            # Still call LLM to validate, but use keyword as hint
            user_prompt = self.user_prompt_template.format(message=message)
            user_prompt += f"\n\nHint: This message likely relates to {keyword_result['intent']}"
            
            response = self.llm.generate_structured(
                prompt=user_prompt,
                system_instruction=self.system_instruction,
            )
            
            validated = self._validate_response(response)
            
            # If LLM agrees with keyword, boost confidence
            if validated["intent"] == keyword_result["intent"]:
                validated["confidence"] = max(validated["confidence"], 0.7)
                validated["reasoning"] = f"Keyword match + LLM confirmation: {validated['reasoning']}"
            
            return validated

        # Regular LLM classification
        user_prompt = self.user_prompt_template.format(message=message)

        # Get classification from LLM
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

    def _handle_edge_cases(self, message: str) -> dict | None:
        """Handle edge cases before calling the LLM."""
        message = message.strip()

        # Empty or truly unclassifiable messages
        if len(message) < 3:
            return {
                "intent": "unclear",
                "confidence": 0.1,
                "reasoning": "Message too short to classify",
            }

        # Just @mentions with no content
        words = message.split()
        if all(w.startswith("@") for w in words):
            return {
                "intent": "unclear",
                "confidence": 0.1,
                "reasoning": "Message contains only @mentions with no content",
            }

        # Just links
        if all(w.startswith("http") for w in words if len(w) > 3):
            return {
                "intent": "unclear",
                "confidence": 0.1,
                "reasoning": "Message contains only links",
            }

        # Non-English detection (simple heuristic)
        non_english_chars = sum(1 for c in message if ord(c) > 127)
        if non_english_chars > len(message) * 0.3:
            return {
                "intent": "unclear",
                "confidence": 0.2,
                "reasoning": "Message appears to be non-English",
            }

        return None

    def _keyword_fallback(self, message: str) -> dict | None:
        """Use keyword matching for short messages as hint for LLM."""
        message_lower = message.lower()
        
        # Only use for short messages (< 50 chars)
        if len(message) > 50:
            return None
        
        # iOS update indicators
        ios_keywords = ["ios", "update", "upgrade", "11.", "10.", "software", "download"]
        if any(kw in message_lower for kw in ios_keywords):
            return {"intent": "ios_update_issues", "confidence": 0.7,
                    "reasoning": "Short message with iOS update keywords"}
        
        # Battery indicators
        battery_keywords = ["battery", "drain", "charge", "power", "percent", "charger"]
        if any(kw in message_lower for kw in battery_keywords):
            return {"intent": "battery_drain_after_update", "confidence": 0.7,
                    "reasoning": "Short message with battery keywords"}
        
        # Crash/freeze indicators
        crash_keywords = ["crash", "freeze", "shut down", "restart", "hang", "stuck"]
        if any(kw in message_lower for kw in crash_keywords):
            return {"intent": "device_crash_freeze", "confidence": 0.7,
                    "reasoning": "Short message with crash/freeze keywords"}
        
        # App indicators
        app_keywords = ["app", "bluetooth", "wifi", "audio", "sound", "safari", "music"]
        if any(kw in message_lower for kw in app_keywords):
            return {"intent": "app_malfunction", "confidence": 0.7,
                    "reasoning": "Short message with app keywords"}
        
        # Account indicators
        account_keywords = ["account", "login", "password", "icloud", "backup", "sign"]
        if any(kw in message_lower for kw in account_keywords):
            return {"intent": "account_access", "confidence": 0.7,
                    "reasoning": "Short message with account keywords"}
        
        return None  # No keyword match, let LLM decide

    def _validate_response(self, response: dict) -> dict:
        """Validate and normalize the LLM response."""
        # Ensure required fields exist
        intent = response.get("intent", "unclear")
        confidence = response.get("confidence", 0.5)
        reasoning = response.get("reasoning", "No reasoning provided")

        # Validate intent name
        valid_intents = [i["name"] for i in self.intent_defs["intents"]]
        
        if intent not in valid_intents:
            # Try to find closest match
            closest = self._find_closest_intent(intent)
            if closest:
                intent = closest
                confidence = max(confidence, 0.5)  # Boost confidence for closest match
                reasoning = f"Corrected from '{response.get('intent')}' to '{closest}'. " + reasoning
            else:
                intent = "unclear"
                confidence = min(confidence, 0.3)
                reasoning = f"Invalid intent '{response.get('intent')}'. " + reasoning

        # Clamp confidence to [0, 1]
        confidence = max(0.0, min(1.0, float(confidence)))

        return {
            "intent": intent,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
        }

    def _find_closest_intent(self, intent: str) -> str | None:
        """Find closest valid intent using string similarity."""
        valid_intents = [i["name"] for i in self.intent_defs["intents"]]
        
        # Simple similarity check
        intent_lower = intent.lower()
        for valid in valid_intents:
            if intent_lower in valid or valid in intent_lower:
                return valid
            # Check for common typos/abbreviations
            if intent_lower.replace("_", "") in valid.replace("_", ""):
                return valid
        
        return None


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
    ]

    for msg in test_messages:
        result = classifier.classify(msg)
        print(f"\nMessage: {msg}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
