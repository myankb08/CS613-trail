
"""
Discourse Coherence Module
Tracks pronouns, references, and maintains discourse-level consistency
"""

from collections import deque
from typing import List, Tuple
import re

class DiscourseTracker:
    """
    Tracks discourse elements:
    - Pronouns and their referents
    - Tense consistency
    - Topic continuity
    """

    def __init__(self, window_size=5):
        self.history = deque(maxlen=window_size)
        self.entity_references = {}
        self.pronouns = {'he', 'she', 'it', 'they', 'them', 'him', 'her', 'his', 'her', 'their'}

    def add_sentence(self, sentence: str, translation: str):
        """Add sentence pair to discourse history"""
        self.history.append({
            'source': sentence,
            'target': translation,
            'entities': self._extract_entities(sentence),
            'pronouns': self._extract_pronouns(sentence)
        })

    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities (simple: capitalized words)"""
        words = text.split()
        entities = [w.strip('.,!?;:') for w in words 
                   if w and w[0].isupper() and len(w) > 1]
        return entities

    def _extract_pronouns(self, text: str) -> List[str]:
        """Extract pronouns"""
        words = set(w.lower() for w in text.split())
        return [p for p in self.pronouns if p in words]

    def get_context_window(self, k: int = 3) -> List:
        """Get last k sentence pairs for context"""
        return list(self.history)[-k:]

    def compute_coherence_score(self) -> float:
        """
        Simple coherence score based on entity repetition
        and pronoun usage patterns
        """
        if len(self.history) < 2:
            return 1.0

        # Check for entity continuation
        recent = list(self.history)
        entity_continuity = 0

        for i in range(1, len(recent)):
            prev_entities = set(recent[i-1]['entities'])
            curr_entities = set(recent[i]['entities'])

            if prev_entities & curr_entities:
                entity_continuity += 1

        continuity_score = entity_continuity / (len(recent) - 1) if len(recent) > 1 else 0
        return min(1.0, continuity_score + 0.5)  # Combine with baseline

class PronounResolver:
    """Resolves pronoun references across sentences"""

    @staticmethod
    def resolve_pronouns(sentence: str, context_history: List[str]) -> str:
        """
        Attempt to resolve pronouns using context.
        Returns augmented sentence with resolved references.
        """
        pronouns_mapping = {
            'he': 'male_entity',
            'she': 'female_entity',
            'it': 'neutral_entity',
            'they': 'plural_entities'
        }

        # This is complex; simplified approach:
        # Just mark pronouns that might need attention
        for pronoun in pronouns_mapping:
            if pronoun in sentence.lower():
                # In a real system, trace back through history
                pass

        return sentence

class TenseConsistencyChecker:
    """Ensures tense consistency across translations"""

    TENSE_KEYWORDS = {
        'past': ['yesterday', 'ago', 'was', 'were', 'had', 'did'],
        'present': ['now', 'today', 'is', 'are', 'do', 'does'],
        'future': ['tomorrow', 'will', 'shall', 'going to']
    }

    @staticmethod
    def detect_tense(text: str) -> str:
        """Detect primary tense in text"""
        text_lower = text.lower()
        tense_scores = {}

        for tense, keywords in TenseConsistencyChecker.TENSE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            tense_scores[tense] = score

        if max(tense_scores.values()) == 0:
            return 'unknown'
        return max(tense_scores, key=tense_scores.get)

    @staticmethod
    def check_consistency(sentences: List[str]) -> Tuple[bool, str]:
        """Check if all sentences have consistent tense"""
        if not sentences:
            return True, "No content"

        tenses = [TenseConsistencyChecker.detect_tense(s) for s in sentences]
        primary_tense = max(set(tenses), key=tenses.count)

        inconsistent = [t for t in tenses if t != primary_tense and t != 'unknown']

        if inconsistent:
            return False, f"Tense inconsistency detected: primary={primary_tense}, conflicts={inconsistent}"
        return True, "Consistent"

