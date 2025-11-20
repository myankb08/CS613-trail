
"""
Advanced Quality Assessment Module
Multilingual-specific quality metrics
"""

from typing import Dict
import re

class MultilingualQualityAssessment:
    """
    Multilingual quality metrics beyond standard BLEU/COMET
    Designed for Indic languages
    """

    INDIC_SCRIPTS = {
        'devanagari': '\u0900-\u097F',
        'telugu': '\u0C00-\u0C7F',
        'tamil': '\u0B80-\u0BFF',
        'kannada': '\u0C80-\u0CFF',
        'bengali': '\u0980-\u09FF',
        'punjabi': '\u0A00-\u0A7F',
        'gujarati': '\u0A80-\u0AFF',
        'marathi': '\u0900-\u097F',
    }

    @staticmethod
    def detect_script(text: str) -> str:
        """Detect which Indic script is used"""
        for script_name, code_range in MultilingualQualityAssessment.INDIC_SCRIPTS.items():
            pattern = f'[{code_range}]'
            if re.search(pattern, text):
                return script_name

        if any(ord(c) < 128 and c.isalpha() for c in text):
            return 'latin'

        return 'unknown'

    @staticmethod
    def check_morphological_complexity(text: str) -> float:
        """
        Assess morphological complexity (important for Indic languages)
        Higher score = more complex
        """
        words = text.split()
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)

        # Indic scripts: higher character count expected
        if avg_word_len > 6:  # Complex
            return 0.8
        elif avg_word_len > 4:  # Medium
            return 0.5
        else:
            return 0.3

    @staticmethod
    def check_diacritic_preservation(source: str, target: str) -> float:
        """
        Check if diacritics/matras are preserved
        Critical for Indic language quality
        """
        # Simplified: count combining characters
        source_diacritics = sum(1 for c in source if '\u0300' <= c <= '\u036F')
        target_diacritics = sum(1 for c in target if '\u0300' <= c <= '\u036F')

        if source_diacritics == 0:
            return 1.0

        preservation = target_diacritics / source_diacritics
        return min(1.0, preservation)

    @staticmethod
    def check_number_preservation(source: str, target: str) -> float:
        """Check if numbers are preserved in translation"""
        source_numbers = re.findall(r'\d+', source)
        target_numbers = re.findall(r'\d+', target)

        if not source_numbers:
            return 1.0

        match_count = len(set(source_numbers) & set(target_numbers))
        return match_count / len(source_numbers)

    @staticmethod
    def compute_comprehensive_quality(source: str, target: str, 
                                     src_lang: str = 'hindi', 
                                     tgt_lang: str = 'telugu') -> Dict:
        """Compute comprehensive quality metrics"""
        metrics = {}

        metrics['script'] = MultilingualQualityAssessment.detect_script(target)
        metrics['morphological_complexity'] = MultilingualQualityAssessment.check_morphological_complexity(target)
        metrics['diacritic_preservation'] = MultilingualQualityAssessment.check_diacritic_preservation(source, target)
        metrics['number_preservation'] = MultilingualQualityAssessment.check_number_preservation(source, target)

        # Composite score
        weights = {
            'diacritic_preservation': 0.3,
            'number_preservation': 0.2,
            'morphological_complexity': 0.5
        }

        composite = sum(metrics[k] * weights.get(k, 0) 
                       for k in metrics if k in weights)
        metrics['composite_score'] = composite

        return metrics

class LanguagePairSpecificMetrics:
    """Language-pair specific quality checks"""

    PAIR_PATTERNS = {
        ('hindi', 'telugu'): {
            'gender_markers': True,
            'case_sensitivity': True,
            'honorifics': True
        },
        ('telugu', 'hindi'): {
            'gender_markers': True,
            'case_sensitivity': True,
        },
    }

    @staticmethod
    def get_language_pair_weights(src_lang: str, tgt_lang: str) -> Dict:
        """Get quality metric weights for specific language pair"""
        pair = (src_lang.lower(), tgt_lang.lower())

        if pair in LanguagePairSpecificMetrics.PAIR_PATTERNS:
            return LanguagePairSpecificMetrics.PAIR_PATTERNS[pair]

        return {}

