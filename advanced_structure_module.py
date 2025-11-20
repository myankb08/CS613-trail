
"""
Advanced Structure Preservation Module
Handles complex document structures with regex patterns and formatting preservation
"""

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class TextSegment:
    """Represents a text segment with its structural metadata"""
    content: str
    segment_type: str  # 'paragraph', 'heading', 'bullet', 'code', 'quote'
    formatting: Dict
    indent_level: int
    original_position: int

class AdvancedStructurePreserver:
    """Advanced structure preservation with support for complex formatting"""

    # Regex patterns for different structures
    PATTERNS = {
        'heading': r'^#+\s+(.+)$',
        'bullet': r'^\s*[-*•]\s+(.+)$',
        'numbered': r'^\s*\d+\.\s+(.+)$',
        'quote': r'^>\s+(.+)$',
        'code': r'^```(.+?)```$',
    }

    @staticmethod
    def segment_text(text: str) -> List[TextSegment]:
        """Break text into structural segments"""
        lines = text.split('\n')
        segments = []
        position = 0

        for line in lines:
            if not line.strip():
                segments.append(TextSegment(
                    content=line,
                    segment_type='whitespace',
                    formatting={},
                    indent_level=0,
                    original_position=position
                ))
            else:
                segment_type, formatting = AdvancedStructurePreserver._classify_segment(line)
                indent = len(line) - len(line.lstrip())

                segments.append(TextSegment(
                    content=line.strip(),
                    segment_type=segment_type,
                    formatting=formatting,
                    indent_level=indent // 4,  # Assume 4-space indents
                    original_position=position
                ))

            position += len(line) + 1

        return segments

    @staticmethod
    def _classify_segment(line: str) -> Tuple[str, Dict]:
        """Classify line type and extract formatting info"""
        for pattern_type, pattern in AdvancedStructurePreserver.PATTERNS.items():
            match = re.match(pattern, line, re.MULTILINE | re.DOTALL)
            if match:
                return pattern_type, {'match_groups': match.groups()}

        return 'paragraph', {}

    @staticmethod
    def reconstruct_with_segments(segments: List[TextSegment],
                                  translated_texts: Dict[int, str]) -> str:
        """Reconstruct text preserving original structure"""
        output = []

        for seg in segments:
            if seg.segment_type == 'whitespace':
                output.append(seg.content)
            elif seg.original_position in translated_texts:
                trans_text = translated_texts[seg.original_position]

                # Reapply formatting
                if seg.segment_type == 'heading':
                    heading_level = seg.content.count('#')
                    output.append('#' * heading_level + ' ' + trans_text)
                elif seg.segment_type == 'bullet':
                    output.append('• ' + trans_text)
                elif seg.segment_type == 'numbered':
                    output.append('1. ' + trans_text)  # Simplified
                elif seg.segment_type == 'quote':
                    output.append('> ' + trans_text)
                else:
                    indent = ' ' * (seg.indent_level * 4)
                    output.append(indent + trans_text)
            else:
                output.append(seg.content)

        return '\n'.join(output)

class FormattingPreserver:
    """Preserve inline formatting like bold, italic, links"""

    INLINE_PATTERNS = {
        'bold': r'\*\*(.+?)\*\*',
        'italic': r'\*(.+?)\*',
        'code': r'`(.+?)`',
        'link': r'\[(.+?)\]\((.+?)\)',
    }

    @staticmethod
    def extract_formatted_text(text: str) -> Dict:
        """Extract formatted regions"""
        formatted = {}

        for fmt_type, pattern in FormattingPreserver.INLINE_PATTERNS.items():
            matches = list(re.finditer(pattern, text))
            formatted[fmt_type] = [(m.group(1), m.start(), m.end()) for m in matches]

        return formatted

    @staticmethod
    def apply_formatting(translated: str, format_map: Dict) -> str:
        """Reapply formatting to translated text"""
        # This is complex due to position shifts in translation
        # Simplified approach: just preserve wrapped text

        # Extract plain content between formatting markers
        plain = translated
        for fmt_type, regions in format_map.items():
            for content, start, end in regions:
                if fmt_type == 'bold':
                    plain = plain.replace(content, f'**{content}**')
                elif fmt_type == 'italic':
                    plain = plain.replace(content, f'*{content}*')
                elif fmt_type == 'code':
                    plain = plain.replace(content, f'`{content}`')

        return plain

