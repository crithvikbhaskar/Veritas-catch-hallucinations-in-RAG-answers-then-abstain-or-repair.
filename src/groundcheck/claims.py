"""Rule-based sentence and clause claim splitter."""

import re

from groundcheck.types import Claim

# Titles that should never end a sentence even when followed by a capitalized word
TITLES = {"mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr."}

# General abbreviations that only avoid sentence split if NOT followed by an uppercase letter
GENERAL_ABBREVS = {"e.g.", "i.e.", "vs.", "etc.", "fig.", "al.", "inc.", "corp.", "co.", "approx.", "dept.", "est."}



class ClaimSplitter:
    """Splits text into verifiable claims with accurate character offsets."""

    def __init__(self, split_clauses: bool = True, min_claim_length: int = 5):
        self.split_clauses = split_clauses
        self.min_claim_length = min_claim_length

    def split(self, text: str) -> list[Claim]:
        """Split text into a list of Claim objects preserving character spans."""
        if not text or not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        claims: list[Claim] = []
        claim_idx = 0

        for sent_idx, (sent_text, sent_start, sent_end) in enumerate(sentences):
            if not self.split_clauses or ";" not in sent_text:
                cleaned = sent_text.strip()
                if len(cleaned) >= self.min_claim_length:
                    # Adjust offsets for leading/trailing whitespace
                    l_trim = len(sent_text) - len(sent_text.lstrip())
                    r_trim = len(sent_text) - len(sent_text.rstrip())
                    c_start = sent_start + l_trim
                    c_end = sent_end - r_trim
                    claims.append(
                        Claim(
                            id=f"claim_{claim_idx}",
                            text=cleaned,
                            start_char=c_start,
                            end_char=c_end,
                            sentence_idx=sent_idx,
                        )
                    )
                    claim_idx += 1
            else:
                # Split on semicolons or clause delimiters while preserving exact character offsets
                clause_spans = self._split_clauses_in_sentence(sent_text, sent_start)
                for c_text, c_start, c_end in clause_spans:
                    cleaned = c_text.strip()
                    if len(cleaned) >= self.min_claim_length:
                        l_trim = len(c_text) - len(c_text.lstrip())
                        r_trim = len(c_text) - len(c_text.rstrip())
                        claims.append(
                            Claim(
                                id=f"claim_{claim_idx}",
                                text=cleaned,
                                start_char=c_start + l_trim,
                                end_char=c_end - r_trim,
                                sentence_idx=sent_idx,
                            )
                        )
                        claim_idx += 1

        return claims

    def _split_into_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with (sentence_text, start_char, end_char)."""
        sentence_spans = []
        # Match potential sentence terminators: .!? or double newlines
        pattern = re.compile(r'([.!?]+(?:\s+|\Z)|(?:\r?\n){1,})')
        start = 0

        for match in pattern.finditer(text):
            end = match.end()
            candidate = text[start:end]
            
            # Check if this period is part of an abbreviation or decimal number
            stripped = candidate.rstrip()
            words = stripped.split()
            last_word = words[-1].lower() if words else ""
            
            # Check if decimal number like 3.14
            is_decimal = False
            if match.group().startswith('.') and match.start() > 0 and match.end() < len(text):
                prev_char = text[match.start() - 1]
                next_char = text[match.end()] if match.end() < len(text) else ''
                if prev_char.isdigit() and next_char.isdigit():
                    is_decimal = True

            # If it's an abbreviation or decimal, check if it should suppress split
            next_text = text[match.end():].lstrip()
            next_is_capital = bool(next_text and next_text[0].isupper())

            if is_decimal:
                continue

            if last_word in TITLES:
                continue

            if last_word in GENERAL_ABBREVS and not next_is_capital:
                continue

            # If last word ends in '.' like 'u.s.' and followed by uppercase, it's sentence boundary!
            if candidate.strip():
                sentence_spans.append((candidate, start, end))
            start = end

        if start < len(text):
            remainder = text[start:]
            if remainder.strip():
                sentence_spans.append((remainder, start, len(text)))

        return sentence_spans

    def _split_clauses_in_sentence(
        self, sent_text: str, sent_offset: int
    ) -> list[tuple[str, int, int]]:
        """Split a sentence into clauses on semicolons."""
        clauses = []
        start = 0
        pattern = re.compile(r';\s*')
        for match in pattern.finditer(sent_text):
            end = match.start()
            clause_text = sent_text[start:end]
            if clause_text.strip():
                clauses.append((clause_text, sent_offset + start, sent_offset + end))
            start = match.end()

        if start < len(sent_text):
            remainder = sent_text[start:]
            if remainder.strip():
                clauses.append((remainder, sent_offset + start, sent_offset + len(sent_text)))

        return clauses
