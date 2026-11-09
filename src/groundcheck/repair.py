"""Claim repair engine: drops unsupported claims and produces a unified diff."""

import difflib
import re

from groundcheck.types import ClaimVerdict, ClaimVerification, PolicyConfig


class ClaimRepairer:
    """Repairs answers by removing unsupported claims and producing clean diffs."""

    def repair(
        self,
        original_text: str,
        verifications: list[ClaimVerification],
        policy: PolicyConfig,
    ) -> tuple[str, str]:
        """Drop unsupported claims from original_text and return (repaired_text, unified_diff)."""
        if not verifications or not original_text:
            return original_text, ""

        # Identify which claims should be removed
        unsupported_claims = [
            cv
            for cv in verifications
            if cv.verdict == ClaimVerdict.UNSUPPORTED
            or (policy.abstain_on_weak and cv.verdict == ClaimVerdict.WEAK)
        ]

        if not unsupported_claims:
            return original_text, ""

        # If every claim is unsupported, empty/clear answer
        if len(unsupported_claims) == len(verifications):
            repaired_text = ""
            diff = self._make_diff(original_text, repaired_text)
            return repaired_text, diff

        # Sort claims to remove in reverse order of start_char to keep offsets valid
        sorted_removals = sorted(
            unsupported_claims, key=lambda cv: cv.claim.start_char, reverse=True
        )

        text_chars = list(original_text)

        for cv in sorted_removals:
            s_idx = cv.claim.start_char
            e_idx = cv.claim.end_char

            # Extend boundaries to include trailing period or semicolon and whitespace
            while e_idx < len(text_chars) and text_chars[e_idx] in " \t.;,!?\r\n":
                e_idx += 1
            # Also clean leading whitespace if at beginning of line
            while s_idx > 0 and text_chars[s_idx - 1] in " \t":
                s_idx -= 1

            # Blank out the characters of this claim
            text_chars[s_idx:e_idx] = []

        repaired_text = "".join(text_chars)

        # Clean up formatting: multiple spaces, orphan commas/semicolons, empty lines
        repaired_text = re.sub(r"[ \t]+", " ", repaired_text)
        repaired_text = re.sub(r"^[ \t]+|[ \t]+$", "", repaired_text, flags=re.MULTILINE)
        repaired_text = re.sub(r"\n{3,}", "\n\n", repaired_text)
        repaired_text = re.sub(r"\s+([.,!?;:])", r"\1", repaired_text)
        repaired_text = repaired_text.strip()

        diff = self._make_diff(original_text, repaired_text)
        return repaired_text, diff

    def _make_diff(self, original: str, repaired: str) -> str:
        """Generate a standard unified diff between original and repaired answers."""
        orig_lines = original.splitlines(keepends=True)
        rep_lines = repaired.splitlines(keepends=True) if repaired else []
        diff_lines = list(
            difflib.unified_diff(
                orig_lines,
                rep_lines,
                fromfile="original_answer",
                tofile="repaired_answer",
                lineterm="",
            )
        )
        return "\n".join(diff_lines)
