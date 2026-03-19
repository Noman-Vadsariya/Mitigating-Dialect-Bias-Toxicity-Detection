# -*- encoding: utf-8 -*-
import re
import sys

assert sys.maxunicode >= 0x10FFFF, "Python 3 UCS-4 build is required."

# Unicode ranges for emoji and symbols
JUNK_RE = (
    '[' +
    '\U00010000-\U0001FFFF' +  # Supplementary Multilingual Plane
    '\U00020000-\U0002FFFF' +  # Supplementary Ideographic Plane
    '\U0000E000-\U0000EFFF' +  # Private Use Area
    '\U00002500-\U00002BFF' +  # Box Drawing, Block Elements, etc.
    '\U0000200B-\U0000200D' +  # Zero-width joiners
    '\U0000FE0E-\U0000FE0F' +  # Variation Selectors
    ']'
)

SUB_RE = re.compile(r'\s*' + JUNK_RE + r'\s*', re.UNICODE)

def clean_emoji_and_symbols(text):
    """
    Removes emojis and symbols from the given text.
    """
    assert isinstance(text, str), "Input must be a string."
    return SUB_RE.sub(" ", text)