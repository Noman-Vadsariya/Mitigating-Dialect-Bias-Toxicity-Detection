import re
import html

# Regular expressions for tokenization
URL_RE = re.compile(r'https?://\S+|www\.\S+')
USER_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#\w+')
PUNCT_RE = re.compile(r'[^\w\s]')
WHITESPACE_RE = re.compile(r'\s+')

def tokenize(text):
    """
    Tokenizes the input text into words, removing URLs, mentions, and hashtags.
    """
    assert isinstance(text, str), "Input must be a string."
    text = html.unescape(text)  # Decode HTML entities
    text = URL_RE.sub('', text)  # Remove URLs
    text = USER_RE.sub('', text)  # Remove user mentions
    text = HASHTAG_RE.sub('', text)  # Remove hashtags
    text = PUNCT_RE.sub('', text)  # Remove punctuation
    text = WHITESPACE_RE.sub(' ', text)  # Normalize whitespace
    return text.strip().split()