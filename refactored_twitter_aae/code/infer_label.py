# import os
# import sys

# # Add path to twitteraae code
# sys.path.append(os.path.dirname(__file__))

import predict

# Load model once (important)
predict.load_model()

def get_aae_score(text):
    """
    Returns probability that text is AAE.
    """
    # simple tokenization (important!)
    tokens = text.lower().split()

    # predict returns list of probabilities
    probs = predict.predict_language(tokens)

    # index 0 = AAE (from model)
    return probs[0]


def get_dialect_label(text, threshold=0.5):
    score = get_aae_score(text)
    return "AAE" if score > threshold else "SAE"



print(get_aae_score("yo bro you wild"))      # should be HIGH (~0.6–0.9)
print(get_aae_score("this is a test"))       # should be LOW (~0.0–0.2)

print(get_dialect_label("yo bro you wild"))

# print(get_aae_score(""))       # should be LOW (~0.0–0.2)
