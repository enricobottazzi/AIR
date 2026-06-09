PROMPT = """
You are explaining the behavior of a neuron in a neural network.
Your response should be a concise explanation (3 to 20 words) that captures what the neuron detects by finding patterns in the provided list of examples.

Rules:
- Keep your explanation concise (3 to 20 words).
- The explanation could be a single word, or phrase, or pattern.
- Avoid simply listing all the tokens. Instead, try to find patterns.
- Just say the pattern itself, and do not start with phrases like "words related to", "concepts related to", or "variations of the word".
- Do not start your explanation with "This neuron detects/predicts".
- Do not mention "tokens" or "patterns" in your explanation.
- Do not capitalize the first letter unless it is a proper noun.
- The explanation should be specific. For example, "unique words" is not a specific enough pattern, nor is "foreign words".
- Not ALL examples in the list have to match the exact same pattern, but a majority should.

Your response should be exactly a short phrase that explains the behavior of the neuron, not a full sentence.

<EXAMPLE_LIST>

{examples}

</EXAMPLE_LIST>

Explain the neuron above with a word or phrase, not a complete sentence.
"""

### TODO: rename into pre-process
def explain_acts(feature: dict, window: tuple[int, int]) -> tuple[str, list[str], list[float]]:
    # `window` is an inclusive (start, end) range of token offsets relative to
    # maxValueTokenIndex: negative = preceding, 0 = the max token, positive = following.
    # e.g. (0, 0) is only the top activating token.
    start, end = window
    assert start <= end, "window start must be <= end"
    examples, weights = [], []
    for act in feature["activations"]:
        i = act["maxValueTokenIndex"]
        tokens = [act["tokens"][i + o] for o in range(start, end + 1) if 0 <= i + o < len(act["tokens"])]
        examples.append("".join(tokens).replace("\u2581", "").strip())
        weights.append(act["maxValue"])
    return PROMPT.format(examples="\n".join(examples)), examples, weights

def explain_logits(feature: dict, positive: bool) -> tuple[str, list[str], list[float]]:
    # Top tokens the feature most promotes (positive) or suppresses (negative).
    key = "pos_str" if positive else "neg_str"
    examples = [t.replace("\u2581", "").strip() for t in feature[key]]
    weights = feature["pos_values" if positive else "neg_values"]
    return PROMPT.format(examples="\n".join(examples)), examples, weights