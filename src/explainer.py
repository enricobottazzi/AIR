from src import prompts

def explain_feature(feature: dict, window: tuple[int, int]) -> str:
    # `window` is an inclusive (start, end) range of token offsets relative to
    # maxValueTokenIndex: negative = preceding, 0 = the max token, positive = following.
    # e.g. (0, 0) is only the top activating token.
    start, end = window
    assert start <= end, "window start must be <= end"
    examples = []
    for act in feature["activations"]:
        i = act["maxValueTokenIndex"]
        tokens = [act["tokens"][i + o] for o in range(start, end + 1) if 0 <= i + o < len(act["tokens"])]
        examples.append("".join(tokens).replace("\u2581", "").strip())
    return prompts.EXAMPLE_LIST.format(examples="\n".join(examples))