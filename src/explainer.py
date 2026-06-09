from src import prompts

def explain_input_feature(feature: dict, window_size: int = 1) -> str:
    examples = []
    for act in feature["activations"]:
        i = act["maxValueTokenIndex"]
        window = act["tokens"][max(0, i - window_size + 1): i + 1]
        examples.append("".join(window))
    return prompts.INPUT_FEATURE.format(examples="\n".join(examples))

def explain_output_feature(feature: dict, window_size: int = 1) -> str:
    raise NotImplementedError


def explain_input_output_feature(feature: dict, window_size: int = 1) -> str:
    raise NotImplementedError


def explain_feature(feature: dict, window_size: int = 1) -> str:
    raise NotImplementedError
