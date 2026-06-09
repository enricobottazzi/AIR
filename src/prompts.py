INPUT_FEATURE = """
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
