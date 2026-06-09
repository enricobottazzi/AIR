import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from src.explainer import explain_acts, explain_logits
from src.correlation_score import gen_correlation_score, embed
from src.llm import ExplainerSetup, complete

API_KEY = next(
    l.split("=", 1)[1]
    for l in Path(".env").read_text().splitlines()
    if l.startswith("OPENROUTER_API_KEY=")
)

setup = ExplainerSetup(model="google/gemini-2.5-flash-lite")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

feature = json.loads(Path("data/gemma-3-27b-it_33_142228.json").read_text())

# input
prompt_1, examples_1, weights_1 = explain_acts(feature, window=(0, 0))

# positive logits
prompt_2, examples_2, weights_2 = explain_logits(feature, positive=True)

# negative logits
prompt_3, examples_3, weights_3 = explain_logits(feature, positive=False)

# output 
prompt_4, examples_4, weights_4 = explain_acts(feature, window=(1, 1))

# cross
prompt_5, examples_5, weights_5 = explain_acts(feature, window=(-25, 25))

# compute correlation scores
correlation_score_1 = gen_correlation_score(embed(examples_1, model), w=weights_1)
correlation_score_2 = gen_correlation_score(embed(examples_2, model), w=weights_2)
correlation_score_3 = gen_correlation_score(embed(examples_3, model), w=weights_3)
correlation_score_4 = gen_correlation_score(embed(examples_4, model), w=weights_4)
correlation_score_5 = gen_correlation_score(embed(examples_5, model), w=weights_5)

print(correlation_score_1)
print(correlation_score_2)
print(correlation_score_3)
print(correlation_score_4)
print(correlation_score_5)