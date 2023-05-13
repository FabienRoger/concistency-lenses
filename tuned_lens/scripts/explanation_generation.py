# %%
from tuned_lens.scripts.generation_utils import threaded_generations
from tuned_lens.scripts.ingredients import (
    Data,
)
from transformers import AutoTokenizer

SYSTEM_PROMPT = """As a language model explainer, your task is to analyze a given truncated text and deduce what information is present in the activations of a language model at any layer in the final position. Keep in mind that the activations at the last position are primarily used to predict the next token. Consequently, they should contain information such as the next word, the past few words, the type of text, the type of text that is likely to follow, natural language processing information (e.g., which words are nouns), stylistic information, and more. Your output should be approximately 500 words in length and provide specific content that aids in predicting the next token, avoiding vague generalizations.

Do not attempt to complete the text provided by the user; instead, begin describing the information present in the activations at the last position immediately."""
# %%
pile_path = "/home/fabien/datasets/pile/val.jsonl"

model_name = "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data = Data(name=[pile_path], max_length=256)
dataset, _ = data.load(tokenizer)
# %%
print(tokenizer.decode(dataset[5]["input_ids"]))
# %%
eos = "<|endoftext|>"
n = 10


def tokenize(toks):
    return tokenizer.decode(toks).split(eos)[-1]


texts = [(tokenize(dataset[i]["input_ids"][:-3]), tokenize(dataset[i]["input_ids"][-3:])) for i in range(n)]
for t, e in texts:
    if eos in t:
        print(t, "|", e)

# %%
prompts = [
    ((text, e), [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]) for text, e in texts
]
results = threaded_generations(prompts, nb_solutions=2, model="gpt-3.5-turbo")
# %%
for r in results:
    print("|".join(r[0]))
    print("=================>")
    for s in r[1]:
        print(s)
        print("-" * 80)
    print("=" * 80)
# %%
