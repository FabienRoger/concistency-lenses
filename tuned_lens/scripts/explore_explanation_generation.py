from tuned_lens.scripts.generation_utils import threaded_generations
from tuned_lens.scripts.ingredients import (
    Data,
)
from transformers import AutoTokenizer

SYSTEM_PROMPT = """As a language model explainer, your task is to analyze a given truncated text and deduce what information is present in the activations of a language model at any layer in the final position. Do not attempt to complete the text provided by the user, and directly provide information in the following format:
Current word: <last word or punctuation of the text>
Current sentence: <current sentence in which the last word is present, summarized if and only if it is more than 10 words long, and otherwise repeated at-verbatim>
Probable next words: <comma-separated list of 5 most probable next words/punctuation>
Text style: <comma-separated list of 3-10 words describing the style of the text>
Paragraph summary: <summary of the paragraph, of at most 30 words>
Text summary: <summary of the text, of at most 30 words>
Probable next sentences: <list of the 3 most likely completions of this text, *each should be exactly 8 words long*: interrupt yourself in the middle of the sentence if you exceed 8 words!! Hint: completions should porbably start with one of the most probable next words>
1. <first completion>
2. <second completion>
3. <third completion>"""


pile_path = "/home/fabien/datasets/pile/val.jsonl"

model_name = "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)
data = Data(name=[pile_path], max_length=256)
dataset, _ = data.load(tokenizer)
# %%

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
