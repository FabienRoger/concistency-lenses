from tuned_lens.scripts.generation_utils import threaded_generations
from tuned_lens.scripts.ingredients import (
    Data,
)
from transformers import AutoTokenizer
import os
from fire import Fire
import json

SYSTEM_PROMPT = """As a language model explainer, your task is to analyze a given truncated text and deduce what information is present in the activations of a language model at any layer in the final position. Do not attempt to complete the text provided by the user, and directly provide information in the following format:
Current word: <last word or punctuation of the text>
Current sentence: <current sentence in which the last word is present, summarized if and only if it is more than 10 words long, and otherwise repeated at-verbatim>
Probable next words: <comma-separated list of 5 most probable next words/punctuation>
Text style: <comma-separated list of 10 words describing the style of the text, its tone, the writer position, ...>
Paragraph summary: <summary of the paragraph, of at most 30 words>
Text summary: <summary of the text, of at most 30 words>
Probable next sentences: <list of the 3 most likely completions of this text, *each should be exactly 8 words long*: interrupt yourself in the middle of the sentence if you exceed 8 words!! Hint: completions should porbably start with one of the most probable next words>
1. <first completion>
2. <second completion>
3. <third completion>
The prompt of the user will always be of the form
"<Text that ends with word or punctuation <w>>
->"
and you should start immediatly with
Current word: <w>"""


def run(
    pile_path: str = "~/datasets/pile/val.jsonl",
    write_path: str = "data.jsonl",
    model_name: str = "EleutherAI/pythia-70m-deduped",
    max_samples: int = 40000,
    max_length: int = 256,
    additional_tokens_kept: int = 20,
    min_length: int = 20,
    nb_solutions: int = 1,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.0,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    expanded_path = os.path.expanduser(pile_path)
    data = Data(name=[expanded_path], max_length=max_length + additional_tokens_kept)
    dataset, _ = data.load(tokenizer)

    eos = "<|endoftext|>"

    def tokenize(toks):
        return tokenizer.decode(toks).split(eos)[-1]

    texts = []
    i = 0
    while len(texts) < max_samples and i < len(dataset):
        pt = dataset[len(texts)]
        if len(pt["input_ids"]) > min_length + additional_tokens_kept:
            texts.append(
                (
                    tokenize(pt["input_ids"][:-additional_tokens_kept]),
                    tokenize(pt["input_ids"][-additional_tokens_kept:]),
                )
            )
        i += 1

    prompts = [
        ((text, e), [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text + "\n->"}])
        for text, e in texts
    ]
    results = threaded_generations(prompts, nb_solutions=nb_solutions, model=model, temperature=temperature)

    expanded_write_path = os.path.expanduser(write_path)
    with open(expanded_write_path, "w") as f:
        for r in results:
            json.dump(r, f)
            f.write("\n")


if __name__ == "__main__":
    Fire(run)
