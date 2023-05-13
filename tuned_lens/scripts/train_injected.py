from transformers import AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
import os
import json
import torch
from tqdm import trange
import wandb
from fire import Fire


def run(
    batch_size: int = 32,
    n_epochs: int = 10,
    completions_path: str = "data.jsonl",
    model_name: str = "EleutherAI/pythia-410m-deduped",
    layers: list[int] = [0, 6, 10, 16],
    lr: float = 5e-4,
    max_length: int = 512,
):
    for k, v in locals().items():
        print(f"{k}: {v}")

    wandb.init(project="sft-lens", config=locals())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ref_model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
    decoder_model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    with open(completions_path, "r") as f:
        completions = [json.loads(l) for l in f]

    all_texts: list[tuple[str, str, str]] = []
    for (text, text_end), answers in completions:
        for answer in answers:
            all_texts.append((text, text_end, answer))

    ref_model_dim = ref_model.config.hidden_size
    decoder_model_dim = decoder_model.config.hidden_size

    adapters = [torch.nn.Linear(ref_model_dim, decoder_model_dim).to(device) for _ in layers]
    for adapter in adapters:
        # identity initialization
        adapter.weight.data.copy_(torch.eye(ref_model_dim, device=device))

    adapter_parameters = sum([list(adapter.parameters()) for adapter in adapters], [])

    optimizer = torch.optim.Adam(list(decoder_model.parameters()) + adapter_parameters, lr=lr)
    # linear lr warmup
    warmup_steps = 0.1 * len(all_texts) * n_epochs / batch_size
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1, step / warmup_steps))

    eos = tokenizer.eos_token

    for epoch in range(n_epochs):
        pbar = trange(0, len(all_texts), batch_size)

        for i in pbar:
            batch = all_texts[i : i + batch_size]

            with torch.no_grad():
                inputs = tokenizer(
                    [eos + t[0] for t in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )
                # get activations at every layer
                outputs = ref_model(**inputs.to(device), output_hidden_states=True)
                # get activations at last position of text
                last_tok_pos = inputs["attention_mask"].sum(dim=1) - 1
                last_pos_activations = [
                    outputs["hidden_states"][l][torch.arange(len(outputs["hidden_states"][l])), last_tok_pos, :]
                    for l in layers
                ]

            losses = []
            generations = []

            for layer_i, (layer, state) in enumerate(zip(layers, last_pos_activations)):
                expected_strings = [f"{eos}Layer {layer}\n{t[2]}{eos}" for t in batch]
                expected_tokens = tokenizer(
                    expected_strings, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(device)

                def state_insertion_hook(module, inp, out):
                    out[:, 0, :] = adapters[layer_i](state[: len(out)])
                    return out

                handle = decoder_model.gpt_neox.embed_in.register_forward_hook(state_insertion_hook)

                try:
                    # generate sequence with injected state of batch elt
                    with torch.no_grad():
                        inputs = tokenizer(f"{eos}Layer {layer}\n", return_tensors="pt")
                        top_generation_tokens = decoder_model.generate(
                            **inputs.to(device),
                            do_sample=False,
                            max_new_tokens=100,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        top_generation_string = tokenizer.decode(top_generation_tokens[0])
                        generations.append(top_generation_string)

                    # compute logits
                    logits = (
                        decoder_model(**expected_tokens, labels=expected_tokens.input_ids)
                        .logits[:, :-1, :]
                        .contiguous()
                    )
                finally:
                    handle.remove()

                labels = expected_tokens.input_ids[:, 1:].contiguous()
                loss_per_pos = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
                ).reshape(labels.shape)
                masked_loss = loss_per_pos * expected_tokens.attention_mask[:, :-1]
                loss = masked_loss.sum() / expected_tokens.attention_mask[:, :-1].sum()

                losses.append(loss.item())

                loss /= len(last_pos_activations)
                loss.backward()

            # clip grad
            torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            pbar.set_description(f"loss: {sum(losses)/len(losses):.4f}")

            generation_table = wandb.Table(columns=["layer", "generation", "text", "text end", "label"])
            for layer_i, s in enumerate(generations):
                generation_table.add_data(layers[layer_i], s, *batch[0])
            to_log = {
                "loss": sum(losses) / len(losses),
                **{f"losses/layer_{layers[i]}": loss for i, loss in enumerate(losses)},
                "generations": generation_table,
            }
            wandb.log(to_log)

    wandb.finish()


if __name__ == "__main__":
    Fire(run)
