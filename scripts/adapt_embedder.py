#!/usr/bin/env python
"""Adapt an embedder to the worlds' vocabulary using Lorepedia text only.

The retrieval experiment leaves two readings of why dense retrieval trails BM25 on \\ds: the
embedder may be weaker at ranking this text, or it may simply never have seen vocabulary like
it. Separating them needs a retriever adapted to the lexicon but not to the items, which is what
the Lorepedia provides: same worlds, same register, no question, answer or supporting-fact
annotation, and no document that any item depends on.

The default objective is the inverse cloze task. A sentence drawn from an article is the
pseudo-query and the rest of that article is its positive; the other articles in the batch are
negatives. This trains the short-to-long asymmetry the evaluation actually scores, which
passage-to-passage objectives do not, and it needs no dropout, so it applies to models whose
configuration sets dropout to zero. The SimCSE variant is kept for a robustness check.

Nothing in either objective reads an item document, a question, an answer or a supporting-fact
annotation, so no signal the evaluation measures can enter through training.

Pooling and prompts follow the model rather than a fixed choice, because vLLM pools by model
config at indexing time (MEAN for MiniLM, CLS for SimCSE) and training a representation the
index never sees would adapt nothing. For the same reason the sentence-transformers pooling
files are copied into the output directory: without them vLLM would pool the adapted checkpoint
differently from the base, and the comparison would measure that instead of the adaptation.

    python scripts/adapt_embedder.py --model sentence-transformers/all-MiniLM-L6-v2 \\
        --pooling mean --lorepedia dist/lorepedia/lorepedia.jsonl --output adapters/minilm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Written by sentence-transformers alongside the weights. vLLM reads them to decide pooling and
# normalisation, so an adapted checkpoint without them is pooled by vLLM's fallback instead.
ST_FILES = ("modules.json", "config_sentence_transformers.json", "sentence_bert_config.json",
            "1_Pooling/config.json", "2_Normalize/config.json")


def articles(path: Path, limit: int | None, seed: int) -> list[tuple[str, list[str]]]:
    out = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            if len(record["sentences"]) >= 2:
                out.append((record["title"], record["sentences"]))
    if limit and len(out) > limit:
        out = random.Random(seed).sample(out, limit)
    return out


def pool(hidden: torch.Tensor, mask: torch.Tensor, how: str) -> torch.Tensor:
    if how == "cls":
        return hidden[:, 0]
    if how == "mean":
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1e-9)
    if how == "last":
        index = mask.sum(1) - 1
        return hidden[torch.arange(hidden.size(0), device=hidden.device), index]
    raise SystemExit(f"unknown pooling: {how}")


def encode(model, tokenizer, texts, pooling, max_length, device, autocast):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                      return_tensors="pt").to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        hidden = model(**batch).last_hidden_state
    # Pooling and the loss run in float32: a bfloat16 softmax over a batch of similarities
    # loses enough precision to distort the gradient at this temperature.
    return F.normalize(pool(hidden.float(), batch["attention_mask"], pooling), dim=-1)


def ict_pair(title: str, sentences: list[str], rng: random.Random, keep: float) -> tuple[str, str]:
    """A sentence as pseudo-query, the rest of its article as the positive.

    The sentence is left in the document a tenth of the time, as in the original task, so the
    model does not learn that a query string never occurs in its own answer.
    """
    index = rng.randrange(len(sentences))
    query = sentences[index]
    rest = sentences if rng.random() < keep else sentences[:index] + sentences[index + 1:]
    return query, title + "\n" + " ".join(rest)


def snapshot_dir(source: str) -> Path | None:
    """Locate the cached snapshot of a model repository.

    Not snapshot_download: the cache holds only the files vLLM ever needed, and that helper
    refuses a partial snapshot outright rather than returning what is there.
    """
    root = Path(source)
    if root.is_dir():
        return root
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    base = home / "hub" / ("models--" + source.replace("/", "--")) / "snapshots"
    snapshots = sorted(base.glob("*/")) if base.is_dir() else []
    return snapshots[-1] if snapshots else None


def copy_pooling_files(source: str, output: Path) -> None:
    """Carry the sentence-transformers pooling configuration over to the adapted checkpoint.

    A model that defines pooling and loses it here would be pooled by vLLM's fallback at
    indexing time, so the adapted run would differ from the unadapted one by pooling as well
    as by training and neither could be read. That case stops the job rather than warning.
    """
    root = snapshot_dir(source)
    if root is None:
        print(f"[adapt] no cached snapshot for {source}; no pooling files copied", flush=True)
        return
    copied, missing = [], []
    for name in ST_FILES:
        if (root / name).is_file():
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(root / name, output / name)
                copied.append(name)
            except OSError as error:  # noqa: PERF203 - each file is reported on its own
                missing.append(f"{name} ({error})")
    print(f"[adapt] pooling files copied from {root}: {copied or 'none present'}", flush=True)
    if missing:
        raise SystemExit(f"pooling files present in the base model but not copied: {missing}")
    if (root / "modules.json").is_file() and "1_Pooling/config.json" not in copied:
        raise SystemExit(
            f"{source} defines sentence-transformers pooling but 1_Pooling/config.json was not "
            "copied; the adapted checkpoint would be pooled differently from the base")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lorepedia", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pooling", required=True, choices=("cls", "mean", "last"))
    parser.add_argument("--objective", default="ict", choices=("ict", "simcse"))
    parser.add_argument("--query-prompt", default="", help="Prefix the model defines for queries.")
    parser.add_argument("--doc-prompt", default="", help="Prefix the model defines for documents.")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--query-max-length", type=int, default=64)
    parser.add_argument("--keep-sentence", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Train the input embeddings only, leaving the encoder fixed.")
    parser.add_argument("--no-autocast", action="store_true",
                        help="Run the forward pass in float32 rather than bfloat16.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus = articles(args.lorepedia, args.limit, args.seed)
    print(f"[adapt] {len(corpus)} Lorepedia articles, objective={args.objective}, "
          f"pooling={args.pooling}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Last-token pooling reads position (number of real tokens - 1), which is the final token
    # only under right padding; a decoder tokenizer that defaults to left padding would have it
    # read padding instead. Decoder tokenizers also often ship without a padding token.
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Weights in float32 with a bfloat16 forward: the optimiser needs float32 masters, and a
    # bfloat16 AdamW at this learning rate does not accumulate updates reliably.
    model = AutoModel.from_pretrained(args.model, dtype=torch.float32).to(device)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    autocast = device == "cuda" and not args.no_autocast
    if args.objective == "simcse":
        # SimCSE's positive pair is two dropout masks over one input. With dropout disabled the
        # two views are identical, the loss is trivially minimised and nothing is learned.
        rates = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
        if not rates or max(rates) == 0.0:
            raise SystemExit(
                f"{args.model} has no active dropout, so the SimCSE positive pair is degenerate; "
                "use --objective ict")

    # The claim under test is that the embedder has not seen vocabulary like this, not that it
    # ranks badly. Training the input embeddings alone addresses exactly that and leaves the
    # ranking behaviour of the encoder intact; full fine-tuning on a narrow in-domain objective
    # overwrites it, which shows up as loss on collections the adaptation never touched.
    if args.freeze_encoder:
        embeddings = model.get_input_embeddings()
        keep = {id(p) for p in embeddings.parameters()}
        for parameter in model.parameters():
            parameter.requires_grad = id(parameter) in keep
    trainable = [p for p in model.parameters() if p.requires_grad]
    total = sum(p.numel() for p in model.parameters())
    print(f"[adapt] training {sum(p.numel() for p in trainable) / 1e6:.1f}M of {total / 1e6:.1f}M "
          f"parameters", flush=True)
    optimiser = torch.optim.AdamW(trainable, lr=args.lr)
    steps = max(1, int(len(corpus) * args.epochs) // args.batch_size)
    warmup = max(1, steps // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda s: s / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup)),
    )
    print(f"[adapt] {steps} steps, batch {args.batch_size}, lr {args.lr}", flush=True)

    order = list(range(len(corpus)))
    rng.shuffle(order)
    position, losses, accuracies = 0, [], []
    for step in range(steps):
        if position + args.batch_size > len(order):
            rng.shuffle(order)
            position = 0
        batch = [corpus[i] for i in order[position : position + args.batch_size]]
        position += args.batch_size

        if args.objective == "ict":
            pairs = [ict_pair(title, sentences, rng, args.keep_sentence)
                     for title, sentences in batch]
            left = encode(model, tokenizer, [args.query_prompt + q for q, _ in pairs],
                          args.pooling, args.query_max_length, device, autocast)
            right = encode(model, tokenizer, [args.doc_prompt + d for _, d in pairs],
                           args.pooling, args.max_length, device, autocast)
        else:
            texts = [args.doc_prompt + title + "\n" + " ".join(sentences)
                     for title, sentences in batch]
            left = encode(model, tokenizer, texts, args.pooling, args.max_length, device, autocast)
            right = encode(model, tokenizer, texts, args.pooling, args.max_length, device, autocast)

        similarity = left @ right.T / args.temperature
        target = torch.arange(similarity.size(0), device=device)
        loss = F.cross_entropy(similarity, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimiser.step()
        schedule.step()
        optimiser.zero_grad(set_to_none=True)

        losses.append(loss.detach().item())
        accuracies.append((similarity.argmax(1) == target).float().mean().item())
        if not math.isfinite(losses[-1]):
            raise SystemExit("loss diverged; lower --lr")
        if step % 100 == 0 or step == steps - 1:
            window = slice(-100, None)
            print(f"[adapt] step {step}/{steps} loss {sum(losses[window]) / len(losses[window]):.4f} "
                  f"in-batch acc {sum(accuracies[window]) / len(accuracies[window]):.3f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    copy_pooling_files(args.model, args.output)
    (args.output / "adaptation.json").write_text(json.dumps({
        "base_model": args.model, "objective": args.objective, "pooling": args.pooling,
        "articles": len(corpus), "steps": steps, "batch_size": args.batch_size, "lr": args.lr,
        "temperature": args.temperature, "max_length": args.max_length,
        "query_max_length": args.query_max_length, "seed": args.seed,
        "query_prompt": args.query_prompt, "doc_prompt": args.doc_prompt,
        "freeze_encoder": args.freeze_encoder, "epochs": args.epochs,
        "autocast_bfloat16": autocast,
        # The first step is recorded alongside the last because a flat downstream result has
        # two very different explanations -- the model learned the objective and the ranking
        # did not move, or the run never trained -- and only the trajectory separates them.
        # Without it the distinction lives in a Slurm log that no analysis can cite.
        "initial_loss": losses[0],
        "initial_in_batch_accuracy": accuracies[0],
        "final_loss": sum(losses[-100:]) / len(losses[-100:]),
        "final_in_batch_accuracy": sum(accuracies[-100:]) / len(accuracies[-100:]),
        "source": "lorepedia only; no item documents, questions or answers",
    }, indent=2))
    print(f"[adapt] -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
