"""Encode a notebook batch with Nemotron's supported Transformers runtime."""

import json
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def main(request_path: Path) -> None:
    request = json.loads(request_path.read_text())
    spec = request["spec"]
    assert version("transformers") == spec["isolated_transformers"]
    assert version("tokenizers") == spec["isolated_tokenizers"]
    torch.manual_seed(request["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], revision=request["revision"], padding_side="left",
    )
    model_kwargs = {"attn_implementation": "eager", "torch_dtype": torch.bfloat16}
    if "code_repo" in spec:
        model_kwargs.update(trust_remote_code=True, code_revision=request["revision"])
    model = AutoModel.from_pretrained(
        spec["model"], revision=request["revision"], **model_kwargs,
    ).to("cuda").eval()
    texts = [spec["prefix"] + text for text in request["texts"]]
    lengths = [len(tokenizer(text, truncation=False)["input_ids"]) for text in texts]

    # Verify the official bidirectional attention path and left-padding behavior.
    probe = tokenizer(["Short.", "A somewhat longer example sentence."],
                      padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        attention = model(**probe, output_attentions=True).attentions[0]
    real = probe.attention_mask[0].nonzero().flatten()
    pad = (~probe.attention_mask[0].bool()).nonzero().flatten()
    assert attention[0, :, real[0], real[-1]].gt(0).all()
    assert attention[0, :, real[0], pad].eq(0).all()
    del probe, attention

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    batches = []
    for start in range(0, len(texts), spec["batch_size"]):
        batch = tokenizer(
            texts[start:start + spec["batch_size"]], padding=True, truncation=True,
            max_length=spec["max_length"], return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state.float()
            # Official model-card pooling, including instruction tokens.
            hidden = hidden.masked_fill(~batch.attention_mask[..., None].bool(), 0.0)
            pooled = hidden.sum(dim=1) / batch.attention_mask.sum(dim=1)[..., None]
            embeddings = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        batches.append(embeddings.cpu().numpy())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output = Path(request["output"])
    np.save(output / "embeddings.npy", np.concatenate(batches).astype(np.float32))
    info = {
        "identity": request["identity"], "slug": request["slug"], "model": spec["model"],
        "revision": request["revision"], "code_revision": request["revision"] if "code_repo" in spec else None,
        "representation": request["representation"], "ids": request["ids"],
        "seconds": elapsed, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "gpu": torch.cuda.get_device_name(0), "transformers": version("transformers"),
        "tokenizers": version("tokenizers"), "torch": version("torch"),
        "huggingface_hub": version("huggingface-hub"),
        "bidirectional_attention_verified": True, "tokens": lengths,
    }
    (output / "isolated_result.json").write_text(json.dumps(info, indent=2))
    print(f"{request['slug']}: official attention checks passed; inference {elapsed:.1f} s.", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
