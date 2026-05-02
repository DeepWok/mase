import logging
from pathlib import Path

import datasets
import random
import torch
import transformers


def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, use_fast=False, use_auth_token=hf_token
        )

    if eval_mode:
        testdata = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="train"
        )
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, use_fast=False, use_auth_token=hf_token
        )

    if eval_mode:
        valdata = datasets.load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
        valenc = valenc.input_ids[:, : (256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids

        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model, use_fast=False, use_auth_token=hf_token
        )

    if eval_mode:
        testdata = datasets.load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        return testenc
    else:
        traindata = datasets.load_dataset(
            "ptb_text_only", "penn_treebank", split="train"
        )
        trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_file_loader(spec, nsamples=128, seqlen=2048):
    """Load a calibration loader saved by ``TokenCollector``.

    spec format: ``"file:<path>"`` (the prefix is stripped before reading).
    The file must be the dict produced by ``TokenCollector._save_and_detach``,
    or a raw ``[(input_ids, target), ...]`` list for backward compatibility.
    """
    path = spec[5:] if spec.startswith("file:") else spec
    payload = torch.load(path, weights_only=False)

    if isinstance(payload, dict) and "loader" in payload:
        loader = payload["loader"]
        saved_seqlen = int(payload.get("seqlen", -1))
        if saved_seqlen > 0 and saved_seqlen != int(seqlen):
            logging.warning(
                "[get_file_loader] requested seqlen=%d but file has %d; using file's value.",
                int(seqlen), saved_seqlen,
            )
    else:
        loader = payload

    have = len(loader)
    if have < int(nsamples):
        logging.warning(
            "[get_file_loader] file %s has %d samples but caller requested %d; "
            "GPTQ will run with %d samples.",
            path, have, int(nsamples), have,
        )
    return loader[: int(nsamples)] if have > int(nsamples) else loader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model="", hf_token=None, eval_mode=False
):
    if name.startswith("file:"):
        return get_file_loader(name, nsamples=nsamples, seqlen=seqlen)
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "ptb" in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if "c4" in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
