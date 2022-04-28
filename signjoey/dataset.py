# coding: utf-8
"""
Data module
"""
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch


def load_dataset_file(filename):
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        cfg,
        path: str,
        tokeniser,
        fields,
        field_names,
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [(field_names[i], fields[i]) for i in range(len(fields))]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = s["name"]
                if seq_id in samples:
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    samples[seq_id] = {
                        "name": s["name"],
                        "signer": s["signer"],
                        "gloss": s["gloss"],
                        "text": s["text"],
                        "sign": s["sign"],
                    }
                    if "pose" in s:
                        samples[seq_id]["pose"] = s["pose"]
                    if "flow" in s:
                        samples[seq_id]["flow"] = s["flow"]

        normalise_before_concat = cfg["normalise_before_concat"]

        examples = []
        for s in samples:
            sample = samples[s]
            l = [
                sample["name"],
                sample["signer"],
                # This is for numerical stability
                sample["sign"] + 1e-8,
                sample["gloss"].strip(),
                #sample["text"].strip(),
                tokeniser.encode(sample["text"].strip().lower())
                if cfg["level"] == "subword" else sample["text"].strip(),
            ]
            if cfg["concat_pose"]:
                l[2] = torch.cat((
                    torch.nn.functional.normalize(l[2], dim=-1) 
                    if normalise_before_concat else l[2], 
                    torch.nn.functional.normalize(sample["pose"], dim=-1)
                    if normalise_before_concat else sample["pose"]
                ), dim=1)
            if cfg["concat_flow"]:
                #flow = torch.cat((torch.zeros(1, sample["flow"].size(-1)), sample["flow"]), dim=0)
                if cfg["flow_options"]["copy_first"]:
                    flow = torch.cat((sample["flow"][:1,:], sample["flow"]), dim=0)
                elif cfg["flow_options"]["copy_last"]:
                    flow = torch.cat((sample["flow"], sample["flow"][-1:,:]), dim=0)
                elif cfg["flow_options"]["zeros_first"]:
                    flow = torch.cat((torch.zeros(1,sample["flow"].size(-1)), sample["flow"]), dim=0)
                elif cfg["flow_options"]["zeros_last"]:
                    flow = torch.cat((sample["flow"], torch.zeros(1,sample["flow"].size(-1))), dim=0)
                l[2] = torch.cat((
                    torch.nn.functional.normalize(l[2], dim=-1)
                    if normalise_before_concat else l[2], 
                    torch.nn.functional.normalize(flow, dim=-1)
                    if normalise_before_concat else flow
                ), dim=1)
            if cfg["pose_stream"]:
                l.append(sample["pose"])
            if cfg["flow_stream"]:
                flow = torch.cat((torch.zeros(1, sample["flow"].size(-1)), sample["flow"]), dim=0)
                l.append(flow)
            examples.append(
                data.Example.fromlist(
                    l,
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
