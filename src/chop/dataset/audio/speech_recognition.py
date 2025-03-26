import logging
from pathlib import Path
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset
import torchaudio
from ..utils import add_dataset_info
from chop.models import DataCollatorCTCWithPadding


logger = logging.getLogger(__name__)

LIBRISPEECH_CONFIG = {
    "sample_rate": 16000,
    "normalize_waveform": True,
    "tokenizer_checkpoint": "facebook/wav2vec2-base-960h",
    "max_audio_length": 16 * 16000, # Validate?
}

processor = Wav2Vec2Processor.from_pretrained(LIBRISPEECH_CONFIG["tokenizer_checkpoint"])

@add_dataset_info(
    name="nyalpatel/condensed_librispeech_asr",
    dataset_source="hf_datasets",
    available_splits=("train", "validation", "test"),
    seq2seqLM=True,
    num_features=LIBRISPEECH_CONFIG["sample_rate"] * 16, # Validate?
    data_collator_cls=DataCollatorCTCWithPadding,
)

class CondensedLibrispeechASRDataset(Dataset):
    def __init__(
        self,
        path,
        split="train",
        tokenizer=None,
        max_token_len=None,
        num_workers=0,
        load_from_cache_file=True,
        auto_setup=False,
        dataset_path=None,
        config=LIBRISPEECH_CONFIG
    ):
        super().__init__()
        self.split = split
        self.dataset_path = Path(dataset_path) if dataset_path else Path(path)
        self.config = config
        self.X = None
        self.Y = None
        self.raw = None 
        
        # Store the additional parameters that the framework passes
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        
        # Automatically call setup if requested
        if auto_setup:
            self.prepare_data()
            self.setup()

    def __len__(self):
        if self.X is None:
            raise ValueError(
                "Dataset is not setup. Please call `dataset.prepare_data()` + `dataset.setup()` or pass `auto_setup=True` before using the dataset."
            )
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.X is None or self.Y is None:
            raise ValueError("Dataset is not setup. Call prepare_data() and setup() first.")
        print(f"[DEBUG] __getitem__ index {idx} returning raw_labels: {self.raw[idx]}")
        return {"input_values": self.X[idx], "labels": self.Y[idx], "raw_labels": self.raw[idx]}


    def prepare_data(self) -> None:
        # Map the standard split name to a Librispeech-specific split for dataset loading
        librispeech_split = "train.clean.100"  # default
        
        if self.split == "train":
            hf_split = "train.clean.100"
        elif self.split == "validation":
            hf_split = "validation.clean"
        elif self.split in ["test", "pred"]:
            hf_split = "test.clean"
        else:
            raise ValueError(f"Unknown split {self.split}")
            
        _preprocess_librispeech_dataset(
            save_path=self.dataset_path,
            config=self.config,
            hf_split=hf_split, 
            local_split_name=self.split
        )

    def setup(self) -> None:
        """Load the .pt files for our split into memory."""
        if self.split == "train":
            x_path, y_path, raw_path = "X_train.pt", "Y_train.pt", "raw_train.pt"
        elif self.split == "validation":
            x_path, y_path, raw_path = "X_validation.pt", "Y_validation.pt", "raw_validation.pt"
        elif self.split == "test":
            x_path, y_path, raw_path = "X_test.pt", "Y_test.pt", "raw_test.pt"
        elif self.split == "pred":
            x_path, y_path, raw_path = "X_pred.pt", "Y_pred.pt", "raw_pred.pt"
        else:
            raise ValueError(f"Unknown split {self.split}")
            
        assert (self.dataset_path / x_path).exists(), \
            f"Missing file {x_path}, have you called prepare_data()?"
        
        self.X = torch.load(self.dataset_path / x_path)
        self.Y = torch.load(self.dataset_path / y_path)
        self.raw = torch.load(self.dataset_path / raw_path)

        print(f"[DEBUG] Loaded {len(self.X)} samples for {self.split} split.")


def _preprocess_librispeech_dataset(
    save_path: Path,
    config: dict,
    hf_split: str, 
    local_split_name: str
):
    """
    Loads a single HF subset (e.g. "train.clean.100"), preprocesses, and saves
    X_<local_split_name>.pt, Y_<local_split_name>.pt, raw_<local_split_name>.pt
    """
    dataset = load_dataset("nyalpatel/condensed_librispeech_asr", split=hf_split)

    input_values, labels, raw_labels = [], [], []
    printed_example = False

    for example in dataset:
        waveform = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        text = example["text"]

        if isinstance(waveform, list):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        if waveform.shape[0] > config["max_audio_length"]:
            continue

        if sampling_rate != config["sample_rate"]:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=config["sample_rate"]
            )(waveform)

        if config["normalize_waveform"]:
            if waveform.numel() > 0 and waveform.std() > 0:
                waveform = (waveform - waveform.mean()) / waveform.std()
            else:
                waveform = waveform - waveform.mean()

        with processor.as_target_processor():
            label = processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        input_values.append(waveform)
        labels.append(label)
        raw_labels.append(text)

        if not printed_example:  # Just debug-print once
            print(f"[DEBUG] Preprocess sample: raw label = '{text}'")
            printed_example = True

    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(input_values, save_path / f"X_{local_split_name}.pt")
    torch.save(labels, save_path / f"Y_{local_split_name}.pt")
    torch.save(raw_labels, save_path / f"raw_{local_split_name}.pt")

    logger.info(
        f"✅ Condensed Librispeech dataset for '{local_split_name}' "
        f"({hf_split}) preprocessed and saved! "
        f"Total samples: {len(input_values)}"
    )
