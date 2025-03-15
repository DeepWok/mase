import logging
from pathlib import Path
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2Processor
from torch.utils.data import Dataset
import torchaudio
from ..utils import add_dataset_info

logger = logging.getLogger(__name__)

LIBRISPEECH_CONFIG = {
    "sample_rate": 16000,
    "normalize_waveform": True,
    "tokenizer_checkpoint": "facebook/wav2vec2-base-960h",
    "train_size": 0.8,
    "validation_size": 0.1,
    "test_size": 0.1,
    "max_audio_length": 16 * 16000,
}

processor = Wav2Vec2Processor.from_pretrained(LIBRISPEECH_CONFIG["tokenizer_checkpoint"])

@add_dataset_info(
    name="nyalpatel/condensed_librispeech_asr",
    dataset_source="hf_datasets",
    # Change this to use standard split names that are already in the enum
    available_splits=("train", "validation", "test"),
    seq2seqLM=True,
    num_features=LIBRISPEECH_CONFIG["sample_rate"] * 16,
)

class CondensedLibrispeechASRDataset(Dataset):
    def __init__(
        self,
        split="train",
        tokenizer=None,
        max_token_len=None,
        num_workers=0,
        load_from_cache_file=True,
        auto_setup=False,
        dataset_path=Path("./preprocessed_data"),
        config=LIBRISPEECH_CONFIG
    ):
        super().__init__()
        self.split = split
        self.dataset_path = dataset_path
        self.config = config
        self.X = None
        self.Y = None
        
        # Store the additional parameters that the framework passes
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.num_workers = num_workers
        self.load_from_cache_file = load_from_cache_file
        
        # Automatically call setup if requested
        if auto_setup:
            self.setup()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def prepare_data(self) -> None:
        # Map the standard split name to a Librispeech-specific split for dataset loading
        librispeech_split = "train.clean.100"  # default
        
        if self.split == "train":
            librispeech_split = "train.clean.100"  # or other training split as needed
        elif self.split == "validation":
            librispeech_split = "validation.clean"
        elif self.split == "test":
            librispeech_split = "test.clean"
            
        _preprocess_librispeech_dataset(self.dataset_path, self.config, split=librispeech_split)

    def setup(self) -> None:
        # Map the standard split name to the corresponding file paths
        if self.split == "train":
            x_path, y_path = "X_train.pt", "Y_train.pt"
        elif self.split == "validation":
            x_path, y_path = "X_val.pt", "Y_val.pt"
        elif self.split == "test":
            x_path, y_path = "X_test.pt", "Y_test.pt"
        else:
            raise ValueError(f"Split {self.split} is not supported.")
            
        assert (self.dataset_path / x_path).exists(), "Dataset is missing, run prepare_data() first."

        self.X = torch.load(self.dataset_path / x_path)
        self.Y = torch.load(self.dataset_path / y_path)


def _preprocess_librispeech_dataset(save_path: Path, config: dict = LIBRISPEECH_CONFIG, split="validation.clean"):
    dataset = load_dataset("nyalpatel/condensed_librispeech_asr", split=split)

    input_values, labels = [], []

    for example in dataset:
        waveform = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]
        text = example["text"]

        # Convert waveform to tensor if it's not already
        if isinstance(waveform, list):
            waveform = torch.tensor(waveform, dtype=torch.float32)

        if sampling_rate != config["sample_rate"]:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=config["sample_rate"]
            )(waveform)

        if config["normalize_waveform"]:
            # Check if waveform has enough elements to calculate std
            if waveform.numel() > 0 and waveform.std() > 0:
                waveform = (waveform - waveform.mean()) / waveform.std()
            else:
                # Handle edge case with zero std
                waveform = waveform - waveform.mean() if waveform.numel() > 0 else waveform

        with processor.as_target_processor():
            label = processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        max_length = config["max_audio_length"]
        if waveform.shape[0] > max_length:
            waveform = waveform[:max_length]
        else:
            pad = torch.zeros(max_length)
            pad[: waveform.shape[0]] = waveform
            waveform = pad

        input_values.append(waveform)
        labels.append(label)

    input_values = torch.stack(input_values)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        input_values, labels, test_size=config["test_size"] + config["validation_size"], random_state=42
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp, Y_temp, test_size=config["test_size"] / (config["test_size"] + config["validation_size"]), random_state=42
    )

    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
    
    torch.save(X_train, save_path / "X_train.pt")
    torch.save(Y_train, save_path / "Y_train.pt")
    torch.save(X_val, save_path / "X_val.pt")
    torch.save(Y_val, save_path / "Y_val.pt")
    torch.save(X_test, save_path / "X_test.pt")
    torch.save(Y_test, save_path / "Y_test.pt")

    print("✅ Condensed Librispeech dataset preprocessed and saved!")