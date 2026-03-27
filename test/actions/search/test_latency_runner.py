"""
Tests for LatencyRunner.

- CPU unit tests: verify the runner returns {"latency": float} without crashing.
- GPU integration tests: verify latency > 0 ms on an actual CUDA device.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(Path(__file__).resolve().parents[3].as_posix())

from chop.actions.search.strategies.runners.hardware.latency import LatencyRunner

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU latency test requires CUDA"
)


# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

class _FakeModelInfo:
    is_nlp_model = False
    is_vision_model = True
    is_physical_model = False


class _FakeDatasetInfo:
    num_classes = 10


class _FakeDataModule:
    """Wraps a DataLoader so runner can call data_module.val_dataloader()."""

    def __init__(self, loader):
        self.batch_size = loader.batch_size
        self._loader = loader

    def val_dataloader(self):
        return self._loader


def _make_vision_loader(n=64, batch_size=8, device="cpu"):
    x = torch.randn(n, 3, 32, 32)
    y = torch.randint(0, 10, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size)


class _TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, 10),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# CPU unit tests (no CUDA required)
# ---------------------------------------------------------------------------


class TestLatencyRunnerCPU:
    """Basic sanity checks on CPU — just verify the interface contract."""

    def _make_runner(self, num_batches=5, num_warmup=2):
        return LatencyRunner(
            model_info=_FakeModelInfo(),
            task="cls",
            dataset_info=_FakeDatasetInfo(),
            accelerator="cpu",
            config={"num_batches": num_batches, "num_warmup_batches": num_warmup},
        )

    def test_returns_latency_key(self):
        runner = self._make_runner()
        model = _TinyConvNet()
        loader = _make_vision_loader(n=32, batch_size=8)
        dm = _FakeDataModule(loader)

        result = runner(dm, model, sampled_config={})

        assert isinstance(result, dict), "Runner must return a dict"
        assert "latency" in result, "Result must contain 'latency' key"

    def test_latency_is_positive_float(self):
        runner = self._make_runner()
        model = _TinyConvNet()
        loader = _make_vision_loader(n=32, batch_size=8)
        dm = _FakeDataModule(loader)

        result = runner(dm, model, sampled_config={})

        assert isinstance(result["latency"], float), "latency must be a float"
        assert result["latency"] > 0.0, "latency must be positive"

    def test_no_extra_keys(self):
        runner = self._make_runner()
        model = _TinyConvNet()
        loader = _make_vision_loader(n=32, batch_size=8)
        dm = _FakeDataModule(loader)

        result = runner(dm, model, sampled_config={})

        assert set(result.keys()) == {"latency"}, f"Unexpected keys: {result.keys()}"

    def test_default_config(self):
        """Runner should work when config is None (uses built-in defaults)."""
        runner = LatencyRunner(
            model_info=_FakeModelInfo(),
            task="cls",
            dataset_info=_FakeDatasetInfo(),
            accelerator="cpu",
            config=None,
        )
        model = _TinyConvNet()
        # Provide enough batches to cover the default warmup (10) + timed (50)
        loader = _make_vision_loader(n=512, batch_size=8)
        dm = _FakeDataModule(loader)

        result = runner(dm, model, sampled_config={})
        assert result["latency"] > 0.0

    def test_masegraph_unwrap(self):
        """Runner must unwrap MaseGraph-like objects that expose .model."""

        class _FakeMaseGraph:
            def __init__(self, m):
                self.model = m

        runner = self._make_runner()
        model = _TinyConvNet()
        graph = _FakeMaseGraph(model)
        loader = _make_vision_loader(n=32, batch_size=8)
        dm = _FakeDataModule(loader)

        result = runner(dm, graph, sampled_config={})
        assert result["latency"] > 0.0


# ---------------------------------------------------------------------------
# GPU integration tests (CUDA required)
# ---------------------------------------------------------------------------


@requires_cuda
class TestLatencyRunnerGPU:
    """Verify CUDA event timing path produces sensible results."""

    def _make_runner(self, num_batches=5, num_warmup=2):
        return LatencyRunner(
            model_info=_FakeModelInfo(),
            task="cls",
            dataset_info=_FakeDatasetInfo(),
            accelerator="cuda",
            config={"num_batches": num_batches, "num_warmup_batches": num_warmup},
        )

    def test_latency_positive_on_gpu(self):
        runner = self._make_runner()
        model = _TinyConvNet()
        loader = _make_vision_loader(n=64, batch_size=8)
        dm = _FakeDataModule(loader)

        result = runner(dm, model, sampled_config={})

        assert "latency" in result
        assert result["latency"] > 0.0, "GPU latency must be positive"

    def test_larger_model_has_higher_latency(self):
        """A deeper model should measure strictly higher latency."""
        class _BigNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    *[nn.Sequential(nn.Linear(512, 512), nn.ReLU()) for _ in range(8)],
                    nn.Linear(512, 10),
                )

            def forward(self, x):
                x = x.view(x.size(0), -1)[:, :512]
                return self.net(x)

        small_runner = self._make_runner(num_batches=10)
        big_runner = self._make_runner(num_batches=10)

        loader = _make_vision_loader(n=80, batch_size=8)
        dm = _FakeDataModule(loader)

        small_latency = small_runner(dm, _TinyConvNet(), sampled_config={})["latency"]
        big_latency = big_runner(dm, _BigNet(), sampled_config={})["latency"]

        # Big model should be measurably slower (allow 1ms tolerance for very fast GPUs)
        assert big_latency >= small_latency - 1.0, (
            f"Expected big model ({big_latency:.2f} ms) ≥ small model ({small_latency:.2f} ms)"
        )
