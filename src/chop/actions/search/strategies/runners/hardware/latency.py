import time
import torch
from .base import HWRunnerBase


class LatencyRunner(HWRunnerBase):
    available_metrics = ("latency",)

    def _post_init_setup(self) -> None:
        self.num_batches = self.config.get("num_batches", 50) if self.config else 50
        self.num_warmup = self.config.get("num_warmup_batches", 10) if self.config else 10

    def _forward(self, batch, model):
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            if isinstance(x, torch.Tensor):
                x = x.to(self.accelerator)
            return model(x)
        elif isinstance(batch, dict):
            batch = {
                k: v.to(self.accelerator)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            return model(**batch)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def __call__(self, data_module, model, sampled_config) -> dict[str, float]:
        # Unwrap MaseGraph if needed
        if not isinstance(model, torch.nn.Module):
            forward_model = model.model
        else:
            forward_model = model

        forward_model.to(self.accelerator)
        forward_model.eval()

        data_loader = data_module.val_dataloader()
        use_cuda = isinstance(self.accelerator, str) and "cuda" in self.accelerator or (
            hasattr(self.accelerator, "type") and "cuda" in self.accelerator.type
        )

        with torch.no_grad():
            # Warmup
            for i, batch in enumerate(data_loader):
                if i >= self.num_warmup:
                    break
                self._forward(batch, forward_model)

            # Timed runs
            if use_cuda:
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                t0 = time.perf_counter()

            for i, batch in enumerate(data_loader):
                if i >= self.num_batches:
                    break
                self._forward(batch, forward_model)

            if use_cuda:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {"latency": elapsed_ms / self.num_batches}
