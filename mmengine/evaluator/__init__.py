from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence


class BaseMetric(metaclass=ABCMeta):
    """Base class for metrics."""

    default_prefix: Optional[str] = None

    def __init__(self, collect_device: str = 'cpu',
                 prefix: Optional[str] = None, **kwargs):
        self.collect_device = collect_device
        self.results: List = []
        self.prefix = prefix or self.default_prefix

    @abstractmethod
    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:
        pass

    @abstractmethod
    def compute_metrics(self, results: list) -> Dict[str, float]:
        pass

    def evaluate(self, size: int) -> Dict[str, float]:
        metrics = self.compute_metrics(self.results)
        self.results.clear()
        if self.prefix:
            metrics = {f'{self.prefix}/{k}': v for k, v in metrics.items()}
        return metrics

    def reset(self):
        self.results.clear()


class Evaluator:
    """Evaluator that wraps multiple metrics."""

    def __init__(self, metrics):
        if isinstance(metrics, BaseMetric):
            self.metrics = [metrics]
        elif isinstance(metrics, (list, tuple)):
            self.metrics = list(metrics)
        else:
            self.metrics = [metrics]

    def process(self, data_batch, data_samples):
        for metric in self.metrics:
            metric.process(data_batch, data_samples)

    def evaluate(self, size: int) -> Dict[str, float]:
        results = {}
        for metric in self.metrics:
            results.update(metric.evaluate(size))
        return results


__all__ = ['BaseMetric', 'Evaluator']
