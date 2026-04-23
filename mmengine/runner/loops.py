class BaseLoop:
    def __init__(self, runner, dataloader, **kwargs):
        self.runner = runner
        self.dataloader = dataloader


class IterBasedTrainLoop(BaseLoop):
    def __init__(self, runner, dataloader, max_iters: int,
                 val_begin: int = 1, val_interval: int = 1000, **kwargs):
        super().__init__(runner, dataloader)
        self.max_iters = max_iters
        self.val_begin = val_begin
        self.val_interval = val_interval


class EpochBasedTrainLoop(BaseLoop):
    def __init__(self, runner, dataloader, max_epochs: int,
                 val_begin: int = 1, val_interval: int = 1, **kwargs):
        super().__init__(runner, dataloader)
        self.max_epochs = max_epochs
        self.val_begin = val_begin
        self.val_interval = val_interval


class ValLoop(BaseLoop):
    def __init__(self, runner, dataloader, evaluator, **kwargs):
        super().__init__(runner, dataloader)
        self.evaluator = evaluator


class TestLoop(BaseLoop):
    def __init__(self, runner, dataloader, evaluator, **kwargs):
        super().__init__(runner, dataloader)
        self.evaluator = evaluator


__all__ = ['IterBasedTrainLoop', 'EpochBasedTrainLoop', 'ValLoop', 'TestLoop']
