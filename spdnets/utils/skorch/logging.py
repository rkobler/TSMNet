import logging
from skorch.callbacks.logging import EpochTimer, PrintLog

log = logging.getLogger(__name__)

class TrainLog(PrintLog):

    def __init__(self, prefix='') -> None:
        super().__init__()
        self.prefix = prefix

    def initialize(self):
        return self
    
    def on_epoch_end(self, net, **kwargs):
        r = net.history[-1]

        if r['epoch'] == 1 or r['epoch'] % 10 == 0:
            log.info(f"{self.prefix} {r['epoch']:3d} : trn={r['train_loss']:.3f}/{r['score_trn']:.2f} val={r['valid_loss']:.3f}/{r['score_val']:.2f}")


