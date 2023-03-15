"""
File containing all the callbacks used in the project
"""
from pytorch_lightning.callbacks import Callback
from carbontracker.tracker import CarbonTracker
import os


def get_callbacks(args):
    """
    Returns a list of callbacks
    """
    callbacks = []
    if not args.disable_carbon_tracker:
        callbacks.append(CarbonTrackerCallback(args))
    return callbacks


class CarbonTrackerCallback(Callback):
    def __init__(self, args):
        """
        Carbontracker callback as described in https://arxiv.org/abs/2007.03051
        """
        super().__init__()
        if args.max_epochs is not None:
            self.max_epochs = args.max_epochs
        else:
            self.max_epochs = 1000
        if args.check_val_every_n_epoch is not None:
            self.val_every_n_epochs = args.check_val_every_n_epoch
        else:
            self.val_every_n_epochs = 1

        self.train_tracker = CarbonTracker(epochs=self.max_epochs, epochs_before_pred=-1, monitor_epochs=1,
                                           log_dir=os.path.join(os.getcwd(), "lightning_logs", "carbontracker"),
                                           log_file_prefix="train_{}".format(args.run_name),
                                           verbose=0)
        self.val_tracker = CarbonTracker(epochs=self.max_epochs // self.val_every_n_epochs,
                                         epochs_before_pred=-1, monitor_epochs=1,
                                         log_dir=os.path.join(os.getcwd(), "lightning_logs", "carbontracker"),
                                         log_file_prefix="val_{}".format(args.run_name),
                                         verbose=0)
    
    def on_train_epoch_start(self, *args, **kwargs):
        self.train_tracker.epoch_start()
    
    def on_train_epoch_end(self, *args, **kwargs):
        self.train_tracker.epoch_end()

    def on_validation_epoch_start(self, *args, **kwargs):
        self.val_tracker.epoch_start()

    def on_validation_epoch_end(self, *args, **kwargs):
        self.val_tracker.epoch_end()
