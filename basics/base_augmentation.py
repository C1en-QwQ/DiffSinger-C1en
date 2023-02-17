class BaseAugmentation:
    """
    Base class for data augmentation.
    All methods of this class should be thread-safe.
    1. *process_item*:
        Apply augmentation to one piece of data.
    """
    def __init__(self, data_dirs: list, augmentation_args: dict):
        self.raw_data_dirs = data_dirs
        self.augmentation_args = augmentation_args

    def process_item(self, item: dict, **kwargs) -> dict:
        raise NotImplementedError()
