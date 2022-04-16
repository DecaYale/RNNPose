
from torch.utils.data import Dataset

from builder import dataset_builder


class DatasetWrapper(Dataset):
    """ convert our dataset to Dataset class in pytorch.
    """

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]

    @property
    def dataset(self):
        return self._dataset


def build(input_reader_config,
          training,
          ) -> DatasetWrapper:

    dataset = dataset_builder.build(
        input_reader_config,
        training,
    )
    dataset = DatasetWrapper(dataset)
    return dataset
