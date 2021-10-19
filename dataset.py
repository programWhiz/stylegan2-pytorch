from collections import namedtuple
from io import BytesIO
import re
import random
import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiDirDataset(Dataset):
    def __init__(self, paths_pattern, build_dataset):
        # Get pairs of [(path, count)]
        self.items = self.parse_paths_pattern(paths_pattern, build_dataset)
        # Total size of pattern, for example if we have [3, 1, 1] pattern size is 5
        self.pattern_size = sum(x.count for x in self.items)
        self.len_ = None

        print("Initialized multi-directory dataset with counts:")
        for item in self.items:
            print(f"{item.path}: freq={item.count}, len={len(item.dataset)}")
        print(f"Total Pattern Count: {self.pattern_size}")
        print(f"Total Len: {len(self)}\n")

    def __len__(self):
        if self.len_ is None:
            self.len_ = sum(len(item.dataset) for item in self.items)
        return self.len_

    class Item:
        def __init__(self, path, count, prob, dataset, cut):
            self.path = path
            self.count = count
            self.prob = prob
            self.cut = cut
            self.dataset = dataset

    def parse_paths_pattern(self, pattern, build_dataset):
        Item = MultiDirDataset.Item
        parts = pattern.split(',')
        items = []

        for part in parts:
            match = re.match(r'(.*):(\d+)$', part)
            path = match.group(1)
            count = int(match.group(2))
            if count < 1:
                raise Exception(f'Count must be positive: {count}')

            dataset = build_dataset(path)
            items.append(Item(path=path, count=count, prob=0, cut=0.0, dataset=dataset))

        total = sum(x.count for x in items)
        cut = 0.0
        for item in items:
            item.prob = item.count / total
            cut += item.prob
            item.cut = cut

        # make sure the cut thresholds sum to 1.0
        items[-1].cut = 1.0

        return items

    def __getitem__(self, idx):
        p = random.random()
        item = next((item for item in self.items if p <= item.cut), self.items[-1])
        return item.dataset[idx % len(item.dataset)]


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
