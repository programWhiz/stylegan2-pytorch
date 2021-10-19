import math
from collections import namedtuple
from io import BytesIO
import re
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
        # Suppose we have items with (count, len):
        # [ (4, 5), (2, 7), (3, 8) ]
        # this gives a "count to length" ratio for items:
        # [ 5/4, 7/2, 8/3 ]
        # Rounding up for each: [ 2, 4, 3 ]
        # Getting max, we need 4 iterations of the pattern to get through the entire
        # 2nd dataset: 4 * len(pattern) == 4 * (4 + 2 + 3) == 4 * 9 == 36
        if self.len_ is None:
            cts_lens = [ (item.count, len(item.dataset)) for item in self.items ]
            ratios = [ int(math.ceil(ln / ct)) for ln, ct in cts_lens ]
            max_len = max(ln for ct, ln in cts_lens)
            self.len_ = max(ratios) * self.pattern_size * max_len
        return self.len_

    def parse_paths_pattern(self, pattern, build_dataset):
        Item = namedtuple('Item', ['path', 'count', 'cut', 'dataset'])
        parts = pattern.split(',')
        pairs = []
        cut = 0
        for part in parts:
            match = re.match(r'(.*):(\d+)$', part)
            path = match.group(1)
            count = int(match.group(2))
            if count < 1:
                raise Exception(f'Count must be positive: {count}')

            cut += count
            dataset = build_dataset(path)
            pairs.append(Item(path=path, count=count, cut=cut, dataset=dataset))

        return pairs

    def __getitem__(self, idx):
        return self.get_sub_sample_index(idx)

    def get_sub_sample_index(self, x : int):
        # figure out which element of the pattern we are in
        pat_idx = x % self.pattern_size
        # Get part of the pattern that contains the subsegment
        item_idx, item = next((i, item) for i, item in enumerate(self.items) if pat_idx < item.cut)
        # Which index is it inside the pattern?
        # Example: a=3, b=2, c=1  total=6
        # if x=36, then 39%6 = 3, so we are in part b
        # We know that we have 38 // 5 = 7 cycles through the pattern,
        # And we have remainder: (3 - b.cut) % b.size = (3 - 3 == 0) % b.size = 0
        # so we are |b| * cycles deep into b: sample_idx = 2 * 7 + 0 = 14
        pattern_cycles = x // self.pattern_size
        base_idx = pattern_cycles * item.count
        # How deep into this one sub-pattern are we?
        prev_cut = self.items[item_idx - 1].cut if item_idx > 0 else 0
        pat_sub_idx = (pat_idx - prev_cut) % item.count
        item_sample_idx = (base_idx + pat_sub_idx) % len(item)
        return item.dataset[item_sample_idx]


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
