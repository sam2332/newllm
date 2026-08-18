"""Small sample dataset for training tests."""

import torch


SAMPLE_STORIES = [
    "grug hunt big mammoth and tribe eat well",
    "fire warm cave and keep wolf away",
    "river give fish and water to tribe",
    "star show path when night dark",
    "tool sharp rock cut meat good",
    "tribe dance under moon for rain",
    "elder tell story of great flood",
    "child find berry bush near cliff",
    "hunter track deer through tall grass",
    "cook stone hot make flat bread",
]


def char_tokenizer(text: str, max_vocab: int = 256):
    """Map each char to an int token."""
    tokens = [ord(c) for c in text]
    return [min(t, max_vocab - 1) for t in tokens]


def collate_pad(batch):
    """Pad batch to same length.

    Accepts either (x, y) or (x, y, mask) samples.  If a per-sample mask is
    provided it is padded; otherwise a ones mask is used.
    """
    has_mask = len(batch[0]) == 3
    max_len = max(len(x) for x, *_ in batch)
    xs = []
    ys = []
    masks = []
    for item in batch:
        x, y = item[0], item[1]
        mask = item[2] if has_mask else torch.ones(len(x))
        pad = max_len - len(x)
        xs.append(torch.nn.functional.pad(x, (0, pad)))
        ys.append(torch.nn.functional.pad(y, (0, pad), value=-100))
        masks.append(torch.nn.functional.pad(mask, (0, pad)))
    return torch.stack(xs), torch.stack(ys), torch.stack(masks)


class StoryDataset(torch.utils.data.Dataset):
    """Simple dataset. Returns (input_tokens, target_tokens)."""

    collate_pad = staticmethod(collate_pad)

    def __init__(self, stories, max_len: int = 40, max_vocab: int = 256):
        self.max_len = max_len
        self.max_vocab = max_vocab
        self.samples = []
        for story in stories:
            tokens = char_tokenizer(story, max_vocab)
            tokens = tokens[:max_len]
            if len(tokens) < 2:
                continue
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


