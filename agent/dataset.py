"""Assistant-only supervised fine-tuning dataset for the JSON agent."""

import re
import torch

from agent.tokenizer import AgentTokenizer, DEFAULT_AGENT_TOKENIZER
from training.dataset import collate_pad


_ASSISTANT_BLOCK = re.compile(r"<assistant>.*?</assistant>", re.DOTALL)


def encode_agent_training_example(text: str, tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER,
                                  max_len: int = 768) -> tuple[list[int], list[float]]:
    """Encode one trace and mark only assistant tokens as supervised.

    Tool results and user messages are context the model reads at inference;
    predicting them during training encourages hallucinated observations.

    An EOT byte immediately following an assistant block is supervised as part
    of that block. In a multi-turn conversation this is the signal that teaches
    the model to stop and yield to the user instead of continuing to write the
    user's next message itself. Supervising only the trailing EOT (the previous
    behaviour) teaches end-of-episode, never end-of-turn.
    """
    token_ids = []
    target_mask = []
    position = 0
    for match in _ASSISTANT_BLOCK.finditer(text):
        context = text[position:match.start()]
        context_tokens = tokenizer.encode(context)
        token_ids.extend(context_tokens)
        target_mask.extend([0.0] * len(context_tokens))

        assistant = match.group(0)
        end = match.end()
        # Absorb an EOT that directly closes this assistant turn.
        if text.startswith("\x03", end):
            assistant += "\x03"
            end += 1
        assistant_tokens = tokenizer.encode(assistant)
        token_ids.extend(assistant_tokens)
        target_mask.extend([1.0] * len(assistant_tokens))
        position = end

    suffix = text[position:]
    suffix_tokens = tokenizer.encode(suffix)
    token_ids.extend(suffix_tokens)
    target_mask.extend([0.0] * len(suffix_tokens))
    if text.endswith("\x03") and target_mask:
        target_mask[-1] = 1.0

    token_ids = token_ids[:max_len]
    target_mask = target_mask[:max_len]
    return token_ids, target_mask


class AgentDataset(torch.utils.data.Dataset):
    """Tagged JSON traces with labels masked outside assistant messages."""

    collate_pad = staticmethod(collate_pad)

    def __init__(self, traces: list[str], max_len: int = 768,
                 tokenizer: AgentTokenizer = DEFAULT_AGENT_TOKENIZER):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.samples = []
        for trace in traces:
            tokens, target_mask = encode_agent_training_example(trace, tokenizer, max_len)
            if len(tokens) < 2 or sum(target_mask[1:]) == 0:
                continue
            self.samples.append((tokens, target_mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        tokens, target_mask = self.samples[index]
        return (
            torch.tensor(tokens[:-1], dtype=torch.long),
            torch.tensor(tokens[1:], dtype=torch.long),
            torch.tensor(target_mask[1:], dtype=torch.float32),
        )
