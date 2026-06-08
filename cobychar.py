import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

from cobygrad import Tensor
from cobynn import Cobyformer
from cobyoptim import Adam, AdamW, CosineWithLinearWarmup, Scheduler
from cobytok import Tokenizer
from cobyutils import CrossEntropyLoss

vocab_dict = {
    "\n": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "J": 10,
    "K": 11,
    "L": 12,
    "M": 13,
    "N": 14,
    "O": 15,
    "P": 16,
    "Q": 17,
    "R": 18,
    "S": 19,
    "T": 20,
    "U": 21,
    "V": 22,
    "W": 23,
    "X": 24,
    "Y": 25,
    "Z": 26,
    "a": 27,
    "b": 28,
    "c": 29,
    "d": 30,
    "e": 31,
    "f": 32,
    "g": 33,
    "h": 34,
    "i": 35,
    "j": 36,
    "k": 37,
    "l": 38,
    "m": 39,
    "n": 40,
    "o": 41,
    "p": 42,
    "q": 43,
    "r": 44,
    "s": 45,
    "t": 46,
    "u": 47,
    "v": 48,
    "w": 49,
    "x": 50,
    "y": 51,
    "z": 52,
    "0": 53,
    "1": 54,
    "2": 55,
    "3": 56,
    "4": 57,
    "5": 58,
    "6": 59,
    "7": 60,
    "8": 61,
    "9": 62,
    ".": 63,
    ",": 64,
    "!": 65,
    "?": 66,
    ";": 67,
    ":": 68,
    '"': 69,
    "'": 70,
    "-": 71,
    "(": 72,
    ")": 73,
    "[": 74,
    "]": 75,
    "{": 76,
    "}": 77,
    "/": 78,
    "\\": 79,
    "@": 80,
    "#": 81,
    "$": 82,
    "%": 83,
    "&": 84,
    "*": 85,
    "+": 86,
    "=": 87,
    "_": 88,
    " ": 89,
    "<|unk|>": 90,
    "<|eos|>": 91,
    "<|pad|>": 92,
}

dataset = load_dataset("roneneldan/TinyStories")
train_data = Dataset.from_dict(dataset["train"][:16_000])
tokenizer = Tokenizer.from_vocab(vocab_dict)


def tokenize_batch(batch):
    return {
        "tokens": [
            tokenizer.pad(tokenizer.tokenize(text), 128, True) for text in batch["text"]
        ]
    }


train_tokenized = train_data.map(
    tokenize_batch, batched=True, batch_size=128, remove_columns=["text"], num_proc=16
)

model = Cobyformer(32, 128, 4, len(vocab_dict), 4)
criterion = CrossEntropyLoss()
decayed_params = [p for p in model.parameters() if p.type == "weight"]
undecayed_params = [p for p in model.parameters() if p.type != "weight"]
optimizer = AdamW(
    [{"params": decayed_params}, {"params": undecayed_params, "decay": 0.00}]
)
scheduler = CosineWithLinearWarmup(optimizer, 16_000 / 64)

pbar = tqdm(train_tokenized.iter(batch_size=128), total=125)
for batch in pbar:
    token_batch = np.array(batch["tokens"])
    one_hot = np.eye(len(vocab_dict))[token_batch]
    prediction = model(one_hot)
    loss = criterion(prediction[..., :-1, :], token_batch[..., 1:])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    pbar.set_postfix_str(f"loss: {loss.data / 128 / 127}")
