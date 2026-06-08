class Tokenizer:
    def __init__(self):
        self.vocab = {}  # data -> index
        self.reverse_vocab = {}  # index -> data
        self.vocab_size = 0
        self.max_len = 0
        self.UNK = "<|unk|>"
        self.EOS = "<|eos|>"
        self.PAD = "<|pad|>"

    @classmethod
    def from_vocab(cls, vocab):
        tokenizer = cls()
        assert tokenizer.UNK in vocab.keys()
        assert tokenizer.EOS in vocab.keys()
        assert tokenizer.PAD in vocab.keys()
        tokenizer.vocab = vocab
        tokenizer.reverse_vocab = {v: k for k, v in vocab.items()}
        tokenizer.vocab_size = len(vocab)
        tokenizer.max_len = max(len(t) for t in vocab.keys())
        return tokenizer

    def tokenize(self, s):
        token_ids = []
        while s:
            for i in range(min(self.max_len, len(s)), 0, -1):
                if s[:i] in self.vocab:
                    token_ids.append(self.vocab[s[:i]])
                    s = s[i:]
                    break
            else:
                token_ids.append(self.vocab[self.UNK])
                s = s[1:]
        token_ids.append(self.vocab[self.EOS])
        return token_ids

    def decode(self, token_ids):
        return "".join(self.reverse_vocab[token] for token in token_ids)

    def pad(self, token_ids, max: int, truncate=False):
        if len(token_ids) >= max:
            if truncate:
                return token_ids[-max:]
            else:
                return token_ids
        return [self.vocab[self.PAD]] * (max - len(token_ids)) + token_ids
