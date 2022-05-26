import torch
import io
import torchtext
from collections import Counter
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import tensor


class Dataset():
    def __init__(self, src_tokenizer, tgt_tokenizer):
        self.tgt_tokenizer = tgt_tokenizer
        self.src_tokenizer = src_tokenizer


    def build_vocab(self, filepath, tokenizer):
        counter = Counter()
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                counter.update(tokenizer(string_))
        return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    
    def set_vocab(self, filepaths):
        src_vocab = self.build_vocab(filepaths[0], self.src_tokenizer)
        src_vocab.set_default_index(src_vocab['<unk>'])
        tgt_vocab = self.build_vocab(filepaths[1], self.tgt_tokenizer)
        tgt_vocab.set_default_index(tgt_vocab['<unk>'])
        
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def data_process(self, filepaths):
        raw_src_iter = iter(io.open(filepaths[0], encoding="utf8"))
        raw_tgt_iter = iter(io.open(filepaths[1], encoding="utf8"))
        data = []
        for (raw_src, raw_tgt) in zip(raw_src_iter, raw_tgt_iter):
            src_tensor_ = torch.tensor([self.src_vocab[token] for token in self.src_tokenizer(raw_src.rstrip("n"))],
                                     dtype=torch.long)
            tgt_tensor_ = torch.tensor([self.tgt_vocab[token] for token in self.tgt_tokenizer(raw_tgt.rstrip("n"))],
                                     )
            data.append((src_tensor_, tgt_tensor_))
        return data

    def generate_batch(self, data_batch):
        PAD_IDX = self.src_vocab['<pad>']
        BOS_IDX = self.src_vocab['<bos>']
        EOS_IDX = self.src_vocab['<eos>']
        src_batch, tgt_batch = [], []
        for (src_item, tgt_item) in data_batch:
            src_batch.append(torch.cat([torch.tensor([BOS_IDX]), src_item, torch.tensor([EOS_IDX])], dim=0))
            tgt_batch.append(torch.cat([torch.tensor([BOS_IDX]), tgt_item, torch.tensor([EOS_IDX])], dim=0))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    def get_iter(self, filepaths, BATCH_SIZE):
        data = self.data_process(filepaths)
        data_iter = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=self.generate_batch)
        return data_iter