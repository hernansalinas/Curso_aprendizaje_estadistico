def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE))
            == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf'
                                    )).masked_fill(mask == 1,
            float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),
                           device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)


def train_epoch(model, train_iter, optimizer):
    model.train()
    losses = 0
    for (idx, (src, tgt)) in enumerate(train_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask) = \
            create_mask(src, tgt_input)

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
            )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                       tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
    return losses / len(train_iter)


def evaluate(model, valid_iter):
    model.eval()
    losses = 0
    for (idx, (src, tgt)) in enumerate(valid_iter):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        (src_mask, tgt_mask, src_padding_mask, tgt_padding_mask) = \
            create_mask(src, tgt_input)

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
            )
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]),
                       tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)

#!/usr/bin/python
# -*- coding: utf-8 -*-


def greedy_decode(
    model,
    src,
    src_mask,
    max_len,
    start_symbol,
    ):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1,
                    1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0],
                                  memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = \
            generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        (_, next_word) = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1,
                       1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


def translate(
    model,
    src,
    src_vocab,
    tgt_vocab,
    src_tokenizer,
    ):
    model.eval()

    tokens = [BOS_IDX]
    for tok in src_tokenizer(src):
      try:
        stoi = src_vocab.get_stoi()[tok]
      except: 
        stoi = src_vocab['<unk>']
      tokens.append(stoi)
    tokens += [EOS_IDX]
    num_tokens = len(tokens)
    src = torch.LongTensor(tokens).reshape(num_tokens, 1)
    src_mask = torch.zeros(num_tokens, num_tokens).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens
                               + 5, start_symbol=BOS_IDX).flatten()
    return ' '.join([tgt_vocab.get_itos()[tok] for tok in
                    tgt_tokens]).replace('<bos>', '').replace('<eos>',
            '')