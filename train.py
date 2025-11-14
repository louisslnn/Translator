import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

from transformer_model import build_transformer
from dataset import BilingualDataset, causal_mask
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import warnings
from pathlib import Path
from tqdm import tqdm
import math
import os

# -------------------------
# greedy_decode (unchanged but kept safe)
# -------------------------
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


# -------------------------
# run_validation: run only occasionally / or at epoch end
# -------------------------
def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device,
                   print_msg, global_state, writer, num_examples=2):
    model.eval()
    count = 0
    source_texts, expected, predicted = [], [], []
    console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)

            # ensure batch size is 1 for greedy decode
            if encoder_input.size(0) != 1:
                encoder_input = encoder_input[:1]
                encoder_mask = encoder_mask[:1]

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch.get('src_text', [""])[0]
            target_text = batch.get('tgt_text', [""])[0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            tqdm.write('-' * console_width)
            tqdm.write(f'SOURCE: {source_text}')
            tqdm.write(f'EXPECTED: {target_text}')
            tqdm.write(f'PREDICTED: {model_out_text}')

            if count == num_examples:
                break

    model.train()
    return source_texts, expected, predicted


# -------------------------
# token helpers
# -------------------------
def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]


def build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


# -------------------------
# dataset + dataloaders
# -------------------------
def get_ds(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

    # split
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                src_lang=config['lang_src'], tgt_lang=config['lang_tgt'],
                                seq_len=config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              src_lang=config['lang_src'], tgt_lang=config['lang_tgt'],
                              seq_len=config['seq_len'])

    # quick max length print (optional)
    max_len_src, max_len_tgt = 0, 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}.")
    print(f"Max length of target sentence: {max_len_tgt}.")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


# -------------------------
# training
# -------------------------
def train_model(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device {device}.")

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    writer = SummaryWriter(config['experiment_name'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config.get('preload'):
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # USE target tokenizer PAD for ignore_index
    pad_id = tokenizer_tgt.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1).to(device)

    save_every = config.get('save_every', 1)
    validate_every_steps = config.get('validate_every_steps', 500)  # validate every N steps

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        running_loss = 0.0
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # ensure dims
            if encoder_input.dim() == 1:
                encoder_input = encoder_input.unsqueeze(0)
            if decoder_input.dim() == 1:
                decoder_input = decoder_input.unsqueeze(0)

            # forward
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            # IMPORTANT: label key is 'labels' (not 'label')
            label = batch.get('labels') or batch.get('label')
            if label is None:
                raise KeyError("Batch must contain 'labels' (target tokens). Check BilingualDataset.")
            label = label.to(device)

            # compute loss
            loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
            running_loss += loss.item()

            # backward / step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            writer.add_scalar('train/loss', loss.item(), global_step)
            global_step += 1

            # occasional validation
            if global_step % validate_every_steps == 0:
                run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                               tqdm.write, global_step, writer, num_examples=2)

            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{running_loss / (global_step+1):.4f}"})

        # epoch end validation
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       tqdm.write, global_step, writer, num_examples=2)

        # save every epoch (or according to save_every)
        if (epoch + 1) % save_every == 0:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
            print(f"Saved checkpoint: {model_filename}")

    writer.close()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
