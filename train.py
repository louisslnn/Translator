from __future__ import annotations

import argparse
import random
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from tqdm import tqdm

from config import get_config, get_weights_file_path
from dataset import BilingualDataset, causal_mask
from transformer_model import build_transformer

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

    with torch.inference_mode():
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

            print_msg('-' * console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'EXPECTED: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')

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
def get_ds(config, pin_memory: bool, seed: Optional[int] = None):
    ds_raw = load_dataset(
        'opus_books',
        f"{config['lang_src']}-{config['lang_tgt']}",
        split='train',
        cache_dir=config.get('dataset_cache_dir')
    )

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

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    num_workers = int(config.get('num_workers', 0))
    persistent_workers = bool(config.get('persistent_workers', False) and num_workers > 0)
    prefetch_factor = config.get('prefetch_factor', None)
    loader_common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if not persistent_workers:
        loader_common.pop("persistent_workers", None)
    if num_workers > 0 and prefetch_factor:
        loader_common["prefetch_factor"] = int(prefetch_factor)

    def _seed_worker(worker_id: int):
        worker_seed = (seed if seed is not None else torch.initial_seed()) % 2**32
        np.random.seed(worker_seed + worker_id)
        random.seed(worker_seed + worker_id)

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=config.get('drop_last', True),
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator if seed is not None else None,
        **loader_common,
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        worker_init_fn=_seed_worker if seed is not None else None,
        generator=generator if seed is not None else None,
        **loader_common,
    )

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    use_checkpoint = config.get('use_gradient_checkpointing', True)
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'], use_checkpoint=use_checkpoint)
    return model


# -------------------------
# training
# -------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: Optional[str]) -> torch.device:
    if device_name:
        requested_device = torch.device(device_name)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise EnvironmentError("CUDA device requested but CUDA is not available.")
        return requested_device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_compile_model(model: nn.Module, config: Dict[str, any]) -> nn.Module:
    if not config.get('enable_compile', False):
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requested but this version of PyTorch does not support it.")

    compile_kwargs = {
        "fullgraph": bool(config.get("compile_fullgraph", False)),
    }
    return torch.compile(model, **compile_kwargs)


def get_autocast_dtype(precision: str) -> torch.dtype:
    precision = (precision or "none").lower()
    if precision in {"fp16", "float16", "half"}:
        return torch.float16
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp32", "float32", "none"}:
        return torch.float32
    raise ValueError(f"Unsupported mixed precision value: {precision}")


def train_model(config):
    device = resolve_device(config.get("device"))
    print(f"Using device {device}.")
    
    # Set PyTorch CUDA memory allocation config to reduce fragmentation
    if device.type == "cuda":
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Clear any existing cache
        torch.cuda.empty_cache()

    set_seed(int(config.get("seed", 42)))

    if device.type == "cuda":
        # Extract device index if specified, otherwise use 0
        device_index = device.index if device.index is not None else 0
        torch.cuda.set_device(device_index)
        if config.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    Path(config['experiment_name']).mkdir(parents=True, exist_ok=True)

    pin_memory = bool(config.get('pin_memory', device.type == 'cuda'))
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config,
        pin_memory=pin_memory,
        seed=int(config.get("seed", 42)),
    )

    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model = maybe_compile_model(model, config)

    writer = SummaryWriter(config['experiment_name'])
    # Ensure lr is a float (safety check)
    lr = float(config['lr']) if not isinstance(config['lr'], (int, float)) else float(config['lr'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-9)

    initial_epoch = 0
    global_step = 0
    if config.get('preload'):
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename, map_location="cpu")
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    # USE target tokenizer PAD for ignore_index
    pad_id = tokenizer_tgt.token_to_id('[PAD]')
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=0.1).to(device)

    save_every = int(config.get('save_every', 1))
    validate_every_steps = int(config.get('validate_every_steps', 500))
    validation_examples = int(config.get('validation_examples', 2))

    # Ensure mixed_precision is a string (safety check)
    precision = str(config.get("mixed_precision", "none")).lower()
    autocast_dtype = get_autocast_dtype(precision)
    use_autocast = device.type == "cuda" and autocast_dtype != torch.float32
    scaler = GradScaler(enabled=use_autocast and autocast_dtype == torch.float16)

    grad_accum_steps = max(1, int(config.get("grad_accumulation_steps", 1)))

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")

        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        # Clear CUDA cache at the start of each epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
        for step_idx, batch in enumerate(batch_iterator, start=1):
            encoder_input = batch['encoder_input'].to(device)  # (batch, seq_len)
            decoder_input = batch['decoder_input'].to(device)  # (batch, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            # ensure dims
            if encoder_input.dim() == 1:
                encoder_input = encoder_input.unsqueeze(0)
            if decoder_input.dim() == 1:
                decoder_input = decoder_input.unsqueeze(0)

            # IMPORTANT: label key is 'labels' (not 'label')
            if 'labels' in batch:
                label = batch['labels']
            elif 'label' in batch:
                label = batch['label']
            else:
                raise KeyError("Batch must contain 'labels' (target tokens). Check BilingualDataset.")
            label = label.to(device)

            autocast_context = autocast(dtype=autocast_dtype) if use_autocast else nullcontext()

            with autocast_context:
                encoder_output = model.encode(encoder_input, encoder_mask)
                decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
                proj_output = model.project(decoder_output)
                loss = loss_fn(proj_output.view(-1, proj_output.size(-1)), label.view(-1))
                loss_for_backprop = loss / grad_accum_steps
                
                # Store loss value before deleting tensors
                loss_value = loss.item()
                
                # Explicitly delete intermediate tensors to free memory
                del encoder_output, decoder_output, proj_output

            running_loss += loss_value

            if scaler.is_enabled():
                scaler.scale(loss_for_backprop).backward()
            else:
                loss_for_backprop.backward()
            
            # Delete loss tensors after backward pass (they're no longer needed)
            del loss, loss_for_backprop

            if step_idx % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                writer.add_scalar('train/loss', loss_value, global_step)
                
                # Periodically clear CUDA cache to prevent fragmentation
                if device.type == "cuda":
                    if step_idx % 50 == 0:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()

                # occasional validation
                if global_step % validate_every_steps == 0:
                    run_validation(
                        model,
                        val_dataloader,
                        tokenizer_src,
                        tokenizer_tgt,
                        config['seq_len'],
                        device,
                        tqdm.write,
                        global_step,
                        writer,
                        num_examples=validation_examples,
                    )

            if step_idx % int(config.get("log_every_steps", 50)) == 0:
                avg_loss = running_loss / step_idx
                batch_iterator.set_postfix({"loss": f"{loss_value:.4f}", "avg_loss": f"{avg_loss:.4f}"})

        # epoch end validation
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device,
                       tqdm.write, global_step, writer, num_examples=validation_examples)

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
    parser = argparse.ArgumentParser(description="Train the translation model.")
    parser.add_argument("--config", type=str, default=None, help="Path to a YAML configuration file.")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--mixed_precision", type=str, default=None, help="Mixed precision mode: fp32 | fp16 | bf16.")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader worker override.")
    parser.add_argument("--grad_accumulation_steps", type=int, default=None, help="Gradient accumulation steps override.")
    parser.add_argument("--enable_compile", action="store_true", help="Enable torch.compile() on supported PyTorch versions.")
    parser.add_argument("--experiment_name", type=str, default=None, help="TensorBoard log directory override.")
    parser.add_argument("--preload", type=str, default=None, help="Checkpoint file name (without path) to preload.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    args = parser.parse_args()

    overrides: Dict[str, Optional[str]] = {}
    for key in ("device", "mixed_precision", "experiment_name", "preload"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    for key in ("num_workers", "grad_accumulation_steps", "seed"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = int(value)

    if args.enable_compile:
        overrides["enable_compile"] = True

    config = get_config(config_path=args.config, overrides=overrides if overrides else None)
    train_model(config)
