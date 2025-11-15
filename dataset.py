import torch
import torch.nn as nn 
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_src.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncate if too long (leave room for SOS, EOS, and at least some padding)
        max_src_len = self.seq_len - 2  # -2 for SOS and EOS
        max_tgt_len = self.seq_len - 1  # -1 for SOS (EOS goes in label)
        
        if len(enc_input_tokens) > max_src_len:
            enc_input_tokens = enc_input_tokens[:max_src_len]
        if len(dec_input_tokens) > max_tgt_len:
            dec_input_tokens = dec_input_tokens[:max_tgt_len]

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # This should never happen now, but keep as safety check
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            # If still too long after truncation, truncate more aggressively
            enc_input_tokens = enc_input_tokens[:max_src_len]
            dec_input_tokens = dec_input_tokens[:max_tgt_len]
            enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
            dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add SOS to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # Add EOS to the label (what we expect as output from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token.item()] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        encoder_mask = (encoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0)
        decoder_mask = (decoder_input != self.pad_token.item()).unsqueeze(0).unsqueeze(0)

        return {
            "encoder_input": encoder_input, # (seq_len)
            "decoder_input": decoder_input, # (seq_len)
            "encoder_mask": encoder_mask.bool(), # (1, 1, seq_len)
            "decoder_mask": decoder_mask.bool() & causal_mask(decoder_input.size(0)), # (1, seq_len) & (seq_len, seq_len)
            "labels": label, # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size, dtype=torch.bool), diagonal=1)
    return ~mask
