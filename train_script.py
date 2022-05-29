from transformer.optimization import TransformerOptimizer
from transformer.training import TransformerTrainer
from transformer.data import TranslationDataset
from transformer.vocabulary import VocabBuilder
from transformer.tokenization import Tokenizer
from transformer.models import Transformer
from transformer.conf import parse_cfg

from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader

import torch.optim as optim
import torch.nn as nn
import numpy as np

import random
import torch
import os

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    seed_everything()
    cfg, wandb_log = parse_cfg()
    
    train_data, valid_data, test_data = Multi30k(language_pair=(cfg.src_lang, cfg.trg_lang))
    
    train_src = [src for src, _ in train_data][:-1] # last one is empty
    train_trg = [trg for _, trg in train_data][:-1]
    
    valid_src = [src for src, _ in valid_data][:-1] # last one is empty
    valid_trg = [trg for _, trg in valid_data][:-1]
    
    test_src = [src for src, _ in test_data][:-1] # last one is empty
    test_trg = [trg for _, trg in test_data][:-1]
    
    src_vocab_builder = VocabBuilder(cfg.src_lang, max_vocab_size=cfg.max_vocab_size)
    trg_vocab_builder = VocabBuilder(cfg.trg_lang, max_vocab_size=cfg.max_vocab_size)
    
    src_vocab = src_vocab_builder.build(
        train_src, 
        min_freq=cfg.min_freq, 
        lowercase=cfg.lowercase, 
        save_dir=f"vocabs"
    )
    
    trg_vocab = trg_vocab_builder.build(
        train_trg, 
        min_freq=cfg.min_freq, 
        lowercase=cfg.lowercase, 
        save_dir=f"vocabs"
    )
    
    print("Source vocab size:", len(src_vocab))
    print("Target vocab size:", len(trg_vocab))
    
    del src_vocab_builder, trg_vocab_builder
    
    src_tokenizer = Tokenizer(
        language=cfg.src_lang,
        vocabulary=src_vocab
    )
    
    trg_tokenizer = Tokenizer(
        language=cfg.trg_lang,
        vocabulary=trg_vocab
    )

    train_dataset = TranslationDataset(
        src_sentences=train_src, 
        trg_sentences=train_trg, 
        src_tokenizer=src_tokenizer, 
        trg_tokenizer=trg_tokenizer, 
        lowercase=cfg.lowercase, 
        max_len=cfg.dataset_max_len
    )
    
    valid_dataset = TranslationDataset(
        src_sentences=valid_src, 
        trg_sentences=valid_trg, 
        src_tokenizer=src_tokenizer, 
        trg_tokenizer=trg_tokenizer, 
        lowercase=cfg.lowercase, 
        max_len=cfg.dataset_max_len
    )
    
    test_dataset = TranslationDataset(
        src_sentences=test_src, 
        trg_sentences=test_trg, 
        src_tokenizer=src_tokenizer, 
        trg_tokenizer=trg_tokenizer, 
        lowercase=cfg.lowercase, 
        max_len=cfg.dataset_max_len
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=16, pin_memory=True)
    
    model = Transformer(
        model_dim=cfg.model_dim, 
        src_vocab_size=len(src_vocab), 
        trg_vocab_size=len(trg_vocab),
        ffn_inner_dim=cfg.ffn_inner_dim,
        N_encoder_layers=cfg.N_encoder_layers, 
        N_decoder_layers=cfg.N_decoder_layers, 
        num_of_attention_heads=cfg.num_of_attention_heads, 
        maximum_sequence_length=cfg.maximum_sequence_length,
        learnable_positional_endocings=cfg.learnable_positional_endocings
    )
    model.to(cfg.device)
    
    if cfg.optimizer_from_paper:
        optimizer = TransformerOptimizer(model.parameters(), cfg.model_dim, warmup_steps=cfg.original_warmup_steps)
        lr_scheduler = None
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=cfg.factor,
            patience=cfg.patience
        )
        
    criterion = nn.CrossEntropyLoss(ignore_index=src_tokenizer.pad_token_id)
    
    trainer = TransformerTrainer(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        calc_bleu=cfg.calc_bleu,
        lr_scheduler=lr_scheduler,
        pad_token_id=src_tokenizer.pad_token_id,
        allow_dynamic_padding=cfg.allow_dynamic_padding, 
        src_tokenizer=src_tokenizer, 
        trg_tokenizer=trg_tokenizer,
        device=cfg.device, 
        wandb_log=wandb_log, 
    )
    
    best_valid_loss = trainer.fit(
        epochs=cfg.epochs, 
        train_dataloader=train_dataloader, 
        valid_dataloader=valid_dataloader, 
        test_dataloader=test_dataloader,
        test_step=cfg.test_step,
        save_path=cfg.save_path
    )
