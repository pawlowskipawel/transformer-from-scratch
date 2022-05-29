args = {
    # global
    "calc_bleu": True,
    "src_lang": "en",
    "trg_lang": "de",
    "device": "cuda",
    "save_path": "checkpoints",
    "test_step": 5, # perform test every 5 epochs
    
    #vocab
    "min_freq": 2,
    "lowercase": True,
    "max_vocab_size": 10000,
    "src_save_path": "vocabs/en.vocab",
    "trg_save_path": "vocabs/de.vocab",
    
    # model
    "model_dim": 512,
    "ffn_inner_dim": 2048, 
    "N_encoder_layers": 6, 
    "N_decoder_layers": 6, 
    "num_of_attention_heads": 8, 
    "maximum_sequence_length": 512,
    "learnable_positional_endocings": False,
    
    # dataset
    "dataset_max_len": 128,
    "allow_dynamic_padding": True,
    
    # training
    "epochs": 50,
    "batch_size": 128,
    
    "optimizer_from_paper": False,
    "learning_rate": 0.0003, # initial learning used when optimizer_from_paper is False
    
    # lr scheduler
    "original_warmup_steps": 4000, # used when optimizer_from_paper is True
    "factor": 0.7,
    "patience": 2,
}