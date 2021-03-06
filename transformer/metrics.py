# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/metrics.ipynb (unless otherwise specified).

__all__ = ['calculate_bleu_score']

# Cell
from torchtext.data.metrics import bleu_score
from .translation import translate_sentence
from tqdm import tqdm

# Cell
def calculate_bleu_score(model, src_sentence, trg_sentence, src_tokenizer, trg_tokenizer):
    ground_truth_sentence = trg_tokenizer(trg_sentence)["tokens"]

    translated_sentence = translate_sentence(model, src_sentence, src_tokenizer, trg_tokenizer)

    bleu = bleu_score([translated_sentence], [[ground_truth_sentence]])

    return round(bleu * 100, 2)