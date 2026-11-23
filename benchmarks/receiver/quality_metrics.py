"""Corpus reconstruction metrics with bounded semantic-embedding memory."""

import math
import re
import unicodedata

import numpy as np
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU, CHRF

METRICS = ("bleu", "chrf", "semantic_similarity", "rouge_l", "token_accuracy")


def normalize_model_text(text):
    """Match the normalized text representation used to train the BPE model."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def detokenize_natural_text(text):
    """Undo punctuation spacing for metrics that expect natural text."""
    return re.sub(r"\s+([.!?,;:])", r"\1", text)


class QualityScorer:
    def __init__(self, tokenizer, semantic_encoder, embedding_batch_size):
        self.tokenizer = tokenizer
        self.encoder = semantic_encoder
        self.batch_size = embedding_batch_size
        self.bleu = BLEU(tokenize="13a", smooth_method="exp")
        self.chrf = CHRF(word_order=2)
        self.rouge = RougeScorer(["rougeL"], use_stemmer=True)

    def encode(self, texts):
        return self.encoder.encode(texts, batch_size=self.batch_size, convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)

    def score_trial(self, references, hypotheses):
        """Score every receiver; embed and tokenize the references once per batch."""
        if not references or not hypotheses or any(
            len(values) != len(references) for values in hypotheses.values()
        ):
            raise ValueError("Quality scoring requires equal nonempty corpora")
        token_refs = [normalize_model_text(text) for text in references]
        token_hyps = {name: [normalize_model_text(text) for text in values]
                      for name, values in hypotheses.items()}
        refs = [detokenize_natural_text(text) for text in token_refs]
        hyps = {name: [detokenize_natural_text(text) for text in values]
                for name, values in token_hyps.items()}
        n = len(refs)
        scores = {name: {
            "bleu": self.bleu.corpus_score(values, [refs]).score,
            "chrf": self.chrf.corpus_score(values, [refs]).score,
            "semantic_similarity": 0.0, "rouge_l": 0.0, "token_accuracy": 0.0,
        } for name, values in hyps.items()}
        for start in range(0, n, self.batch_size):
            stop = start + self.batch_size
            ref_batch = refs[start:stop]
            ref_emb = self.encode(ref_batch)
            ref_ids = self.tokenizer.encode(token_refs[start:stop])
            for name, values in hyps.items():
                hyp_batch = values[start:stop]
                score = scores[name]
                score["semantic_similarity"] += float(
                    np.sum(ref_emb * self.encode(hyp_batch), dtype=np.float64)
                )
                score["rouge_l"] += sum(self.rouge.score(ref, hyp)["rougeL"].fmeasure
                                        for ref, hyp in zip(ref_batch, hyp_batch))
                for ref, hyp in zip(ref_ids, self.tokenizer.encode(token_hyps[name][start:stop])):
                    denominator = max(len(ref), len(hyp))
                    score["token_accuracy"] += (
                        sum(a == b for a, b in zip(ref, hyp)) / denominator if denominator else 1.0
                    )
        for score in scores.values():
            for metric in ("semantic_similarity", "rouge_l", "token_accuracy"):
                score[metric] /= n
            if not all(math.isfinite(value) for value in score.values()):
                raise FloatingPointError("Nonfinite reconstruction metric")
        return scores

    def signatures(self):
        return {"bleu": str(self.bleu.get_signature()), "chrf": str(self.chrf.get_signature()),
                "text_normalization": "NFKC, lowercase, whitespace collapsed; spaces before .!?,;: removed for natural-text metrics",
                "semantic_similarity": "mean cosine of L2-normalized sentence embeddings",
                "rouge_l": "mean sentence ROUGE-L F1, English stemming",
                "token_accuracy": "mean position-wise BPE accuracy after re-encoding model-normalized text before natural-text detokenization; denominator max(ref, hyp) length"}
