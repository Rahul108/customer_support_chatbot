from typing import List
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


class Evaluator:
    """Evaluation metrics for chatbot responses."""

    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def compute_rouge_scores(self, reference: str, hypothesis: str) -> dict:
        """
        Compute ROUGE scores between reference and hypothesis.
        Args:
            reference: Reference text
            hypothesis: Hypothesis/generated text
        Returns:
            Dictionary with ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        }

    def compute_bleu_score(self, reference: List[str], hypothesis: List[str]) -> float:
        """
        Compute BLEU score (simplified version).
        Args:
            reference: Reference tokens
            hypothesis: Hypothesis tokens
        Returns:
            BLEU score
        """
        matches = sum(1 for ref, hyp in zip(reference, hypothesis) if ref == hyp)
        return matches / len(hypothesis) if hypothesis else 0.0

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple semantic similarity based on token overlap.
        Args:
            text1: First text
            text2: Second text
        Returns:
            Similarity score (0-1)
        """
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0

    def evaluate_batch(
        self, references: List[str], hypotheses: List[str]
    ) -> dict:
        """
        Evaluate a batch of responses.
        Args:
            references: List of reference texts
            hypotheses: List of hypothesis texts
        Returns:
            Dictionary with aggregated metrics
        """
        rouge_scores = []
        similarity_scores = []

        for ref, hyp in zip(references, hypotheses):
            rouge = self.compute_rouge_scores(ref, hyp)
            rouge_scores.append(rouge)
            similarity = self.compute_semantic_similarity(ref, hyp)
            similarity_scores.append(similarity)

        avg_rouge1 = np.mean([r["rouge1"] for r in rouge_scores])
        avg_rouge2 = np.mean([r["rouge2"] for r in rouge_scores])
        avg_rougeL = np.mean([r["rougeL"] for r in rouge_scores])
        avg_similarity = np.mean(similarity_scores)

        return {
            "avg_rouge1": avg_rouge1,
            "avg_rouge2": avg_rouge2,
            "avg_rougeL": avg_rougeL,
            "avg_similarity": avg_similarity,
        }

    @staticmethod
    def compute_perplexity(loss: float) -> float:
        """
        Compute perplexity from loss.
        Args:
            loss: Cross-entropy loss
        Returns:
            Perplexity
        """
        return np.exp(loss)
