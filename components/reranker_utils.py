"""
Reranker Utils Module

This module contains utility functions for reranking search results.
"""
from typing import List, Dict, Any


def diagnose_reranker_model(model, test_pairs=None):
    """
    Diagnose the reranker model to check if it's working properly.

    Parameters:
        model: The reranker model to test
        test_pairs: Optional test pairs to use; if None, some default pairs will be created

    Returns:
        dict: Diagnostic results
    """
    if test_pairs is None:
        # Create some simple test pairs
        test_pairs = [
            ("What is diabetes?",
             "Diabetes is a disease that occurs when your blood glucose is too high."),
            ("What are the symptoms of flu?",
             "Flu symptoms include fever, cough, sore throat, body aches."),
            ("How to lose weight?",
             "Exercise regularly and maintain a healthy diet to lose weight.")
        ]

    try:
        print("\n" + "="*50)
        print("RERANKER MODEL DIAGNOSTICS")
        print("="*50)
        print(f"Model: {model.__class__.__name__}")
        print(f"Test pairs: {len(test_pairs)}")

        # Test prediction
        scores = model.predict(test_pairs)

        # Check for NaN values
        scores_list = scores.tolist() if hasattr(scores, 'tolist') else list(scores)
        valid_scores = [s for s in scores_list if s == s]  # Filter out NaN

        print(
            f"Scores shape/length: {scores.shape if hasattr(scores, 'shape') else len(scores)}")
        print(f"Valid scores: {len(valid_scores)}/{len(scores_list)}")
        print(f"Score stats: min={min(valid_scores) if valid_scores else 'N/A'}, "
              f"max={max(valid_scores) if valid_scores else 'N/A'}, "
              f"mean={sum(valid_scores)/len(valid_scores) if valid_scores else 'N/A'}")
        print(f"Sample scores: {valid_scores[:3]}")
        print("="*50 + "\n")

        return {
            "success": True,
            "valid_scores": len(valid_scores),
            "total_scores": len(scores_list),
            "has_nan": len(valid_scores) != len(scores_list),
            "stats": {
                "min": min(valid_scores) if valid_scores else None,
                "max": max(valid_scores) if valid_scores else None,
                "mean": sum(valid_scores)/len(valid_scores) if valid_scores else None,
            }
        }
    except Exception as e:
        print(f"❌ Reranker diagnostic error: {e}")
        return {
            "success": False,
            "error": str(e)
        }
