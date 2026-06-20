import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.precompute.engineer_features import compute_domain_fit

def test_seo_marketing_manager():
    """
    Ensure an SEO Marketing Manager with 'search' and 'ranking' does not score high.
    It should be gated low because it's a non-tech title and matches SEO context.
    """
    career_history = [
        {
            "description": "Improved search engine optimization for the website. "
                           "Achieved first page google ranking for our core product. "
                           "Increased search traffic by 200%. "
                           "Built and managed a team of writers."
        }
    ]
    score = compute_domain_fit(career_history, "Marketing Manager", "SEO Expert")
    
    # Should trigger the 0.02 hard gate because SEO context zeroed domain_count
    assert score <= 0.10, f"SEO Marketing Manager scored too high: {score}"
    assert score == 0.02, f"Expected exactly 0.02, got {score}"


def test_genuine_ml_engineer():
    """
    Ensure a genuine ML Engineer with production ranking language scores highly,
    and their score doesn't drop due to stricter rules.
    """
    career_history = [
        {
            "description": "Built a ranking system for ecommerce. "
                           "Deployed model to production serving 10M real users at scale. "
                           "Implemented semantic search and vector search using sentence-transformers. "
                           "Improved NDCG and MRR by 15%."
        }
    ]
    score = compute_domain_fit(career_history, "Machine Learning Engineer", "Ranking specialist")
    
    # Should score very highly
    assert score > 0.80, f"Genuine ML Engineer scored too low: {score}"


if __name__ == "__main__":
    test_seo_marketing_manager()
    test_genuine_ml_engineer()
    print("[PASS] Domain fit unit tests passed!")
