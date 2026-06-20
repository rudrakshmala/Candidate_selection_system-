import pandas as pd
import pytest

def test_submission_format():
    """
    Validate submission.csv format against Spec 10.
    Spec requires: exactly 100 rows, headers: candidate_id, rank, score, reasoning.
    """
    csv_path = "submission.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        pytest.fail("submission.csv not found. Did you run rank.py?")

    # 1. Exactly 100 rows
    assert len(df) == 100, f"Expected exactly 100 rows, found {len(df)}"

    # 2. Required columns
    expected_cols = ["candidate_id", "rank", "score", "reasoning"]
    for col in expected_cols:
        assert col in df.columns, f"Missing required column: '{col}'"

    # 3. No extra columns
    for col in df.columns:
        assert col in expected_cols, f"Unexpected extra column: '{col}'"

    # 4. Rank should be 1 to 100
    assert df["rank"].min() == 1, "Rank should start at 1"
    assert df["rank"].max() == 100, "Rank should end at 100"
    assert df["rank"].is_monotonic_increasing, "Ranks should be sorted ascending"
    
    # 5. Score should be sorted descending
    assert df["score"].is_monotonic_decreasing, "Scores should be sorted descending"

    print("[PASS] submission.csv format is valid.")

if __name__ == "__main__":
    test_submission_format()
