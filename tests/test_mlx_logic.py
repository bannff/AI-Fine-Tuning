import json
import tempfile
from pathlib import Path

from hypothesis import given
from hypothesis import strategies as st

from peft.runtime.mlx_logic import prepare_mlx_data


@given(
    st.lists(
        st.dictionaries(keys=st.text(min_size=1), values=st.text()),
        min_size=10,
        max_size=1000,
    )
)
def test_prepare_mlx_data_splits_correctly(sample_data):
    """
    Hypothesis property-based test to ensure our dataset splitting logic
    maintains data integrity and adheres precisely to the specified ratio.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.jsonl"
        output_dir = Path(temp_dir) / "output"

        # 1. Setup mock data
        with open(input_file, "w", encoding="utf-8") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        # 2. Execute target function
        split_ratio = (0.8, 0.1, 0.1)
        result_dir = prepare_mlx_data(str(input_file), str(output_dir), split_ratio)

        # 3. Asserts
        assert Path(result_dir).exists()

        train_file = Path(result_dir) / "train.jsonl"
        valid_file = Path(result_dir) / "valid.jsonl"
        test_file = Path(result_dir) / "test.jsonl"

        assert train_file.exists()
        assert valid_file.exists()
        assert test_file.exists()

        # Verification of lengths
        def count_lines(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return sum(1 for _ in f)

        train_len = count_lines(train_file)
        valid_len = count_lines(valid_file)
        test_len = count_lines(test_file)

        total_len = len(sample_data)

        # Verify sizes match mathematical expectations
        expected_train = int(total_len * split_ratio[0])
        expected_valid = int(total_len * split_ratio[1])
        expected_test = total_len - (expected_train + expected_valid)

        assert train_len == expected_train
        assert valid_len == expected_valid
        assert test_len == expected_test
        assert train_len + valid_len + test_len == total_len


def test_prepare_mlx_data_integrity():
    """Test that actual data is preserved across the split."""
    sample_data = [{"id": i, "val": f"test_{i}"} for i in range(50)]

    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.jsonl"
        output_dir = Path(temp_dir) / "output"

        with open(input_file, "w", encoding="utf-8") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")

        prepare_mlx_data(str(input_file), str(output_dir))

        # Check integrity of first file
        train_file = Path(output_dir) / "train.jsonl"
        with open(train_file, "r") as f:
            lines = f.readlines()

        first_record = json.loads(lines[0])
        assert first_record["id"] == 0
        assert first_record["val"] == "test_0"
