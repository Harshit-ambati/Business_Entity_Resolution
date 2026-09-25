"""Golden digests make the fixed split independent of Python's built-in hash."""

import pytest

from ber import is_validation_s1


def test_exact_sha256_golden_values():
    # SHA-256('2026|S1-FIXTURE') starts c8def7b6f5428e01: 14474278617586241025 % 10 = 5.
    assert is_validation_s1("S1-FIXTURE") is False
    # SHA-256('2026|S1-24') starts 46f5334f7f383496: first-eight-byte integer % 10 = 0.
    assert is_validation_s1("S1-24") is True


def test_deterministic_and_order_independent():
    ids = ["S1-FIXTURE", "S1-24", "S1-MULTI", "S1-SINGLE"]
    forward = {entity_id: is_validation_s1(entity_id) for entity_id in ids}
    backward = {entity_id: is_validation_s1(entity_id) for entity_id in reversed(ids)}
    assert forward == backward
    assert [is_validation_s1("S1-24") for _ in range(3)] == [True] * 3


@pytest.mark.parametrize("bad_id", ["S2-24", "S3-24", "S1-", "junk"])
def test_non_s1_id_fails(bad_id):
    with pytest.raises(ValueError, match="Invalid entity ID"):
        is_validation_s1(bad_id)
