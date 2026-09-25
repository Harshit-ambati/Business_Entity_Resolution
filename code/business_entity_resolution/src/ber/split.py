"""Fixed, order-independent S1 holdout assignment from the team contract."""

from hashlib import sha256

from .contracts import validate_entity_id


def is_validation_s1(source1_entity_id: str) -> bool:
    """Return True for the fixed approximately 10% S1 validation partition."""
    validate_entity_id(source1_entity_id, ("S1-",))
    digest = sha256(("2026|" + source1_entity_id).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value % 10 == 0
