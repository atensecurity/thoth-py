"""A policy violation survives copy and pickle, which frameworks use when they re-raise from a task."""

import copy
import pickle

import pytest

from thoth.exceptions import ThothDeferredError, ThothPolicyViolation


@pytest.mark.parametrize("cls", [ThothPolicyViolation, ThothDeferredError])
@pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy, lambda exc: pickle.loads(pickle.dumps(exc))])
def test_a_violation_is_rebuilt_with_its_fields(cls, clone):
    original = cls("send_email", "outside recipient", violation_id="v-1", receipt={"id": "r-1"}, matched_rule_ids=["R1"])

    rebuilt = clone(original)

    assert type(rebuilt) is cls
    assert str(rebuilt) == str(original)
    assert (rebuilt.tool_name, rebuilt.reason, rebuilt.violation_id) == ("send_email", "outside recipient", "v-1")
    assert rebuilt.receipt == {"id": "r-1"}
    assert rebuilt.matched_rule_ids == ["R1"]
