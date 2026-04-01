# thoth/exceptions.py


class ThothPolicyViolation(Exception):  # noqa: N818
    def __init__(
        self,
        tool_name: str,
        reason: str,
        violation_id: str | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.reason = reason
        self.violation_id = violation_id
        super().__init__(f"Thoth blocked tool '{tool_name}': {reason}" + (f" (violation_id={violation_id})" if violation_id else ""))
