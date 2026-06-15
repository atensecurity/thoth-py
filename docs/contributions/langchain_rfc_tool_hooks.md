# RFC: Pre-execution tool call interception hooks for policy enforcement

I am using `StateGraph` + `ToolNode` for production agents, and I keep hitting a gap around
pre-execution control over tool calls.

## Problem

LangGraph routes model tool calls into `ToolNode` (`_run_one` / `_arun_one`) and then executes
the selected tool. In practice, there is no simple first-class `before_tool_call` hook with a
stable contract to intercept, inspect, and block/modify a call before execution.

Today, the practical options are custom `ToolNode` subclassing and private execution-path
overrides, which feels fragile over upgrades.

For teams using `Command`-based routing and `interrupt()` flows, this matters because policy
decisions need to happen at the exact boundary before tool execution, not after side effects.

## Proposed minimal interface

Add a small optional hook on `ToolNode`:

- Input: `tool_name`, `tool_input`, `graph_state`
- Output: `ALLOW`, `BLOCK`, or `MODIFY`, with a human-readable reason

Python sketch:

```python
class ToolNode:
    def __init__(
        self,
        tools,
        before_tool_call=None,  # new
    ):
        ...

    async def _arun_one(self, call, input, config):
        if self.before_tool_call:
            decision = await self.before_tool_call(
                tool_name=call.name,
                tool_input=call.args,
                state=input,
            )
            if decision.action == "BLOCK":
                return ToolMessage(
                    content=f"Blocked: {decision.reason}",
                    tool_call_id=call.id,
                )
        # existing execution path continues
```

## Why this helps

- Gives teams a stable, documented pre-execution interception point.
- Avoids brittle subclassing against internal methods.
- Keeps behavior composable with existing `Command` and `interrupt()` patterns.
- Makes policy/governance controls straightforward without changing agent logic.

Has anyone solved this cleanly with custom ToolNode subclasses? Would love to see patterns before proposing a full PR.
