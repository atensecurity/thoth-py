# Thoth SDK

Thoth SDK instruments your AI agents for governance, policy enforcement, and behavioral monitoring.
Every tool call is evaluated against your organization's security policies in <100ms — blocking,
stepping up for human approval, or silently observing based on your configured enforcement mode.

**Package name:** `aten-thoth` | **PyPI:** `pip install aten-thoth`

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [How It Works](#how-it-works)
4. [Integration Examples](#integration-examples)
   - [LangChain AgentExecutor](#langchain-agentexecutor)
   - [LangGraph](#langgraph)
   - [OpenAI Function Calling](#openai-function-calling)
   - [Anthropic Claude](#anthropic-claude)
   - [CrewAI](#crewai)
   - [AutoGen](#autogen)
   - [Custom Tools (Generic)](#custom-tools-generic)
5. [Enforcement Modes](#enforcement-modes)
6. [Policy Decisions](#policy-decisions)
7. [Handling Violations](#handling-violations)
8. [Step-Up Authentication](#step-up-authentication)
9. [Session Inspection](#session-inspection)
10. [Configuration Reference](#configuration-reference)
11. [Dashboard](#dashboard)

---

## Installation

```bash
pip install aten-thoth
```

**With LangChain / LangGraph support:**

```bash
pip install "aten-thoth[langchain]"
```

**With OpenAI support:**

```bash
pip install "aten-thoth[openai]"
```

**With AutoGen support:**

```bash
pip install "aten-thoth[autogen]"
```

**Requirements:** Python 3.12+

---

## Quick Start

**1. Get your API key** from the [Aten Security dashboard](https://app.aten.security) under
**Settings → API Keys**.

**2. Set the environment variable:**

```bash
export THOTH_API_KEY="thoth_live_..."
```

**3. Instrument your agent — three lines of code:**

```python
import os
import thoth

# Instrument your agent — returns the same object, mutated in-place
agent = thoth.instrument(
    agent,
    agent_id="document-summarizer",
    approved_scope=["read_file", "summarize"],   # tools outside this list are blocked
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@example.com",
    enforcement="progressive",   # observe → step_up → block
    # api_key set via THOTH_API_KEY env var
)

# Every tool call is now governed — no other changes required
result = agent.run("Summarize the attached document and send it to the team.")
```

That's it. No AWS credentials, no infrastructure setup — the SDK connects to Aten's managed
service using your API key and sends events and enforcement requests over HTTPS.

---

## How It Works

```
Agent calls tool
      │
      ▼
 Thoth intercepts (wrap_tool)
      │
      ├── Emits TOOL_CALL_PRE event → Aten API (async, non-blocking)
      │
      ├── Calls enforcer /v1/enforce
      │        │
      │        ├── ALLOW   → tool executes normally
      │        ├── STEP_UP → waits for human approval (polls /v1/enforce/hold/{token})
      │        └── BLOCK   → raises ThothPolicyViolation
      │
      ├── Tool executes (if allowed)
      │
      └── Emits TOOL_CALL_POST event → Aten API (async, non-blocking)
```

Events are batched and flushed to the Aten ingest API in a background daemon thread with
at-most-10 per batch. If the enforcer is unreachable, Thoth fails open (ALLOW) and logs a
warning — it never blocks production traffic due to an infrastructure fault.

---

## Integration Examples

### LangChain AgentExecutor

Thoth detects `AgentExecutor` automatically and wraps both `tool.run` and `tool._run`:

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import thoth

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    ...

@tool
def read_file(path: str) -> str:
    """Read a file from the local filesystem."""
    ...

llm = ChatOpenAI(model="gpt-4o")
agent = create_openai_tools_agent(llm, tools=[web_search, read_file], prompt=...)
executor = AgentExecutor(agent=agent, tools=[web_search, read_file])

# One call instruments all tools
executor = thoth.instrument(
    executor,
    agent_id="research-agent",
    approved_scope=["web_search", "read_file"],
    tenant_id="acme-corp",
    user_id="bob@acme.com",
    enforcement="block",
)

# Now every tool invocation is policy-checked
result = executor.invoke({"input": "Find recent SEC filings for AAPL"})
```

### LangGraph

For LangGraph agents, wrap tool callables directly using `Tracer.wrap_tool`. Because LangGraph
doesn't use `AgentExecutor`, the low-level API is needed here:

```python
import os
import thoth
from thoth.models import ThothConfig, EnforcementMode
from thoth.session import SessionContext
from thoth.emitter import HttpEmitter
from thoth.enforcer_client import EnforcerClient
from thoth.step_up import StepUpClient
from thoth.tracer import Tracer
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

@tool
def search_database(query: str) -> str:
    """Search the internal knowledge base."""
    ...

@tool
def send_slack_message(channel: str, message: str) -> str:
    """Post a message to Slack."""
    ...

api_key = os.environ["THOTH_API_KEY"]
config = ThothConfig(
    agent_id="langgraph-analyst",
    approved_scope=["search_database", "send_slack_message"],
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement=EnforcementMode.BLOCK,
    api_key=api_key,
)
session = SessionContext(config)
tracer = Tracer(
    config=config,
    session=session,
    emitter=HttpEmitter(api_url=config.resolved_api_url, api_key=api_key),
    enforcer=EnforcerClient(config),
    step_up=StepUpClient(config),
)

governed_search = tracer.wrap_tool("search_database", search_database)
governed_slack = tracer.wrap_tool("send_slack_message", send_slack_message)

llm = ChatAnthropic(model="claude-sonnet-4-6")
agent = create_react_agent(llm, [governed_search, governed_slack])

result = agent.invoke({"messages": [("user", "Search for Q4 data and post it to #analytics")]})
```

See `examples/langgraph_example.py` for the full working script.

### Anthropic Claude

Use `instrument_anthropic()` to wrap tool functions for the Anthropic agentic loop:

```python
import os
import anthropic
import thoth
from thoth import ThothPolicyViolation

governed = thoth.instrument_anthropic(
    {"search_docs": search_docs, "delete_record": delete_record},
    agent_id="claude-research-agent",
    approved_scope=["search_docs"],
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="step_up",
    # THOTH_API_KEY read from env automatically
)

client = anthropic.Anthropic()
messages = [{"role": "user", "content": "Find our data retention policy."}]

while True:
    response = client.messages.create(
        model="claude-opus-4-6", max_tokens=1024, tools=TOOLS, messages=messages
    )
    if response.stop_reason == "end_turn":
        break
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            try:
                result = governed[block.name](block.input)   # Thoth runs here
            except ThothPolicyViolation as e:
                result = f"[blocked: {e.reason}]"
            tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(result)})
    messages += [{"role": "assistant", "content": response.content},
                 {"role": "user", "content": tool_results}]
```

See `examples/anthropic_example.py` for the full working script.

### OpenAI Function Calling

Use `instrument_openai()` to wrap tool functions for the OpenAI tool-calling loop:

```python
import os
import openai
import thoth
from thoth import ThothPolicyViolation

def search_docs(query: str) -> str:
    ...

def send_email(to: str, subject: str, body: str) -> str:
    ...

governed = thoth.instrument_openai(
    {"search_docs": search_docs, "send_email": send_email},
    agent_id="openai-agent",
    approved_scope=["search_docs"],
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="charlie@acme.com",
    enforcement="block",
    # THOTH_API_KEY read from env automatically
)

client = openai.OpenAI()
messages = [{"role": "user", "content": "Find docs about access control"}]

while True:
    response = client.chat.completions.create(model="gpt-4o", tools=TOOLS, messages=messages)
    msg = response.choices[0].message
    if not msg.tool_calls:
        break
    messages.append(msg)
    for call in msg.tool_calls:
        try:
            import json
            result = governed[call.function.name](json.loads(call.function.arguments))
        except ThothPolicyViolation as e:
            result = f"[blocked: {e.reason}]"
        messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
```

See `examples/openai_example.py` for the full working script.

### CrewAI

`thoth.instrument()` detects CrewAI `Agent` objects automatically:

```python
import os
import thoth
from crewai import Agent
from crewai.tools import tool

@tool("web_search")
def web_search(query: str) -> str:
    """Search the web."""
    ...

researcher = Agent(role="Researcher", goal="Find information", tools=[web_search])

thoth.instrument(
    researcher,
    agent_id="crewai-researcher",
    approved_scope=["web_search"],
    tenant_id=os.environ["THOTH_TENANT_ID"],
    user_id="alice@acme.com",
    enforcement="block",
    # THOTH_API_KEY read from env automatically
)
```

See `examples/crewai_example.py` for the full working script.

### AutoGen

Use `wrap_autogen_tools()` to govern AutoGen's `function_map`:

```python
import os
import autogen
from thoth.integrations.autogen import wrap_autogen_tools
from thoth.tracer import Tracer
# ... build tracer as shown in the LangGraph section above ...

governed = wrap_autogen_tools(
    {"search_docs": search_docs, "send_email": send_email},
    tracer=tracer,
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    function_map=governed,   # Thoth enforcement runs on every function call
)
```

See `examples/autogen_example.py` for the full working script.

### Custom Tools (Generic)

Any object with a `.tools` list and tools that have a `.run` method or are directly callable
will be instrumented automatically:

```python
import thoth
from thoth import ThothPolicyViolation

class FileTool:
    name = "read_file"

    def run(self, path: str) -> str:
        with open(path) as f:
            return f.read()

class MyAgent:
    tools = [FileTool()]

    def run(self, prompt: str) -> str:
        # ... your agent logic calls self.tools[0].run(...)
        ...

agent = MyAgent()

agent = thoth.instrument(
    agent,
    agent_id="file-agent",
    approved_scope=["read_file"],
    tenant_id="acme-corp",
    enforcement="block",
)

try:
    result = agent.tools[0].run("/etc/passwd")
except ThothPolicyViolation as e:
    print(f"Blocked: {e.tool_name} — {e.reason}")
    if e.violation_id:
        print(f"Violation record: {e.violation_id}")
```

---

## Enforcement Modes

Set via the `enforcement` parameter to `instrument()`.

| Mode | Value | Behavior |
|---|---|---|
| Observe | `observe` | All tool calls pass through. Events are still emitted for audit. No blocking, no step-up. Use for initial rollout and baselining. |
| Step-Up | `step_up` | Suspicious calls trigger a human approval request (e.g. Slack DM to a reviewer). Tool execution is held until approved or timed out. |
| Block | `block` | Calls that violate policy raise `ThothPolicyViolation` immediately. |
| Progressive | `progressive` | Default. Enforcer chooses the appropriate response per tool call based on policy rules. |

---

## Policy Decisions

The enforcer returns one of three decisions for each tool call:

| Decision | Meaning | Agent behavior |
|---|---|---|
| `ALLOW` | Call is within policy. | Tool executes immediately. |
| `STEP_UP` | Call requires human approval. | SDK polls `/v1/enforce/hold/{token}` until approved or timed out. On timeout → `BLOCK`. |
| `BLOCK` | Call violates policy. | `ThothPolicyViolation` is raised before the tool executes. |

Enforcer errors (network timeout, 5xx) always result in `ALLOW` so that infrastructure faults
never interrupt production workloads. All errors are logged at `WARNING` level.

---

## Handling Violations

```python
from thoth import ThothPolicyViolation

try:
    result = agent.tools[0].run(user_input)
except ThothPolicyViolation as e:
    # e.tool_name   — the tool that was blocked
    # e.reason      — human-readable policy reason
    # e.violation_id — reference ID for the violation record in Maat dashboard
    logger.warning("Policy violation on %s: %s (id=%s)", e.tool_name, e.reason, e.violation_id)
    return {"error": "This action is not permitted under your current access policy."}
```

---

## Step-Up Authentication

When the enforcer returns `STEP_UP`, Thoth automatically:

1. Sends an approval request to the configured notification channel (Slack by default).
2. Polls `/v1/enforce/hold/{hold_token}` every `step_up_poll_interval_seconds` (default: 5s).
3. If approved within `step_up_timeout_minutes` (default: 15 minutes) → tool executes.
4. If timed out → raises `ThothPolicyViolation` with reason `"step-up auth timeout"`.

Configure timeouts via `ThothConfig`:

```python
from thoth.models import ThothConfig, EnforcementMode

config = ThothConfig(
    agent_id="sensitive-agent",
    approved_scope=["delete_record"],
    tenant_id="acme-corp",
    enforcement=EnforcementMode.STEP_UP,
    step_up_timeout_minutes=5,       # fail fast in production
    step_up_poll_interval_seconds=3,
)
```

---

## Session Inspection

Each `instrument()` call creates a `SessionContext` stored in a `contextvars.ContextVar`. Inspect
it from anywhere in the same async context:

```python
from thoth import get_current_session

session = get_current_session()
if session:
    print(f"Session ID:    {session.session_id}")
    print(f"Tools called:  {session.tool_calls}")
    print(f"Token spend:   {session.token_spend}")
    print(f"In scope:      {session.is_in_scope('web_search')}")
```

---

## Configuration Reference

### `instrument()` parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `agent` | `Any` | — | Agent object to instrument. Must have a `.tools` list. |
| `agent_id` | `str` | — | Stable identifier for this agent definition. Used in policy rules. |
| `approved_scope` | `list[str]` | — | List of tool names this agent is authorized to call. |
| `tenant_id` | `str` | — | Your Maat tenant ID. |
| `user_id` | `str` | `"system"` | Identity of the user on whose behalf the agent acts. |
| `enforcement` | `str` | `"progressive"` | Enforcement mode: `observe`, `step_up`, `block`, or `progressive`. |
| `api_key` | `str \| None` | `$THOTH_API_KEY` | API key from the Aten dashboard. Events sent over HTTPS — no AWS credentials required. |
| `session_id` | `str \| None` | auto-generated UUID | Pass an existing session ID to continue a session across calls. |

### `ThothConfig` fields

| Field | Type | Default | Description |
|---|---|---|---|
| `agent_id` | `str` | — | Stable identifier for this agent. |
| `approved_scope` | `list[str]` | — | Tool names the agent is authorized to call. |
| `tenant_id` | `str` | — | Your Maat tenant ID. |
| `user_id` | `str` | `"system"` | User identity for the session. |
| `enforcement` | `EnforcementMode` | `PROGRESSIVE` | Enforcement mode. |
| `api_key` | `str \| None` | `None` | Aten API key. Falls back to `THOTH_API_KEY` env var via `_build_components`. |
| `api_url` | `str` | `https://api.aten.security` | Base URL for the Aten managed API. Override via `THOTH_API_URL` env var. |
| `step_up_timeout_minutes` | `int` | `15` | Timeout for step-up approval. |
| `step_up_poll_interval_seconds` | `int` | `5` | Polling interval for step-up hold status. |

### Environment Variables

| Variable | Description |
|---|---|
| `THOTH_API_KEY` | API key from the Aten dashboard. Enables HTTPS event transport. Example: `thoth_live_abc123...` |
| `THOTH_API_URL` | Override the Aten API base URL. Useful for self-hosted deployments. Example: `https://thoth.your-domain.com` |
| `THOTH_ENFORCER_URL` | Override the enforcer URL. Defaults to the Aten API when `api_key` is set. Example: `https://enforcer.your-domain.com` |

When both an explicit parameter and an environment variable are set, the explicit parameter takes precedence.

---

## Dashboard

View sessions, violations, step-up requests, and policy decisions in the
[Maat Governance Dashboard](https://app.aten.security).

The dashboard shows:
- **Sessions** — per-agent session timelines with all tool calls
- **Violations** — blocked or escalated actions with full context
- **Approvals** — step-up queue for human reviewers
- **Policies** — view and edit the rules driving enforcement decisions
- **Behavioral Analytics** — drift detection and anomaly scores over time

---

## License

Copyright 2026 Aten Security, Inc. All rights reserved.
