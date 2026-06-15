package thoth.policies.langgraph.research_agent

import future.keywords.if
import future.keywords.in

# thoth_rule effect=BLOCK action_prefix=tool_call:fetch_external_api reason=Outbound_HTTP_domain_not_allowlisted
# thoth_rule effect=STEP_UP action_prefix=tool_call:write_local_file reason=File_writes_require_human_approval
# thoth_rule effect=FLAG action_prefix=tool_call:read_customer_records reason=Bulk_data_read_requires_audit_flag

# Canonical references expected by bundle validators.
principal := object.get(input, "principal", {})
action := object.get(input, "action", {})
context := object.get(input, "context", {})

trimspace(value) := trim(sprintf("%v", [value]), " \t\n\r")

action_name := lower(trimspace(object.get(action, "name", "")))
action_payload := object.get(action, "payload", {})

destination_domain := lower(trimspace(object.get(action_payload, "domain", "")))
record_count := to_number(object.get(action_payload, "record_count", 0))
target_path := trimspace(object.get(action_payload, "path", ""))

allowlisted_domains := {
  "api.github.com",
  "hn.algolia.com",
  "www.sec.gov",
}

is_http_tool if {
  action_name in {"fetch_external_api", "web_search"}
}

deny[reason] if {
  principal != {}
  context != {}
  is_http_tool
  destination_domain != ""
  not destination_domain in allowlisted_domains
  reason := sprintf("BLOCK: outbound domain %q is not allowlisted", [destination_domain])
}

step_up[reason] if {
  principal != {}
  context != {}
  action_name == "write_local_file"
  reason := sprintf("STEP_UP: file write requires approval (path=%q)", [target_path])
}

flag[reason] if {
  principal != {}
  context != {}
  action_name == "read_customer_records"
  record_count > 10
  reason := sprintf("FLAG: bulk data read of %.0f records", [record_count])
}

allow if {
  count(deny) == 0
}
