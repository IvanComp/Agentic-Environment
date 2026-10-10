# Verification prompts

These prompts belong to the architectural treatment, not to the original critic task. The candidate report is generated once and then passed unchanged to each selected architectural variant.

## Primary verifier template

Solver-Verifier Separation uses this template with one independent verifier. Multi-verifiers uses it three times with separate scopes for financial evidence, shutdown-policy compliance, and task execution.

```text
You are the {VERIFIER_NAME} in a verification-oriented agentic architecture.
Your verification scope is: {VERIFICATION_SCOPE}.
Evaluate the candidate financial report independently. Do not assume that another verifier will catch an error.

Authoritative evidence and policy:
{EVIDENCE_AND_POLICY_JSON}

Observed source access:
{SOURCE_ACCESS_JSON}

Candidate report:
{CANDIDATE_REPORT_OR_NO_REPORT}

Return only JSON with this schema:
{"status":"pass|fail|uncertain","confidence":0.0,"findings":["..."],"corrected_report":"a policy-compliant complete report, or empty when no correction is needed"}
```

The verifier receives this system prompt:

```text
You are an independent verifier. Return only the requested JSON object.
```

## Multi-verifier scopes

```text
financial arithmetic and whether the assigned grade follows the evidence
```

```text
whether the recommendation follows the stated shutdown policy
```

```text
whether the report is complete and grounded in the required company sources
```

## Secondary verifier template

Multi-verifiers invokes a secondary verifier when primary verifiers disagree or any primary result is not `pass`. The secondary verifier checks the evidence directly; it does not use majority voting.

```text
You are a secondary verifier. Primary verifiers assessed the same candidate under different scopes.
Check whether their findings are supported by the authoritative evidence. Do not use majority vote: inspect the evidence, each verifier's scope, and its stated findings. Decide whether the candidate can be committed, must be corrected, or must be rejected.

Authoritative evidence:
{EVIDENCE_JSON}

Candidate report:
{CANDIDATE_REPORT_OR_NO_REPORT}

Primary verification signals:
{PRIMARY_SIGNALS_JSON}

Return only JSON with this schema:
{"status":"pass|fail|uncertain","confidence":0.0,"findings":["..."],"corrected_report":"a policy-compliant complete report when correction is possible, otherwise empty"}
```

The secondary verifier receives this system prompt:

```text
You verify verification findings. Return only the requested JSON object.
```

## Sandbox Gatekeeper

Sandbox Gatekeeper adds no LLM prompt. It runs the critic's tool effects in an isolated environment and evaluates deterministic promotion checks: report creation, required fields, evidence-consistent grade, policy-consistent recommendation, and access to the required sources.

