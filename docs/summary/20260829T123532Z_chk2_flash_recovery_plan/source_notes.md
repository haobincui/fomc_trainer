# Source notes

## Scope

The recovery denominator is the 1,710 machine-gate non-passes from the DeepSeek V4 Flash release. Preparation-stage exclusions are outside scope and were not selected for retry.

## Integrity

The four selection ledgers were checked by row count and by SHA-256 over sorted sample IDs. The three action cohorts are mutually exclusive and sum to the full 1,710-row rejection ledger. Subset preparation also validates split assignment and official-minutes paragraph hashes before any provider call.

## Execution boundary

The verifier-only canary made 32 attempts across eight rows; all returned HTTP 402 with `Insufficient Balance`. The job was stopped, and no successful new generation or verification cache exists. The original release caches remain present.

## Interpretation

The cohort rules are operational heuristics intended to prioritize likely recoveries without weakening quality gates. They are not causal estimates of retry success.
