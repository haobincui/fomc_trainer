# chk3 v2 data validation and prompt repair

## Summary

- Preserve `deepseek_v4_pro_v1` and generate a separate
  `deepseek_v4_pro_v2` release for all 2,072 chk3 samples.
- Change only the chk3 target generator, its tests, and the chk3 SFT system
  prompt. Do not change the retrain-v2 DAG, derived-release contract, or
  dataset path binding.
- Run all implementation and generation commands in the `fomc_trainer` conda
  environment.

## Validation and prompt contract

- Compare number-expression multisets after exact Decimal normalization.
  Support exact thousand/million/billion/trillion scaling,
  percent/percentage-point/basis-point conversion, and decimal/fraction
  conversion. Reject rounding, approximation, newly derived values, missing
  values, added values, or changed multiplicity.
- Remove internal `ev-<hex>` citations before number comparison. Treat mixed
  and simple fractions as one number expression rather than separate digits.
- Compare case-sensitive canonical month sets so `May` is a month and `may` is
  a modal verb. Reject added or missing months.
- Reject attribution categories introduced only by the Minutes target,
  including staff, participants or related actors, Federal Reserve bodies,
  meeting/discussion framing, votes, decisions, and policy actions.
- Reject reasoning that discusses JSON/API/schema/keys/response contracts,
  prompts/instructions, validation errors, teacher/student roles, tools, or
  the act of answering. Limit reasoning to 2,400 local-tokenizer tokens and
  the fully rendered example to 4,096 tokens.
- Require the YAML student system prompt to be byte-identical to the generator
  prompt contract.

## Testing and generation

- Add unit coverage for every allowed exact conversion and every new failure
  rule, including cache version isolation, repair behavior, tokenizer budgets,
  and YAML prompt equality.
- Run pytest, ruff, and chk3 dry-run before any provider request.
- After local gates pass, run DeepSeek V4 Pro generation at concurrency 8 into
  the v2 output directory. Resume only v2 caches after interruption.
- The release is training-eligible only when train/eval/test counts are
  1683/199/190, all 2,072 rows are accepted, no validation failures remain,
  reasoning is at most 2,400 tokens, and the rendered sequence is at most
  4,096 tokens without truncation.

## Fixed assumptions

- All conversions must be exactly provable with Decimal arithmetic; no
  tolerance or approximate equality is allowed.
- Actual month names are capitalized in the canonical English source data;
  lowercase `may` is never treated as a month.
- Existing run manifests pinned to the old YAML are not resumed after the
  prompt changes. A later training workflow must create a new run.

## Implementation note

- A live pilot showed that DeepSeek native reasoning still acknowledges its
  provider JSON envelope despite explicit negative instructions. The v2
  projection therefore keeps the raw `reasoning_content` unchanged in cache
  provenance and deterministically removes transport/meta sentences, complete
  source quotations, and duplicated final drafts before constructing the SFT
  completion. The cleaned reasoning remains provider-generated and is checked
  by the same meta, fidelity, and token gates.
- Calendar years are validated with the date set rather than the quantity
  Counter. Repeating the same year beside several source months is therefore
  allowed, while adding or omitting a year is rejected with a date error.
  Evidence IDs are removed before both quantity and date extraction so digit
  substrings inside an `ev-` token cannot be mistaken for years.
- Month-day expressions such as `August 5` are likewise compared as dates,
  not as economic quantities. Their normalized date set must remain equal,
  while a faithful paragraph may repeat the same date for clarity.
