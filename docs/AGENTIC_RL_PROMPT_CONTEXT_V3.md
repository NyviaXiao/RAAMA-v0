# Public prompt/context preview v3

`adaptive_mas.agents.public_prompt_context` builds a separate, point-in-time public-evidence context. It filters sources by publication/availability date, security, and any supplied topic tags; ranks by question overlap; de-duplicates identical public content; and keeps local source-record IDs only in a separate reference map. The model prompt cites document titles exactly; titleless financial facts use their report period and exact field names.

The evidence pool is neutral. A caller may attach `evidence_side` as `支持`, `反证`, or `未分类` based only on contemporaneous public evidence; the builder never derives a side from future returns and never sends these labels to the model. If an explicit counter-evidence source is available, the highest-ranked counter is reserved before relevance fill; when explicit support is also available, the highest-ranked support is reserved as well. Unmarked or conflicting duplicate labels remain unclassified; the audit does not claim that the pipeline identified a counterexample. The output contract separately allows up to three supporting and three contrary citations, without requiring either output list to be nonempty.

Full-text inputs can be supplied as sections and paragraphs. Selected passages retain adjacent paragraphs; table blocks keep their headers, units, qualifiers, and rows together. Every body citation locator carries its section, paragraph, and page where supplied. Missing bodies remain `title_only`. Structured financial facts appear once, with report period and field names, rather than being duplicated in an excerpt. Role prompts share concise source/safety rules, while prior-role input accepts only bounded structured summaries and public citations.

`max_context_chars` bounds serialized evidence cards only. A separate context-token bound counts the UTF-8 bytes of the complete serialized `messages` array, including system rules, summaries, evidence, schema, and generation budget, then adds 128 framing-reserve bytes as a conservative input-token upper bound. The configured 600-token output reserve is added to that bound, and the combined amount must fit `max_context_token_upper_bound` (8,600 by default). This is a conservative bound, not an exact tokenizer count. If the complete context does not fit, the builder drops lowest-ranked unreserved evidence first while preserving at least one citable source, then drops summaries from the end of the caller-supplied priority order. It never silently truncates text and fails explicitly if the reserved evidence and output still cannot fit.

## Local preview

```powershell
.venv\Scripts\python.exe scripts\prepare_agentic_rl_qwen.py preview --event-id <existing-event-id> --output .tmp\agentic_rl_20261021_review\prompt_context_v3_preview.json
```

This command is offline. Its prompt/schema versions are `raama_public_evidence_context_v3_20261022` and `public_evidence_context_v3`; it does not alter the approved v2 manifest, hash, responses, or frozen artifacts. The recorded local demonstration reuses one already-approved title-only event. It does not exercise chapter retrieval because no archived announcement body was available. The focused synthetic unit fixture checks reserved contrary evidence, section/table locators, adjacency, and atomic units; that fixture is not a source-document demonstration. Neither is evidence of a provider call or policy performance.

No additional data, API calls, or full historical replay are required by this preview path.
