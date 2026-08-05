# AI intelligence architecture

Phase 3 introduces an AI subsystem that consumes only the universal `Job` model.
ATS providers and scraping code neither import nor configure AI components.

## Pipeline and boundaries

```text
universal Job
  -> deterministic JobDocument
  -> EmbeddingProvider
  -> VectorSimilaritySearch
  -> JobAnalyzer / LLMProvider
  -> RankingEngine
  -> Recommendation
```

Each arrow crosses an abstract interface. The composition root in `bootstrap.py`
selects concrete implementations from environment configuration. A future hosted
embedding service, vector database, analysis implementation, or ranking policy can
replace its port without changing providers or aggregation.

## Embeddings and semantic retrieval

`BGEEmbeddingProvider` uses
[`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) through
Sentence Transformers. It follows the model card's asymmetric retrieval approach:
job documents are encoded as passages and semantic profiles use the recommended
query instruction. The model loads lazily and is downloaded by Sentence
Transformers on the first inference if it is not already present locally.

`CachedEmbeddingProvider` keys each vector by normalized text, model identity, and
`AI_CACHE_VERSION`. It batches only missing inputs. `CosineSimilaritySearch`
compares job vectors with distinct responsibility-based label profiles. It also
compares internship against permanent employment and hands-on technical work
against nontechnical commercial, legal, and customer work. Both relative margins
must pass before LLM analysis or ranking. Computed scores are cached separately.
`VectorSimilaritySearch` is the seam for a future vector database.

Labels are configured using `AI_LABELS`; they are deliberately strings rather than
an enum. A new label therefore does not require an application release.

## Structured LLM analysis

Every adapter implements `LLMProvider.generate(StructuredRequest)`. Provider
responses are converted to the same strict Pydantic `JobAnalysis`, which rejects
extra fields, out-of-range scores, malformed JSON, blank reasoning, and labels
outside the configured catalog. Invalid output is cached as invalid, preventing an
identical malformed inference from being repeated. Secrets are never placed in
prompts, logs, cache keys, or settings representations.

The schema includes explicit `is_internship` and `is_technical` decisions. The
pipeline rejects false values, low confidence, or low internship relevance. The
prompt instructs providers to ignore employer boilerplate and select no more than
three responsibility-based labels.

Supported adapters use documented public APIs:

| Provider | Endpoint / structured-output mechanism |
| --- | --- |
| OpenAI | [Responses API `text.format`](https://platform.openai.com/docs/api-reference/responses/create) |
| Anthropic | [Messages `output_config.format`](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) |
| Gemini | [`generateContent` JSON Schema](https://ai.google.dev/gemini-api/docs/structured-output) |
| OpenRouter | [Chat-completions `response_format`](https://openrouter.ai/docs/guides/features/structured-outputs) |
| Ollama | [Local chat `format` schema](https://docs.ollama.com/capabilities/structured-outputs) |

Select a provider entirely through configuration:

```dotenv
LLM_PROVIDER=anthropic
LLM_MODEL=your-supported-model-id
ANTHROPIC_API_KEY=your-key
```

Remote adapters accept `LLM_API_KEY` as a common override or their standard key:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or
`OPENROUTER_API_KEY`. Ollama needs no key and defaults to
`http://localhost:11434`; `LLM_BASE_URL` can override any endpoint. Model names are
configuration because availability changes independently of this application. The
current OpenAI default, when OpenAI is selected and `LLM_MODEL` is blank, is
`gpt-5.6-sol`.

## Cache lifecycle

`SQLiteInferenceCache` persistently stores three independent namespaces:

- document and query embeddings;
- semantic similarity results;
- valid or malformed LLM analysis outputs.

Exact version matches never regenerate. Increment `AI_CACHE_VERSION` whenever the
schema, prompt policy, or desired inference behavior changes. This logically
invalidates old rows immediately. The cache port also exposes namespace-specific
or complete physical invalidation for administrative tooling in a later phase.

## Ranking and fallback

`WeightedRankingEngine` combines semantic similarity, LLM confidence, AI
relevance, and internship relevance. Every weight and the final recommendation
threshold are environment-configurable. With `LLM_PROVIDER=none`, semantic values
still produce rankings and no remote inference occurs.

Keyword rules are not consulted alongside healthy AI results. They are retained
only by `KeywordFallbackRecommender`, used when AI is explicitly disabled or an
embedding/LLM/cache stage raises an expected intelligence error. This keeps a
recoverable scan useful without allowing keywords to override a successful
semantic decision.

## Live quality evaluation

`python main.py evaluate --minimum-accuracy 0.92` runs the configured real LLM
against the packaged 32-case balanced corpus. It reports binary
technical-internship accuracy, multi-label precision/recall/F1, and individual
failures. The command rejects semantic-only configuration and exits nonzero below
the accuracy gate. Unit tests use mocks and are intentionally not accepted as
evidence for a live quality percentage.
