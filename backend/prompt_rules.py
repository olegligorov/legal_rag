"""Shared prompt fragments used by the per-step grounding prompt and the agent.

These are the rules that must hold whether the LLM is the single-shot
generator (``backend/templates.py``) or the multi-hop agent's reasoner /
synthesizer (``backend/agent/prompts.py``). Keeping them in one place
prevents drift — if a thesis evaluation reveals a needed rule change,
both prompts pick it up.

Each constant is a self-contained block of markdown-formatted prose.
Compose them by f-string interpolation into the parent prompt.
"""

LANGUAGE_RULE = """- Detect the language of the user's question. Answer in the same language (Bulgarian or English).
- Quote legal text verbatim from the source — never translate, paraphrase, or summarize quoted statute text."""


CITATION_RULE = """- Cite every legal claim inline as `[<article>, <law_id>]` using the exact values that appear in the source headers (e.g. `[Чл. 128, Кодекс на труда]`).
- Place the citation immediately after the claim it supports, not bundled at the end.
- A claim covered by multiple articles gets multiple bracketed citations.
- Do not invent, renumber, paraphrase, or drop citations. Reproduce them verbatim."""


ANSWER_SHAPE_RULE = """- Lead with the direct answer in 1–3 sentences.
- If the question requires elaboration (procedure, conditions, exceptions), follow with a short bulleted list. Each bullet ends with its citation.
- No greetings, apologies, or meta-commentary."""


GROUNDING_RULE = """- Use ONLY the articles provided in the context. Do not rely on prior knowledge of Bulgarian law.
- Do not invent article numbers, chapters, or legal text. If an article is referenced inside the context but its full text is not present, treat that reference as unverified.
- If the context does not contain enough information to answer, respond with exactly:
  `INSUFFICIENT_CONTEXT: <one short sentence in the question's language describing what is missing>`
  Do not guess, do not partially answer, do not suggest the user rephrase."""
