import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

_ARTICLE_REFERENCE_RE = re.compile(
    r"\bчл\.\s*\d+[а-я]?(?:\s*,?\s*ал\.\s*\d+[а-я]?)?", re.IGNORECASE
)

PLANNER_PROMPT = """You plan retrieval for a Bulgarian legal case.
Use only the statutes available to the retrieval system. Break the case into distinct
legal issues and produce focused Bulgarian search queries. Include queries for legal
elements, exceptions, remedies, deadlines, and burden of proof when relevant.

Return JSON only:
{{"issues":["..."],"queries":["..."]}}

Rules:
- Return between 2 and {max_queries} queries.
- Do not answer the case.
- Do not invent article numbers.
- Keep every query self-contained and specific.

Case:
{case_text}
"""

ANALYSIS_PROMPT = """You are a Bulgarian legal analyst. Analyze the case using ONLY the
provided legal provisions. Do not use prior legal knowledge and do not invent facts,
article numbers, procedures, deadlines, or legal consequences.

Write in the language of the case. Structure the answer as:
1. Факти и допускания
2. Правни въпроси
3. Приложими норми
4. Правен анализ
5. Извод и практически следващи стъпки

For every legal claim cite the supporting provision immediately as
`[<article>, <law_id>]`. Clearly distinguish facts from assumptions and present
material counterarguments or alternative outcomes. If the supplied provisions are
insufficient, state exactly what is missing under the relevant heading instead of
guessing. Do not reveal hidden chain-of-thought; provide concise legal reasoning and
the evidentiary basis for each conclusion.

Case:
{case_text}

Identified issues:
{issues}

Retrieved provisions:
{context}
"""


@dataclass(frozen=True)
class SearchStep:
    query: str
    documents_found: int
    new_documents: int


class LegalCaseAgent:
    """Multi-step legal case analysis over the existing hybrid retriever."""

    def __init__(self, retriever: Any, llm: Any):
        self._retriever = retriever
        self._llm = llm

    def analyze(
        self,
        case_text: str,
        top_n_per_search: int,
        max_searches: int,
        score_threshold: float,
        min_docs: int,
    ) -> dict[str, Any]:
        plan = self._plan(case_text, max_searches)
        queries = self._augment_queries(plan["queries"], case_text, max_searches)

        unique_docs: dict[str, tuple[Document, float]] = {}
        trace: list[SearchStep] = []
        searched: set[str] = set()

        for query in queries:
            if query.casefold() in searched:
                continue
            searched.add(query.casefold())
            documents, scores = self._retriever.search(
                query=query,
                top_n=top_n_per_search,
                score_threshold=score_threshold,
                min_docs=min_docs,
            )
            before = len(unique_docs)
            for document, score in zip(documents, scores, strict=False):
                key = self._document_key(document)
                existing = unique_docs.get(key)
                if existing is None or score > existing[1]:
                    unique_docs[key] = (document, score)
            trace.append(
                SearchStep(
                    query=query,
                    documents_found=len(documents),
                    new_documents=len(unique_docs) - before,
                )
            )

        ranked = sorted(unique_docs.values(), key=lambda item: item[1], reverse=True)
        documents = [item[0] for item in ranked]
        scores = [item[1] for item in ranked]

        analysis = self._synthesize(case_text, plan["issues"], documents)
        return {
            "analysis": analysis,
            "issues": plan["issues"],
            "search_trace": [
                {
                    "step": index,
                    "query": step.query,
                    "documents_found": step.documents_found,
                    "new_documents": step.new_documents,
                }
                for index, step in enumerate(trace, 1)
            ],
            "documents": documents,
            "scores": scores,
        }

    def _plan(self, case_text: str, max_queries: int) -> dict[str, list[str]]:
        prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
        response = self._llm.invoke(
            prompt.format_messages(case_text=case_text, max_queries=max_queries)
        )
        content = self._message_text(response)
        try:
            payload = json.loads(self._extract_json(content))
            issues = self._clean_strings(payload.get("issues", []))
            queries = self._clean_strings(payload.get("queries", []))
        except (json.JSONDecodeError, TypeError, AttributeError):
            logger.warning("Legal agent planner returned invalid JSON; using fallback query")
            issues, queries = [], []

        return {
            "issues": issues or ["Правна квалификация и приложими норми"],
            "queries": queries[:max_queries],
        }

    def _augment_queries(
        self, planned_queries: list[str], case_text: str, max_searches: int
    ) -> list[str]:
        queries = list(planned_queries)
        if not queries:
            queries.append(case_text)

        references = _ARTICLE_REFERENCE_RE.findall(case_text)
        for reference in references:
            queries.append(f"{reference} приложимост, предпоставки и последици")

        queries.append(f"изключения, срокове, защита и правни последици: {case_text}")
        return self._deduplicate(queries)[:max_searches]

    def _synthesize(
        self, case_text: str, issues: list[str], documents: list[Document]
    ) -> str:
        if not documents:
            return (
                "INSUFFICIENT_CONTEXT: Не са открити правни норми, достатъчни за "
                "анализ на казуса."
            )

        context = "\n\n".join(self._format_document(document) for document in documents)
        prompt = ChatPromptTemplate.from_template(ANALYSIS_PROMPT)
        response = self._llm.invoke(
            prompt.format_messages(
                case_text=case_text,
                issues="\n".join(f"- {issue}" for issue in issues),
                context=context,
            )
        )
        return self._message_text(response).strip()

    @staticmethod
    def _format_document(document: Document) -> str:
        metadata = document.metadata
        return (
            f"{metadata.get('law_id', '')} | {metadata.get('chapter', '')} | "
            f"{metadata.get('article', '')}\n{document.page_content}"
        )

    @staticmethod
    def _document_key(document: Document) -> str:
        metadata = document.metadata
        return "|".join(
            [
                str(metadata.get("source", "")),
                str(metadata.get("article", "")),
                document.page_content,
            ]
        )

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            return "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    @staticmethod
    def _extract_json(text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text

    @staticmethod
    def _clean_strings(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        return [str(value).strip() for value in values if str(value).strip()]

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for value in values:
            key = value.casefold().strip()
            if key and key not in seen:
                seen.add(key)
                result.append(value.strip())
        return result
