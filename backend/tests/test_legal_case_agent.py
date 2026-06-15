import json

from langchain_core.documents import Document

from agents.legal_case_agent import LegalCaseAgent


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            return FakeMessage(
                json.dumps(
                    {
                        "issues": ["Валидност на договора", "Право на обезщетение"],
                        "queries": ["валидност на договор", "обезщетение при неизпълнение"],
                    },
                    ensure_ascii=False,
                )
            )
        return FakeMessage("Правен анализ [Чл. 1, Тестов закон]")


class FakeRetriever:
    def __init__(self):
        self.queries = []
        self.document = Document(
            page_content="Нормативен текст",
            metadata={
                "source": "law.json",
                "law_id": "Тестов закон",
                "chapter": "Глава I",
                "article": "Чл. 1",
            },
        )

    def search(self, query, **kwargs):
        self.queries.append(query)
        return [self.document], [0.9]


def test_agent_runs_multiple_searches_and_deduplicates_documents():
    retriever = FakeRetriever()
    agent = LegalCaseAgent(retriever, FakeLLM())

    result = agent.analyze(
        case_text="Страните спорят по договор и претендират обезщетение.",
        top_n_per_search=5,
        max_searches=4,
        score_threshold=0.4,
        min_docs=1,
    )

    assert len(retriever.queries) >= 2
    assert len(result["documents"]) == 1
    assert result["issues"] == ["Валидност на договора", "Право на обезщетение"]
    assert result["analysis"].startswith("Правен анализ")
    assert all("query" in step for step in result["search_trace"])


def test_agent_falls_back_when_planner_output_is_not_json():
    class InvalidPlannerLLM(FakeLLM):
        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                return FakeMessage("not json")
            return FakeMessage("Анализ")

    retriever = FakeRetriever()
    result = LegalCaseAgent(retriever, InvalidPlannerLLM()).analyze(
        case_text="Подробен правен казус, който следва да бъде анализиран.",
        top_n_per_search=3,
        max_searches=3,
        score_threshold=0.4,
        min_docs=1,
    )

    assert result["issues"] == ["Правна квалификация и приложими норми"]
    assert retriever.queries[0].startswith("Подробен правен казус")
