from langchain_core.documents import Document

MAX_CHARS = 600
OVERLAP_CHARS = 80


class Chunker:
    def _build_prefix(self, metadata: dict) -> str:
        """Build a prefix string from metadata hierarchy (law_id | chapter | section | article)."""
        parts = [metadata.get("law_id", "")]
        if metadata.get("chapter"):
            parts.append(metadata["chapter"])
        if metadata.get("section"):
            parts.append(metadata["section"])
        if metadata.get("article"):
            parts.append(metadata["article"])
        return " | ".join(p for p in parts if p)

    def _split_by_length(self, text: str) -> list[str]:
        """Split text into segments under MAX_CHARS at paragraph then word boundaries."""
        paragraphs = text.split("\n")
        segments = []
        current = ""

        for para in paragraphs:
            if len(para) > MAX_CHARS:
                if current:
                    segments.append(current.strip())
                    current = ""
                while len(para) > MAX_CHARS:
                    cut = para.rfind(" ", 0, MAX_CHARS)
                    if cut == -1:
                        cut = MAX_CHARS
                    segments.append(para[:cut].strip())
                    # carry overlap into next segment for context continuity
                    para = para[max(0, cut - OVERLAP_CHARS) :].strip()
                current = para
                continue

            candidate = (current + "\n" + para).strip() if current else para
            if len(candidate) > MAX_CHARS:
                segments.append(current.strip())
                # start next segment with overlap from end of current
                overlap = (
                    current[-OVERLAP_CHARS:].strip() if len(current) > OVERLAP_CHARS else current
                )
                current = (overlap + "\n" + para).strip()
            else:
                current = candidate

        if current.strip():
            segments.append(current.strip())

        return segments

    def _chunk_doc(self, doc: Document) -> list[Document]:
        """Chunk a single document into segments with metadata and overlap for continuity."""
        prefix = self._build_prefix(doc.metadata)

        if len(doc.page_content) <= MAX_CHARS:
            return [
                Document(
                    page_content=f"{prefix}\n{doc.page_content}" if prefix else doc.page_content,
                    metadata={**doc.metadata, "chunk_index": 0, "total_chunks": 1},
                )
            ]

        segments = self._split_by_length(doc.page_content)
        return [
            Document(
                page_content=f"{prefix}\n{segment}" if prefix else segment,
                metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(segments)},
            )
            for i, segment in enumerate(segments)
        ]

    def create_chunks(self, raw_docs: list[Document]) -> list[Document]:
        """Chunk a list of documents into smaller segments for embedding."""
        chunks = []
        for doc in raw_docs:
            chunks.extend(self._chunk_doc(doc))
        return chunks
