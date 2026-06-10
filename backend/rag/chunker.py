from langchain_core.documents import Document

from config import CHUNK_MAX_CHARS as MAX_CHARS, CHUNK_OVERLAP_CHARS as OVERLAP_CHARS


class Chunker:
    def _split_long_paragraph(self, para: str) -> list[str]:
        """Split a single paragraph that is longer than MAX_CHARS into segments
        at word boundaries, carrying OVERLAP_CHARS of context between them."""
        out = []
        while len(para) > MAX_CHARS:
            cut = para.rfind(" ", 0, MAX_CHARS)
            if cut == -1:
                cut = MAX_CHARS
            out.append(para[:cut].strip())
            # carry overlap; if cut <= OVERLAP_CHARS the overlap would wrap back to 0
            # and cause an infinite loop — advance by cut in that case
            start = cut - OVERLAP_CHARS if cut > OVERLAP_CHARS else cut
            para = para[start:].strip()
        if para:
            out.append(para)
        return out

    def _split_by_length(self, text: str) -> list[str]:
        """Split text into segments under MAX_CHARS at paragraph then word boundaries."""
        paragraphs = text.split("\n")
        segments: list[str] = []
        current = ""

        for para in paragraphs:
            if len(para) > MAX_CHARS:
                if current:
                    segments.append(current.strip())
                    current = ""
                segments.extend(self._split_long_paragraph(para))
                continue

            candidate = (current + "\n" + para).strip() if current else para
            if len(candidate) > MAX_CHARS:
                segments.append(current.strip())
                overlap = (
                    current[-OVERLAP_CHARS:].strip() if len(current) > OVERLAP_CHARS else current
                )
                merged = (overlap + "\n" + para).strip() if overlap else para
                # If even the overlap+para combo overshoots, fall back to splitting
                # the paragraph alone — overlap is best-effort, not mandatory.
                if len(merged) > MAX_CHARS:
                    pieces = self._split_long_paragraph(merged)
                    segments.extend(pieces[:-1])
                    current = pieces[-1] if pieces else ""
                else:
                    current = merged
            else:
                current = candidate

        if current.strip():
            segments.append(current.strip())

        return segments

    def _chunk_doc(self, doc: Document) -> list[Document]:
        """Chunk a single document into segments with metadata for context continuity."""
        if len(doc.page_content) <= MAX_CHARS:
            return [
                Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "chunk_index": 0, "total_chunks": 1},
                )
            ]

        segments = self._split_by_length(doc.page_content)
        return [
            Document(
                page_content=segment,
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
