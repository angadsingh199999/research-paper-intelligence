class ContextBuilder:

    def __init__(self, max_chunks=5):
        self.max_chunks = max_chunks

    def build(self, results):
        """
        Convert retrieved chunks into a clean context
        that can be given to the LLM.
        """

        selected_results = results[:self.max_chunks]

        context_parts = []

        for index, result in enumerate(
            selected_results,
            start=1
        ):

            paper_id = result.get(
                "paper_id",
                ""
            )

            paper_title = result.get(
                "paper_title",
                ""
            )

            section = result.get(
                "section",
                ""
            )

            page_start = result.get(
                "page_start",
                0
            )

            page_end = result.get(
                "page_end",
                0
            )

            chunk_id = result.get(
                "chunk_id",
                ""
            )

            text = result.get(
                "text",
                ""
            )

            context_parts.append(
                f"""
EVIDENCE {index}

Paper ID: {paper_id}

Paper Title: {paper_title}

Section: {section}

Pages: {page_start}-{page_end}

Chunk ID: {chunk_id}

Text:
{text}
""".strip()
            )

        return "\n\n" + (
            "\n\n----------------------------------------\n\n"
        ).join(context_parts)