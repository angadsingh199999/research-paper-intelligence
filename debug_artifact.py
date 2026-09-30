import re
from app.ingestion.ingest import ingest_paper
from app.indexing.chunker import create_chunks

paper = ingest_paper('data/raw_papers/paper1.pdf')
chunks = create_chunks(paper)

for chunk in chunks:
    if chunk.chunk_id == 'paper1_chunk_0009':
        text = chunk.text
        m = re.search(r'journal.{0,10}of.{0,10}innovation', text, re.IGNORECASE | re.DOTALL)
        if m:
            start = max(0, m.start() - 40)
            end = m.end() + 40
            snippet = text[start:end]
            print('SNIPPET:', repr(snippet))
            print('CODEPOINTS:')
            for ch in snippet:
                label = repr(ch)
                print("  {:>6}  U+{:04X}".format(label, ord(ch)))
        else:
            print('No match even with loose regex -- artifact may not be in this chunk at all.')