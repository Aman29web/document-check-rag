from pathlib import Path
import uuid

import chromadb
import ollama

# Local Ollama model that turns text into embeddings (lists of numbers)
EMBED_MODEL = "nomic-embed-text"

# Each chunk is about this many words, and repeats a few words from
# the previous chunk so sentences cut in half are not lost
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# ChromaDB saves its data in the chroma_db folder next to this file,
# so uploaded documents are still there after a restart
chroma_client = chromadb.PersistentClient(path=str(Path(__file__).parent / "chroma_db"))
collection = chroma_client.get_or_create_collection(
    name="documents",
    configuration={"hnsw": {"space": "cosine"}},
    embedding_function=None
)

ollama_client = ollama.AsyncClient()


def split_into_chunks(text):
    # Build chunks from whole lines and keep the line breaks, so things that
    # belong together (like a company name and its dates) stay readable
    lines = []
    for line in text.splitlines():
        words = line.split()
        # A very long line (like a big Word paragraph) is cut into smaller pieces
        for start in range(0, len(words), CHUNK_SIZE):
            lines.append(" ".join(words[start:start + CHUNK_SIZE]))

    chunks = []
    current_lines = []
    word_count = 0

    for line in lines:
        line_words = len(line.split())

        # This line would make the chunk too big, so finish the chunk first
        if current_lines and word_count + line_words > CHUNK_SIZE:
            chunks.append("\n".join(current_lines))

            # Start the next chunk with the last few lines (about CHUNK_OVERLAP words)
            overlap_lines = []
            overlap_count = 0
            for previous in reversed(current_lines):
                if overlap_count + len(previous.split()) > CHUNK_OVERLAP:
                    break
                overlap_lines.insert(0, previous)
                overlap_count += len(previous.split())
            current_lines = overlap_lines
            word_count = overlap_count

        current_lines.append(line)
        word_count += line_words

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks


async def embed(texts, prefix):
    # nomic-embed-text works best when told whether the text is a
    # document being stored or a question being searched for
    embeddings = []

    # Send in batches so very large documents do not overload Ollama
    for start in range(0, len(texts), 32):
        batch = [prefix + text for text in texts[start:start + 32]]
        response = await ollama_client.embed(model=EMBED_MODEL, input=batch)
        embeddings.extend(response.embeddings)

    return embeddings


def find_document_by_hash(file_hash):
    # Returns the document_id if this exact file was uploaded before
    found = collection.get(where={"file_hash": file_hash}, limit=1, include=["metadatas"])
    if found["metadatas"]:
        return found["metadatas"][0]["document_id"]
    return None


async def add_document(filename, pages, file_hash):
    # pages is a list of (page_number, text); page_number is None for Word files
    document_id = str(uuid.uuid4())
    chunk_texts = []
    metadatas = []

    for page_number, page_text in pages:
        for chunk in split_into_chunks(page_text):
            chunk_texts.append(chunk)
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "file_hash": file_hash
            }
            if page_number is not None:
                metadata["page"] = page_number
            metadatas.append(metadata)

    embeddings = await embed(chunk_texts, "search_document: ")

    collection.add(
        ids=[f"{document_id}-{i}" for i in range(len(chunk_texts))],
        embeddings=embeddings,
        documents=chunk_texts,
        metadatas=metadatas
    )

    return document_id, len(chunk_texts)


def list_documents():
    documents = {}

    for metadata in collection.get(include=["metadatas"])["metadatas"]:
        document_id = metadata["document_id"]
        if document_id not in documents:
            documents[document_id] = {
                "document_id": document_id,
                "filename": metadata["filename"],
                "chunks": 0
            }
        documents[document_id]["chunks"] += 1

    return list(documents.values())


def delete_document(document_id):
    found = collection.get(where={"document_id": document_id}, include=[])["ids"]
    if found:
        collection.delete(ids=found)
    return len(found) > 0


async def search(question, document_id=None, top_k=5):
    question_embedding = await embed([question], "search_query: ")

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        where={"document_id": document_id} if document_id else None
    )

    matches = []
    for text, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        matches.append({
            "filename": metadata["filename"],
            "document_id": metadata["document_id"],
            "page": metadata.get("page"),
            "similarity": round(1 - distance, 3),
            "text": text
        })

    return matches
