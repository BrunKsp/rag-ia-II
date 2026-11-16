"""
RF01: Pipeline de ingestão completo.

Etapas:
  1. Leitura e limpeza do CSV
  2. Agrupamento por (cultura, UF) → documentos
  3. Segmentação em chunks (chunk_size=600, chunk_overlap=80)
  4. Geração de embeddings via NVIDIA API
  5. Armazenamento no ChromaDB com metadados (source, topic, uf, etc.)

Justificativa de chunking:
  - chunk_size=600 tokens: captura ~3-5 municípios de uma cultura+estado por chunk,
    mantendo contexto regional suficiente sem exceder o limite do modelo de embedding.
  - chunk_overlap=80 tokens: garante que a transição entre chunks preserve o contexto
    da portaria e cultura para buscas de similaridade.
"""

import asyncio
import os
import pandas as pd
import chromadb
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = "nvidia/nv-embedqa-mistral-7b-v2"
COLLECTION_NAME = "tabua_risco_safra"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_BASE_DIR, "arquivos", "dados-abertos-tabua-de-risco-safra-2026-2027.csv")
CHROMA_PATH = os.path.join(_BASE_DIR, "chroma_db")
BATCH_SIZE = 10   # modelo 7B aceita lotes menores
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80
# Limita municípios por documento para manter o corpus gerenciável.
# Com 1.1M linhas, amostrar 300 por (cultura, UF) ainda representa bem a diversidade regional.
MAX_ROWS_PER_DOC = int(os.getenv("MAX_ROWS_PER_DOC", "300"))


def _row_to_text(row: pd.Series) -> str:
    decades = []
    for i in range(1, 37):
        val = row.get(f"dec{i}", 0)
        if pd.notna(val) and int(val) > 0:
            decades.append(f"dec{i}={int(val)}%")
    decade_str = " ".join(decades) if decades else "sem risco declarado"

    return (
        f"Município: {row['municipio']} ({row['UF']}). "
        f"Clima: {row['Nome_Clima']}. "
        f"Solo: {row['Cod_Solo']}. Ciclo: {row['Cod_Ciclo']}. "
        f"Manejo: {row['Nome_Outros_Manejos']}. "
        f"Portaria: {row['Portaria']}. "
        f"Risco por décade: {decade_str}."
    )


def _build_document(group: pd.DataFrame, cultura: str, uf: str) -> str:
    header = (
        f"Tábua de Risco Safra 2026-2027 — Cultura: {cultura} — Estado: {uf}.\n"
        f"Registros disponíveis: {len(group)}.\n\n"
    )
    rows_text = "\n".join(_row_to_text(row) for _, row in group.iterrows())
    return header + rows_text


def _chunk_text(text: str) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)

    if len(tokens) <= CHUNK_SIZE:
        return [text]

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def _get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        extra_body={"input_type": "passage", "truncate": "END"},
    )
    return [item.embedding for item in response.data]


def ingest():
    print("=" * 60)
    print("  Ingestão — Tábua de Risco Safra 2026-2027")
    print("=" * 60)

    print("\n[1/5] Carregando e limpando CSV...")
    df = pd.read_csv(CSV_PATH, sep=";")
    df = df.dropna(subset=["Nome_cultura", "municipio", "UF"])
    df["Nome_cultura"] = df["Nome_cultura"].str.strip()
    df["municipio"] = df["municipio"].str.strip()
    df["UF"] = df["UF"].str.strip()

    dec_cols = [f"dec{i}" for i in range(1, 37)]
    df[dec_cols] = df[dec_cols].fillna(0).astype(int)

    n_culturas = df["Nome_cultura"].nunique()
    n_estados = df["UF"].nunique()
    print(f"     {len(df)} registros | {n_culturas} culturas | {n_estados} estados")

    print(f"\n[2/5] Agrupando em documentos (cultura × estado)...")
    groups_raw = list(df.groupby(["Nome_cultura", "UF"]))
    # Amostragem: limita municípios por documento para tornar a ingestão viável
    groups = [(key, grp.sample(min(len(grp), MAX_ROWS_PER_DOC), random_state=42)) for key, grp in groups_raw]
    total_sampled = sum(len(g) for _, g in groups)
    print(f"     {len(groups)} documentos | {total_sampled} registros amostrados (máx {MAX_ROWS_PER_DOC}/doc)")

    print(f"\n[3/5] Segmentando em chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []

    for (cultura, uf), group in tqdm(groups):
        doc_text = _build_document(group, cultura, uf)
        doc_id = f"{cultura}_{uf}".replace(" ", "_")
        chunks = _chunk_text(doc_text)
        portaria = str(group["Portaria"].iloc[0]) if not group.empty else ""

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "doc_id": doc_id,
                "topic": cultura,       # RF01: topic key
                "uf": uf,
                "source": portaria,     # RF01: documento de origem
                "portaria": portaria,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "municipios_count": len(group),
            })

    print(f"     {len(all_chunks)} chunks gerados")

    print(f"\n[4/5] Gerando embeddings com {EMBEDDING_MODEL}...")
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )

    all_embeddings: list[list[float]] = []
    for start in tqdm(range(0, len(all_chunks), BATCH_SIZE)):
        batch_texts = [c["text"] for c in all_chunks[start : start + BATCH_SIZE]]
        all_embeddings.extend(_get_embeddings(client, batch_texts))

    print(f"\n[5/5] Armazenando no ChromaDB...")
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in all_chunks]
    ids = [f"{c['doc_id']}_c{c['chunk_index']}" for c in all_chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in all_chunks]

    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(texts))
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=all_embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"\n[+] Salvando embeddings no MongoDB (paralelo)...")
    try:
        import mongo_store
        mongo_docs = [
            {
                "chunk_id": ids[i],
                "doc_id": all_chunks[i]["doc_id"],
                "topic": all_chunks[i]["topic"],
                "uf": all_chunks[i]["uf"],
                "text": all_chunks[i]["text"],
                "embedding": all_embeddings[i],
                "metadata": {k: v for k, v in all_chunks[i].items() if k not in ("text",)},
            }
            for i in range(len(all_chunks))
        ]
        saved = asyncio.run(mongo_store.save_embeddings(mongo_docs))
        print(f"     {saved} embeddings salvos no MongoDB")
    except Exception as e:
        print(f"     MongoDB ignorado: {e}")

    print(f"\n✓ Ingestão concluída!")
    print(f"  {len(all_chunks)} chunks | {len(groups)} documentos | {n_culturas} culturas | {n_estados} estados")


if __name__ == "__main__":
    ingest()
