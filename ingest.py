"""
Ingestão do CSV de tábua de risco safra 2026-2027.
Gera embeddings via NVIDIA API e armazena no ChromaDB.
"""

import os
import pandas as pd
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
COLLECTION_NAME = "tabua_risco_safra"
CSV_PATH = "dados-abertos-tabua-de-risco-safra-2026-2027.csv"
BATCH_SIZE = 50


def build_row_text(row: pd.Series) -> str:
    """Converte uma linha do CSV em texto descritivo para embedding."""
    decades = []
    for i in range(1, 37):
        val = row.get(f"dec{i}", 0)
        if pd.notna(val) and int(val) > 0:
            decades.append(f"décade {i}: {int(val)}%")

    decade_text = (
        ", ".join(decades) if decades else "sem risco identificado nas décadas"
    )

    return (
        f"Cultura: {row['Nome_cultura']}. "
        f"Safra: {row['SafraIni']}/{row['SafraFin']}. "
        f"Município: {row['municipio']} - {row['UF']}. "
        f"Clima: {row['Nome_Clima']}. "
        f"Manejo: {row['Nome_Outros_Manejos']}. "
        f"Portaria: {row['Portaria']}. "
        f"Risco por décade: {decade_text}."
    )


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        extra_body={"input_type": "passage", "truncate": "END"},
    )
    return [item.embedding for item in response.data]


def ingest():
    print("Carregando CSV...")
    df = pd.read_csv(CSV_PATH, sep=";")
    print(f"  {len(df)} linhas carregadas. Colunas: {len(df.columns)}")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )

    chroma = chromadb.PersistentClient(path="./chroma_db")

    # Recria a coleção para garantir ingestão limpa
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(COLLECTION_NAME)

    texts = [build_row_text(row) for _, row in df.iterrows()]
    ids = [str(i) for i in range(len(df))]
    metadatas = df[
        ["Nome_cultura", "municipio", "UF", "SafraIni", "SafraFin",
         "Nome_Outros_Manejos", "Nome_Clima", "Portaria"]
    ].to_dict(orient="records")

    # Converte campos numéricos para string (ChromaDB só aceita str/int/float/bool)
    for m in metadatas:
        for k, v in m.items():
            if pd.isna(v):
                m[k] = ""
            elif isinstance(v, float):
                m[k] = int(v) if v.is_integer() else v

    print(f"\nGerando embeddings em lotes de {BATCH_SIZE}...")
    all_embeddings = []
    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch = texts[start : start + BATCH_SIZE]
        embeddings = get_embeddings(client, batch)
        all_embeddings.extend(embeddings)

    print("\nArmazenando no ChromaDB...")
    for start in tqdm(range(0, len(texts), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(texts))
        collection.add(
            ids=ids[start:end],
            documents=texts[start:end],
            embeddings=all_embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"\nIngestão concluída! {len(texts)} registros no ChromaDB.")


if __name__ == "__main__":
    ingest()
