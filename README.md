# RAG - Tábua de Risco Safra 2026-2027

Projeto de Retrieval-Augmented Generation (RAG) sobre os dados abertos da tábua de risco safra 2026-2027, usando modelos gratuitos da NVIDIA.

## Modelos utilizados

| Papel | Modelo NVIDIA |
|-------|--------------|
| Embeddings | `nvidia/nv-embedqa-e5-v5` |
| LLM (geração) | `meta/llama-3.3-70b-instruct` |

Ambos disponíveis gratuitamente em https://build.nvidia.com/

## Setup

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar API Key

```bash
cp .env.example .env
# Edite .env e adicione sua NVIDIA_API_KEY
```

Obtenha sua chave em: https://build.nvidia.com/

### 3. Ingestão (gerar embeddings e indexar)

```bash
python ingest.py
```

Isso vai:
- Ler o CSV com ~milhares de registros
- Converter cada linha em texto descritivo
- Gerar embeddings via NVIDIA API
- Armazenar localmente no ChromaDB (`./chroma_db/`)

> A ingestão pode demorar alguns minutos dependendo do tamanho do CSV.

### 4. Consultar

**Interface interativa:**
```bash
python main.py
```

**Query direta:**
```bash
python query.py "Qual o risco para soja em Goiás irrigado?"
```

## Exemplos de perguntas

- Qual o risco para milho em municípios de MG?
- Quais culturas têm risco na décade 11 em SP?
- Qual a diferença de risco entre irrigado e sequeiro para cevada?
- Quais municípios de GO têm menor risco para plantio?

## Estrutura

```
rag-ia/
├── dados-abertos-tabua-de-risco-safra-2026-2027.csv
├── ingest.py      # Lê CSV, gera embeddings, salva no ChromaDB
├── query.py       # Busca contexto e gera resposta com LLM
├── main.py        # CLI interativa
├── requirements.txt
├── .env.example
└── chroma_db/     # Criado após ingest.py (não versionar)
```
