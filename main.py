"""
Interface CLI interativa para o RAG de tábua de risco safra.
"""

import os
import sys
import chromadb
from dotenv import load_dotenv

load_dotenv()


def check_setup():
    if not os.getenv("NVIDIA_API_KEY"):
        print("ERRO: NVIDIA_API_KEY não definida. Copie .env.example para .env e adicione sua chave.")
        print("Obtenha em: https://build.nvidia.com/")
        sys.exit(1)

    if not os.path.exists("./chroma_db"):
        print("Base de dados não encontrada. Execute primeiro:\n  python ingest.py")
        sys.exit(1)

    try:
        chroma = chromadb.PersistentClient(path="./chroma_db")
        col = chroma.get_collection("tabua_risco_safra")
        count = col.count()
        print(f"Base carregada: {count} registros indexados.")
    except Exception:
        print("Coleção não encontrada. Execute:\n  python ingest.py")
        sys.exit(1)


def main():
    print("=" * 60)
    print("  RAG - Tábua de Risco Safra 2026-2027 (NVIDIA)")
    print("=" * 60)

    check_setup()

    from query import query_rag

    print("\nDigite sua pergunta (ou 'sair' para encerrar).\n")
    print("Exemplos:")
    print("  - Qual o risco para soja em Goiás?")
    print("  - Quais municípios de SP têm risco na décade 15?")
    print("  - Qual a situação do milho irrigado em MG?\n")

    while True:
        try:
            question = input("Pergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando.")
            break

        if not question:
            continue
        if question.lower() in ("sair", "exit", "quit"):
            print("Encerrando.")
            break

        print("\nBuscando...\n")
        try:
            answer = query_rag(question)
            print(f"Resposta:\n{answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"Erro: {e}\n")


if __name__ == "__main__":
    main()
