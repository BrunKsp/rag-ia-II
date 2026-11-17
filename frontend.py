"""
RF05: Interface Streamlit de chat com exibição de fontes documentais.
"""

import json
import streamlit as st
import httpx

BACKEND_URL = "http://localhost:8000"


def json_pretty(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)

st.set_page_config(
    page_title="Tábua de Risco Safra 2026-2027",
    page_icon="🌾",
    layout="wide",
)

st.title("🌾 Tábua de Risco Safra 2026-2027")
st.caption("Assistente RAG · Dados oficiais do governo federal · Modelos NVIDIA")

# --- Sidebar ---
with st.sidebar:
    st.header("📋 Culturas disponíveis")

    if st.button("🔄 Carregar culturas", use_container_width=True):
        with st.spinner("Consultando..."):
            try:
                r = httpx.get(f"{BACKEND_URL}/topics", timeout=15)
                st.session_state["topics_data"] = r.json()
            except Exception as e:
                st.error(f"Backend indisponível: {e}")

    if "topics_data" in st.session_state:
        data = st.session_state["topics_data"]
        for t in data.get("topics", []):
            st.markdown(f"**{t['name']}** `{t['chunk_count']} chunks`")
            st.caption(f"Estados: {', '.join(t['states'][:5])}" + ("..." if len(t['states']) > 5 else ""))

    st.divider()
    st.markdown("**💡 Exemplos de consulta:**")
    examples = [
        "Qual o risco para soja em Goiás?",
        "Quais municípios de MG têm menor risco para milho irrigado?",
        "Qual a situação da cevada grãos em São Paulo?",
        "Quais culturas estão disponíveis?",
        "Compare risco de soja sequeira vs irrigada",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:25]}", use_container_width=True):
            st.session_state["pending_query"] = ex

    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# --- Histórico ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg.get("blocked"):
            st.warning(msg["content"])
        else:
            st.markdown(msg["content"])

        if msg.get("sources"):
            with st.expander(f"📚 Fontes documentais ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    score_pct = f"{src.get('score', 0):.1%}"
                    st.markdown(
                        f"- **{src.get('topic', '')}** · {src.get('uf', '')} "
                        f"· `{src.get('portaria', '')}` · relevância: **{score_pct}**"
                    )

        if msg.get("tool_calls"):
            with st.expander("🔧 Ferramentas utilizadas"):
                for tc in msg["tool_calls"]:
                    st.code(f"{tc['tool']}({json_pretty(tc.get('args', {}))})", language="python")


# --- Input ---
query = st.chat_input("Faça uma pergunta sobre risco de safra...")
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")

if query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base de conhecimento..."):
            try:
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"][:-1]
                    if m["role"] in ("user", "assistant")
                ]
                r = httpx.post(
                    f"{BACKEND_URL}/chat",
                    json={"query": query, "history": history},
                    timeout=90,
                )
                r.raise_for_status()
                data = r.json()

                blocked = data.get("blocked", False)
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                tool_calls = data.get("tool_calls", [])

                if blocked:
                    st.warning(answer)
                else:
                    st.markdown(answer)

                if sources:
                    with st.expander(f"📚 Fontes documentais ({len(sources)})"):
                        for src in sources:
                            score_pct = f"{src.get('score', 0):.1%}"
                            st.markdown(
                                f"- **{src.get('topic', '')}** · {src.get('uf', '')} "
                                f"· `{src.get('portaria', '')}` · relevância: **{score_pct}**"
                            )

                if tool_calls:
                    with st.expander("🔧 Ferramentas utilizadas"):
                        for tc in tool_calls:
                            st.code(
                                f"{tc['tool']}({json_pretty(tc.get('args', {}))})",
                                language="python",
                            )

            except httpx.ConnectError:
                answer = "❌ Backend indisponível. Execute: `python backend.py`"
                sources, tool_calls, blocked = [], [], False
                st.error(answer)
            except Exception as e:
                answer = f"❌ Erro: {e}"
                sources, tool_calls, blocked = [], [], False
                st.error(answer)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "tool_calls": tool_calls,
        "blocked": blocked,
    })
