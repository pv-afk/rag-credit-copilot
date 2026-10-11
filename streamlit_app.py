import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

CONFIDENCE_LABELS = {
    "high": "высокая",
    "medium": "средняя",
    "low": "низкая",
    "insufficient_context": "недостаточно контекста",
}
CONFIDENCE_COLORS = {"high": "green", "medium": "orange", "low": "red", "insufficient_context": "gray"}

st.set_page_config(page_title="RAG Copilot — Кредитный аналитик", page_icon="📊", layout="centered")

st.title("📊 RAG Copilot для кредитного аналитика")
st.caption(
    "Гибридный поиск (BM25 + векторный поиск) по базе публичных документов о кредитном риске, "
    "с цитируемыми структурированными ответами. Учебный проект — без внутренних банковских данных."
)

with st.sidebar:
    st.header("О проекте")
    st.write(
        "База знаний: Базель III, МСФО 9, модели PD/LGD/EAD, методология рейтингов, "
        "ковенанты, залоги, стресс-тестирование, индикаторы раннего предупреждения, "
        "реструктуризация, синдицированное кредитование, суверенный риск, риск контрагента, KYC/AML."
    )
    st.write(f"API: `{API_URL}`")
    top_k = st.slider("Число фрагментов (Top-K)", min_value=1, max_value=10, value=5)

question = st.text_input(
    "Задайте вопрос по кредитному анализу",
    placeholder="Каков минимальный норматив CET1 по Базелю III?",
)

if st.button("Спросить", type="primary") and question.strip():
    with st.spinner("Ищу контекст и формирую ответ..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question, "top_k": top_k}, timeout=180)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            st.error(f"Запрос не выполнен: {e}")
            data = None

    if data:
        if not data["is_answerable"]:
            st.warning(data["answer"])
        else:
            st.subheader("Ответ")
            st.write(data["answer"])

            confidence = data["confidence"]
            color = CONFIDENCE_COLORS.get(confidence, "gray")
            label = CONFIDENCE_LABELS.get(confidence, confidence)
            st.markdown(f"**Уверенность:** :{color}[{label}]")

            if data["citations"]:
                st.subheader("Цитаты")
                for c in data["citations"]:
                    st.markdown(f"- `{c['chunk_id']}` — *{c['doc_title']}*\n  > {c['quote']}")

        with st.expander("ID найденных фрагментов (отладка)"):
            st.write(data["retrieved_chunk_ids"])
