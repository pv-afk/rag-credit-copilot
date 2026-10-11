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


def ask(question: str, top_k: int) -> tuple[dict | None, str | None]:
    """Возвращает (ответ API, None) или (None, текст ошибки для пользователя)."""
    try:
        resp = requests.post(f"{API_URL}/query", json={"question": question, "top_k": top_k}, timeout=180)
    except requests.ConnectionError:
        return None, f"Не удалось подключиться к API по адресу {API_URL}. Запущен ли сервер?"
    except requests.Timeout:
        return None, "API не ответил за 180 секунд."
    if resp.ok:
        return resp.json(), None
    try:
        detail = resp.json().get("detail")
    except ValueError:
        detail = None
    return None, f"Ошибка API ({resp.status_code}): {detail or resp.reason}"


st.set_page_config(page_title="RAG Copilot — кредитный аналитик", page_icon="📊", layout="centered")

st.title("📊 RAG Copilot для кредитного аналитика")
st.caption(
    "Гибридный поиск (BM25 + векторный поиск) по базе документов о кредитном риске "
    "с цитатами из источников. Учебный проект: документы синтетические, внутренних банковских данных нет."
)

with st.sidebar:
    st.header("О проекте")
    st.write(
        "База знаний: Базель III, МСФО 9, модели PD/LGD/EAD, методология рейтингов, "
        "ковенанты, залоги, стресс-тестирование, индикаторы раннего предупреждения, "
        "реструктуризация, синдицированное кредитование, суверенный риск, риск контрагента, KYC/AML."
    )
    st.write(f"API: `{API_URL}`")
    top_k = st.slider("Число фрагментов в контексте (Top-K)", min_value=1, max_value=10, value=5)

question = st.text_input(
    "Задайте вопрос по кредитному анализу",
    placeholder="Каков минимальный норматив CET1 по Базелю III?",
)

if st.button("Спросить", type="primary"):
    if not question.strip():
        st.info("Введите вопрос.")
        st.stop()

    with st.spinner("Ищу фрагменты и формирую ответ..."):
        data, error = ask(question, top_k)

    if error:
        st.error(error)
        st.stop()

    if not data["is_answerable"]:
        st.warning(data["answer"])
    else:
        st.subheader("Ответ")
        st.write(data["answer"])

        confidence = data["confidence"]
        color = CONFIDENCE_COLORS.get(confidence, "gray")
        label = CONFIDENCE_LABELS.get(confidence, confidence)
        # Это самооценка LLM, а не измеренная вероятность правильного ответа.
        st.markdown(f"**Самооценка модели:** :{color}[{label}]")

        if data["citations"]:
            st.subheader("Цитаты")
            for c in data["citations"]:
                st.markdown(f"- `{c['chunk_id']}` — *{c['doc_title']}*\n  > {c['quote']}")

    with st.expander("ID найденных фрагментов (отладка)"):
        st.write(data["retrieved_chunk_ids"])
