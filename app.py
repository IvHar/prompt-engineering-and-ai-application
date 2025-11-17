import os
from typing import Dict

import pandas as pd
import streamlit as st

from synthetic_factory import LLMDataBuilder
from sql_agent import SQLConversationAgent, QueryResult
from observability import setup_langfuse, trace_data_generation, trace_nl_query

st.set_page_config(
    page_title="AI Data Assistant",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f7;
        color: #111827;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif;
    }
    /* Hide default Streamlit header / toolbar */
    header[data-testid="stHeader"] {
        display: none;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    .main-container {
        max-width: 900px;
        margin: 0 auto;
    }
    h1, h2, h3, h4 {
        color: #111827;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #111827;
        color: #ffffff;
        border-radius: 999px;
        border: 1px solid #111827;
        padding: 0.45rem 1.4rem;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #1f2933;
        border-color: #1f2933;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        background-color: #e5e7eb;
        color: #4b5563;
        font-size: 0.75rem;
    }
    .card {
        background-color: #ffffff;
        border-radius: 0.75rem;
        border: 1px solid #e5e7eb;
        padding: 1rem 1.25rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }
    .chat-bubble-user {
        background-color: #111827;
        color: #ffffff;
        border-radius: 1rem;
        padding: 0.75rem 1rem;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bubble-assistant {
        background-color: #ffffff;
        border-radius: 1rem;
        padding: 0.75rem 1rem;
        max-width: 80%;
        border: 1px solid #e5e7eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

setup_langfuse()

if "generated_data" not in st.session_state:
    st.session_state.generated_data: Dict[str, pd.DataFrame] | None = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result: QueryResult | None = None
if "last_question" not in st.session_state:
    st.session_state.last_question: str | None = None

st.sidebar.title("AI Data Assistant")
st.sidebar.caption("Synthetic data & NL→SQL with Gemini")
st.sidebar.markdown("---")

nav = st.sidebar.radio(
    "Workspace",
    ["📊 Data Generation", "💬 Talk to your data", "📈 Query history"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Stack**
- Vertex AI Gemini  
- Streamlit  
- PostgreSQL  
- Langfuse
"""
)

if nav == "📊 Data Generation":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)

        st.markdown("### 📊 Synthetic data workspace")
        st.caption(
            "Upload a PostgreSQL DDL schema, describe how the data should look, "
            "and generate synthetic tables that you can later query with natural language."
        )

        col_main, col_side = st.columns([2.2, 1])

        with col_main:
            ddl_file = st.file_uploader(
                "PostgreSQL DDL file",
                type=["sql", "ddl", "txt"],
                help="The file should contain one or more CREATE TABLE statements.",
            )
            user_prompt = st.text_area(
                "High-level instructions for the data",
                placeholder=(
                    "Example: generate a realistic employee and project dataset "
                    "for an enterprise. Use European names, dates in 2023–2024, "
                    "and plausible numeric ranges."
                ),
                height=130,
            )

        with col_side:
            st.markdown("#### Settings")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Higher values make the data more diverse and creative.",
            )
            max_tokens = st.number_input(
                "Max output tokens",
                min_value=1000,
                max_value=8000,
                value=6000,
                step=500,
            )
            rows_per_table = st.number_input(
                "Rows per table",
                min_value=5,
                max_value=200,
                value=30,
                step=5,
            )
            generate_btn = st.button("Generate data", use_container_width=True)

        if generate_btn:
            if ddl_file is None or not user_prompt.strip():
                st.error("Please upload a DDL file and provide data instructions.")
            else:
                ddl_text = ddl_file.read().decode("utf-8")
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "gd-gcp-gridu-genai")
                location = os.getenv("GOOGLE_REGION", "us-central1")

                try:
                    builder = LLMDataBuilder(
                        project_id=project_id,
                        location=location,
                        model_name="gemini-2.5-flash",
                    )

                    def _build() -> Dict[str, pd.DataFrame]:
                        return builder.generate_dataset(
                            ddl_text=ddl_text,
                            user_instructions=user_prompt,
                            rows_per_table=int(rows_per_table),
                            temperature=float(temperature),
                            max_tokens=int(max_tokens),
                        )

                    with st.spinner("Generating synthetic tables with Gemini..."):
                        st.session_state.generated_data = trace_data_generation(
                            _build, ddl_text, user_prompt
                        )

                    st.success("Synthetic data generated successfully.")
                except Exception as exc:
                    st.error(f"Data generation failed: {exc}")
                    if "credentials" in str(exc).lower():
                        st.info(
                            "Authentication issue detected. "
                            "Run `gcloud auth application-default login` "
                            "or provide a service account JSON via GOOGLE_APPLICATION_CREDENTIALS."
                        )

        if st.session_state.generated_data:
            st.markdown("#### Preview generated tables")
            for name, df in st.session_state.generated_data.items():
                with st.expander(f"{name} (rows: {len(df)})", expanded=False):
                    st.dataframe(df, use_container_width=True, height=220)
                    csv_bytes = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download CSV",
                        data=csv_bytes,
                        file_name=f"{name}.csv",
                        mime="text/csv",
                        key=f"dl_{name}",
                    )

        st.markdown("</div>", unsafe_allow_html=True)

elif nav == "💬 Talk to your data":
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)

        st.markdown("### 💬 Talk to your data")
        st.caption(
            "Ask a question in natural language and let the assistant generate "
            "a PostgreSQL query and visualise the result."
        )

        if st.session_state.generated_data is None:
            st.info("You need to generate some data first in the Data Generation tab.")
        else:
            with st.expander("View available tables and columns", expanded=False):
                for name, df in st.session_state.generated_data.items():
                    cols = ", ".join(df.columns)
                    st.markdown(f"- **{name.lower()}** — {cols}")

            st.markdown("#### Ask a question")

            question = st.text_area(
                "Your message",
                placeholder="Example: show employees who work on projects with a budget above 200000.",
                height=90,
                label_visibility="collapsed",
                key="nl_question",
            )

            col_btn, _ = st.columns([1, 5])
            with col_btn:
                ask_btn = st.button("Send", use_container_width=True)

            st.markdown("##### Quick examples")
            example_cols = st.columns(3)

            examples = [
                "List the 5 highest-paid employees with their departments.",
                "Show total salary cost per department as a chart.",
                "Find companies that have departments located in TX.",
            ]

            def set_example(text: str) -> None:
                st.session_state.nl_question = text

            for idx, (col, text) in enumerate(zip(example_cols, examples)):
                col.button(
                    text,
                    key=f"example_{idx}",
                    use_container_width=True,
                    on_click=set_example,
                    args=(text,),
                )

            if ask_btn and question.strip():
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "gd-gcp-gridu-genai")
                location = os.getenv("GOOGLE_REGION", "us-central1")

                pg_cfg = {
                    "host": os.getenv("POSTGRES_HOST", "postgres"),
                    "port": int(os.getenv("POSTGRES_PORT", 5432)),
                    "database": os.getenv("POSTGRES_DB", "synthetic_data"),
                    "user": os.getenv("POSTGRES_USER", "postgres"),
                    "password": os.getenv("POSTGRES_PASSWORD", "postgres"),
                }

                try:
                    agent = SQLConversationAgent(
                        tables=st.session_state.generated_data,
                        project_id=project_id,
                        location=location,
                        model_name="gemini-2.5-flash",
                        pg_config=pg_cfg,
                    )

                    def _run() -> QueryResult:
                        return agent.ask(question)

                    with st.spinner("Thinking and executing your query..."):
                        result: QueryResult = trace_nl_query(_run, question)

                    st.session_state.last_result = result
                    st.session_state.last_question = question
                    st.session_state.query_history.insert(
                        0,
                        {
                            "question": question,
                            "sql": result.sql,
                            "rows": len(result.data),
                        },
                    )
                except Exception as exc:
                    st.error(f"Query failed: {exc}")
                    st.session_state.last_result = None

            if st.session_state.last_result is not None:
                result = st.session_state.last_result
                last_q = st.session_state.last_question or ""

                st.markdown("#### Conversation")
                if last_q:
                    st.markdown(
                        f'<div class="chat-bubble-user">{last_q}</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    '<div class="chat-bubble-assistant">',
                    unsafe_allow_html=True,
                )
                st.markdown("**Generated SQL**")
                st.code(result.sql, language="sql")
                st.markdown("**Result table**")
                st.dataframe(result.data, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                csv_result = result.data.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download result as CSV",
                    data=csv_result,
                    file_name="query_result.csv",
                    mime="text/csv",
                )

                numeric_df = result.data.select_dtypes(include=["number"])
                if not result.data.empty and numeric_df.shape[1] >= 1:
                    st.markdown("#### Quick chart")

                    all_columns = list(result.data.columns)
                    default_x = all_columns[0]

                    numeric_cols = list(numeric_df.columns)

                    y_candidates = [c for c in numeric_cols if c != default_x]

                    if not y_candidates:
                        st.info(
                            "The result has only one numeric column or no suitable pair for X/Y, "
                            "so an automatic chart is not very informative."
                        )
                    else:
                        x_col = st.selectbox(
                            "X axis",
                            options=all_columns,
                            index=all_columns.index(default_x),
                        )

                        y_options = [c for c in numeric_cols if c != x_col]
                        if not y_options:
                            st.info(
                                "The selected X axis uses the only numeric column, "
                                "so there is no separate Y axis to plot."
                            )
                        else:
                            y_col = st.selectbox(
                                "Y axis",
                                options=y_options,
                                index=0,
                            )

                            chart_type = st.selectbox(
                                "Chart type",
                                ["Line", "Bar", "Area"],
                            )

                            chart_base = result.data[[x_col, y_col]].set_index(x_col)
                            if chart_type == "Line":
                                st.line_chart(chart_base)
                            elif chart_type == "Bar":
                                st.bar_chart(chart_base)
                            else:
                                st.area_chart(chart_base)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    with st.container():
        st.markdown('<div class="main-container">', unsafe_allow_html=True)

        st.markdown("### 📈 Query history")

        if not st.session_state.query_history:
            st.info("No queries executed yet.")
        else:
            total = len(st.session_state.query_history)
            total_rows = sum(q["rows"] for q in st.session_state.query_history)
            avg_rows = total_rows / total if total else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Total queries", total)
            c2.metric("Total rows returned", total_rows)
            c3.metric("Average rows / query", f"{avg_rows:.1f}")

            st.markdown("#### Recent queries")

            for idx, item in enumerate(st.session_state.query_history[:10]):
                with st.expander(f"Query {idx + 1}: {item['question'][:70]}..."):
                    st.markdown("**Question**")
                    st.write(item["question"])
                    st.markdown("**SQL**")
                    st.code(item["sql"], language="sql")
                    st.markdown(f"**Rows**: {item['rows']}")

            if st.button("Clear history"):
                st.session_state.query_history = []
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
