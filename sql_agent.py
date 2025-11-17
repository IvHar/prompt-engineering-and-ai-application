import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import psycopg2

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    VERTEX_READY = True
except ImportError:
    VERTEX_READY = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class QueryResult:
    sql: str
    data: pd.DataFrame
    execution_time: float
    backend: str


class SQLConversationAgent:
    """
    Conversational SQL agent that:
    - turns natural-language questions into SQL using Gemini,
    - executes queries against PostgreSQL only,
    - keeps an internal (lowercase) table name mapping to avoid case issues.
    """

    def __init__(
        self,
        tables: Dict[str, pd.DataFrame],
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash",
        pg_config: Optional[Dict] = None,
    ):
        if not VERTEX_READY:
            raise RuntimeError(
                "vertexai is not installed. Install google-cloud-aiplatform to enable NL→SQL."
            )

        vertexai.init(project=project_id, location=location)
        self._model = GenerativeModel(model_name)

        self._tables: Dict[str, pd.DataFrame] = {}
        for raw_name, df in tables.items():
            normalized = raw_name.lower()
            self._tables[normalized] = df

        self._pg_cfg = pg_config or {}
        self._pg_conn: Optional[psycopg2.extensions.connection] = None

        self._init_postgres()

    def _init_postgres(self) -> None:
        """
        Open a PostgreSQL connection and load all dataframes as tables
        using lowercase table names.
        """
        try:
            self._pg_conn = psycopg2.connect(
                host=self._pg_cfg.get("host", "localhost"),
                port=self._pg_cfg.get("port", 5432),
                database=self._pg_cfg.get("database", "postgres"),
                user=self._pg_cfg.get("user", "postgres"),
                password=self._pg_cfg.get("password", ""),
            )
            logger.info("Connected to PostgreSQL backend")
            self._load_tables_into_postgres()
        except Exception as exc:
            logger.error("Failed to initialise PostgreSQL: %s", exc)
            self._pg_conn = None
            raise RuntimeError(f"PostgreSQL connection failed: {exc}") from exc

    def _load_tables_into_postgres(self) -> None:
        """
        Create tables in PostgreSQL using lowercase names and insert dataframe rows.
        Column names are used as-is (PostgreSQL will fold them to lowercase when unquoted).
        """
        if self._pg_conn is None:
            raise RuntimeError("PostgreSQL connection is not available")

        cur = self._pg_conn.cursor()

        for name, df in self._tables.items():
            try:
                cur.execute(f"DROP TABLE IF EXISTS {name} CASCADE")

                col_defs = []
                for col in df.columns:
                    dt = df[col].dtype
                    if pd.api.types.is_integer_dtype(dt):
                        pg_type = "INTEGER"
                    elif pd.api.types.is_float_dtype(dt):
                        pg_type = "NUMERIC"
                    elif pd.api.types.is_bool_dtype(dt):
                        pg_type = "BOOLEAN"
                    elif pd.api.types.is_datetime64_any_dtype(dt):
                        pg_type = "TIMESTAMP"
                    else:
                        pg_type = "TEXT"
                    col_defs.append(f"{col} {pg_type}")

                create_sql = f"CREATE TABLE {name} ({', '.join(col_defs)})"
                cur.execute(create_sql)

                for _, row in df.iterrows():
                    placeholders = ", ".join(["%s"] * len(row))
                    insert_sql = f"INSERT INTO {name} VALUES ({placeholders})"
                    cur.execute(insert_sql, tuple(row))

                self._pg_conn.commit()
                logger.info("Loaded %d rows into PostgreSQL table %s", len(df), name)
            except Exception as exc:
                logger.error("Failed to load table %s into PostgreSQL: %s", name, exc)
                self._pg_conn.rollback()

        cur.close()

    def _schema_summary(self) -> str:
        """
        Produce a textual summary of available tables and columns for the LLM.
        Table names are always lowercase to match PostgreSQL.
        """
        lines = []
        for name, df in self._tables.items():
            lines.append(f"table {name}:")
            for col in df.columns:
                dtype = str(df[col].dtype)
                examples = df[col].dropna().head(3).tolist()
                lines.append(f"  - {col} ({dtype}), examples: {examples}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _clean_sql(raw_sql: str) -> str:
        """
        Remove markdown fences, SQL comments and trailing semicolons.
        """
        sql = re.sub(r"```[\w]*\n", "", raw_sql)
        sql = sql.replace("```", "")
        sql = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
        sql = sql.strip().rstrip(";")
        return sql

    def _generate_sql(self, question: str) -> str:
        """
        Ask Gemini to generate a single SELECT statement in PostgreSQL dialect.
        """
        schema_text = self._schema_summary()

        prompt = f"""
You are a senior data analyst.

Below is a description of the available tables and columns:

{schema_text}

User question:
{question}

Write ONE SQL query in PostgreSQL dialect that answers the question.

IMPORTANT RULES:
- Only SELECT is allowed. Never use INSERT, UPDATE, DELETE, DROP or CREATE.
- Always use lowercase table names as shown in the schema summary (e.g. "employees", not "Employees").
- Use valid column names from the schema.
- Prefer explicit JOIN with ON conditions.
- For "top N" style questions, use ORDER BY and LIMIT N.
- Do not wrap the SQL in markdown or add explanations.
Return ONLY the SQL string.
"""

        response = self._model.generate_content(
            prompt,
            generation_config=GenerationConfig(
                temperature=0.3,
                max_output_tokens=512,
            ),
        )

        raw_sql = response.text or ""
        sql = self._clean_sql(raw_sql)
        logger.info("Generated SQL: %s", sql)
        return sql

    def _execute_postgres(self, sql: str) -> pd.DataFrame:
        """
        Run a SELECT statement in PostgreSQL and return a DataFrame.
        """
        if self._pg_conn is None:
            raise RuntimeError("PostgreSQL connection is not available")

        cur = self._pg_conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
            df = pd.DataFrame(rows, columns=cols)
            return df
        except Exception as exc:
            raise RuntimeError(f"Failed to execute query in PostgreSQL: {exc}") from exc
        finally:
            cur.close()

    def ask(self, question: str) -> QueryResult:
        """
        Generate SQL for the given natural-language question and execute it
        in PostgreSQL, returning a QueryResult object.
        """
        start = time.time()
        sql = self._generate_sql(question)
        df = self._execute_postgres(sql)
        elapsed = time.time() - start

        logger.info(
            "Query executed via PostgreSQL in %.3f s, %d rows",
            elapsed,
            len(df),
        )

        return QueryResult(
            sql=sql,
            data=df,
            execution_time=elapsed,
            backend="PostgreSQL",
        )

    def explain(self, question: str) -> Dict[str, str]:
        """
        Generate SQL and then ask the model to explain what the query does.
        """
        sql = self._generate_sql(question)

        explain_prompt = f"""
Explain in clear English what the following SQL query does:

{sql}

Describe:
- which tables are used;
- which columns are referenced;
- how data is filtered and/or aggregated;
- what the final result represents.
"""

        response = self._model.generate_content(
            explain_prompt,
            generation_config=GenerationConfig(
                temperature=0.5,
                max_output_tokens=512,
            ),
        )
        explanation = response.text or "Explanation is not available."

        return {"sql": sql, "explanation": explanation}

    def __del__(self) -> None:
        if self._pg_conn is not None:
            try:
                self._pg_conn.close()
                logger.info("PostgreSQL connection closed")
            except Exception:
                pass