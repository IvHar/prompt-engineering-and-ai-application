import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig, Part

    VERTEX_READY = True
except ImportError:
    VERTEX_READY = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class ColumnSpec:
    name: str
    raw_type: str
    nullable: bool = True
    primary_key: bool = False
    unique: bool = False
    has_default: bool = False


@dataclass
class TableSpec:
    name: str
    columns: List[ColumnSpec] = field(default_factory=list)


@dataclass
class DatabaseSpec:
    tables: Dict[str, TableSpec] = field(default_factory=dict)

DDL_CREATE_TABLE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)\s*\((.*?)\);",
    re.IGNORECASE | re.DOTALL,
)


def parse_postgres_ddl(ddl_text: str) -> DatabaseSpec:
    """
    Parse a simplified PostgreSQL DDL into an internal schema representation.

    Rules:
    - Only CREATE TABLE blocks are handled.
    - Each "column line" must start with a valid SQL identifier.
    - Lines starting with PRIMARY KEY, FOREIGN KEY, CONSTRAINT, UNIQUE, CHECK,
      or SQL comments (--) are ignored as table-level constraints.
    """
    db = DatabaseSpec()

    for table_name, body in DDL_CREATE_TABLE_RE.findall(ddl_text):
        table = TableSpec(name=table_name.strip())

        for raw_line in body.split(","):
            line = raw_line.strip()
            if not line:
                continue

            upper = line.upper()

            if upper.startswith(
                ("PRIMARY KEY", "FOREIGN KEY", "CONSTRAINT", "UNIQUE", "CHECK")
            ):
                continue

            if line.startswith("--"):
                continue
            if not re.match(r"[A-Za-z_]\w*\s+", line):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            col_name = parts[0].strip()
            col_type = parts[1].strip().upper()

            col = ColumnSpec(
                name=col_name,
                raw_type=col_type,
                nullable="NOT NULL" not in upper,
                primary_key="PRIMARY KEY" in upper,
                unique="UNIQUE" in upper,
                has_default="DEFAULT" in upper,
            )
            table.columns.append(col)

        if table.columns:
            db.tables[table.name] = table

    logger.info("Parsed %d tables from DDL", len(db.tables))
    return db


class LLMDataBuilder:
    """
    High-level synthetic dataset builder powered by Vertex AI Gemini.

    If credentials or Vertex AI are not available, it gracefully falls back
    to a deterministic local generator for all tables.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-2.5-flash",
    ):
        self._use_llm = False
        self._model: Optional[GenerativeModel] = None

        if not VERTEX_READY:
            logger.warning(
                "vertexai is not installed; falling back to local synthetic generator only."
            )
            return

        try:
            vertexai.init(project=project_id, location=location)
            self._model = GenerativeModel(model_name)
            self._use_llm = True
            logger.info(
                "Vertex AI initialised for project %s in %s with model %s",
                project_id,
                location,
                model_name,
            )
        except Exception as exc:
            logger.warning(
                "Vertex AI initialisation failed (%s). "
                "Synthetic data will be generated locally without LLM.",
                exc,
            )
            self._model = None
            self._use_llm = False

    def generate_dataset(
        self,
        ddl_text: str,
        user_instructions: str,
        rows_per_table: int,
        temperature: float,
        max_tokens: int,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a dataset for all tables defined in the DDL.
        If LLM is not available, all tables are generated via the local fallback.
        """
        schema = parse_postgres_ddl(ddl_text)
        frames: Dict[str, pd.DataFrame] = {}

        for table_name, table in schema.tables.items():
            logger.info("Generating rows for table %s", table_name)

            if not self._use_llm:
                df = self._generate_fallback(table, rows_per_table)
            else:
                df = self._generate_single_table(
                    table=table,
                    all_tables=schema,
                    ddl_text=ddl_text,
                    user_instructions=user_instructions,
                    rows_per_table=rows_per_table,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            frames[table_name] = df

        return frames

    def _generate_single_table(
        self,
        table: TableSpec,
        all_tables: DatabaseSpec,
        ddl_text: str,
        user_instructions: str,
        rows_per_table: int,
        temperature: float,
        max_tokens: int,
    ) -> pd.DataFrame:
        """
        Ask Gemini to generate a JSON array of rows for a single table.
        Falls back to a deterministic local generator if something goes wrong.
        """
        if not self._use_llm or self._model is None:
            return self._generate_fallback(table, rows_per_table)

        prompt = self._build_prompt(
            table=table,
            all_tables=all_tables,
            ddl_text=ddl_text,
            user_instructions=user_instructions,
            rows_per_table=rows_per_table,
        )

        try:
            response = self._model.generate_content(
                [Part.from_text(prompt)],
                generation_config=GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            raw_json = response.text
            rows = json.loads(raw_json)

            if not isinstance(rows, list) or not rows:
                logger.warning(
                    "Model returned non-list JSON for table %s, using fallback",
                    table.name,
                )
                return self._generate_fallback(table, rows_per_table)

            df = pd.DataFrame(rows)

            for col in table.columns:
                if col.name not in df.columns:
                    df[col.name] = None

            ordered_cols = [c.name for c in table.columns]
            df = df[ordered_cols]

            logger.info("Generated %d rows for table %s using LLM", len(df), table.name)
            return df

        except Exception as exc:
            logger.warning(
                "Failed to generate data via LLM for table %s (%s); using fallback instead.",
                table.name,
                exc,
            )
            return self._generate_fallback(table, rows_per_table)

    def _build_prompt(
        self,
        table: TableSpec,
        all_tables: DatabaseSpec,
        ddl_text: str,
        user_instructions: str,
        rows_per_table: int,
    ) -> str:
        col_lines: List[str] = []
        for col in table.columns:
            flags: List[str] = []
            if col.primary_key:
                flags.append("PRIMARY KEY")
            if not col.nullable:
                flags.append("NOT NULL")
            if col.unique:
                flags.append("UNIQUE")
            flag_suffix = f" ({', '.join(flags)})" if flags else ""
            col_lines.append(f"- {col.name}: {col.raw_type}{flag_suffix}")

        table_overview = "\n".join(col_lines)

        return f"""
You are a data generation assistant.

Goal:
Create a synthetic dataset for PostgreSQL table `{table.name}`.

Table schema:
{table_overview}

Global database DDL:
{ddl_text}

User instructions:
{user_instructions}

Requirements:
- Generate EXACTLY {rows_per_table} rows.
- Respect column data types and constraints.
- Primary key columns must contain unique values.
- Use realistic looking values (names, dates, amounts, etc.).
- Output MUST be a JSON array of objects. Do not wrap it in markdown or add commentary.
"""

    def _generate_fallback(self, table: TableSpec, rows: int) -> pd.DataFrame:
        """
        Local deterministic generator as a safety net when LLM output fails
        or Vertex AI is not available.
        """
        records: List[Dict[str, object]] = []

        for i in range(rows):
            row: Dict[str, object] = {}
            for col in table.columns:
                t = col.raw_type.upper()

                if t in ("INT", "INTEGER", "BIGINT", "SMALLINT", "SERIAL", "BIGSERIAL"):
                    row[col.name] = i + 1
                elif any(x in t for x in ("CHAR", "TEXT", "UUID")):
                    row[col.name] = f"{col.name}_{i + 1}"
                elif any(x in t for x in ("NUMERIC", "DECIMAL", "REAL", "DOUBLE")):
                    row[col.name] = round((i + 1) * 10.0, 2)
                elif t == "DATE":
                    row[col.name] = f"2024-01-{(i + 1):02d}"
                elif "TIME" in t:
                    row[col.name] = f"2024-01-{(i + 1):02d} 12:00:00"
                elif "BOOL" in t:
                    row[col.name] = bool(i % 2 == 0)
                else:
                    row[col.name] = f"value_{i + 1}"

            records.append(row)

        logger.info(
            "Fallback generator produced %d rows for table %s",
            rows,
            table.name,
        )
        return pd.DataFrame(records)
