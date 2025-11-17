import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional

from langfuse import Langfuse

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_langfuse: Optional[Langfuse] = None


def setup_langfuse() -> None:
    """
    Initialise Langfuse client from environment variables.
    """
    global _langfuse
    try:
        _langfuse = Langfuse(
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        logger.info("Langfuse client initialised")
    except Exception as exc:
        logger.warning("Failed to initialise Langfuse: %s", exc)
        _langfuse = None


def trace_data_generation(func: Callable[[], Dict[str, Any]], ddl: str, prompt: str) -> Dict[str, Any]:
    """
    Trace the synthetic data generation step.
    """
    if _langfuse is None:
        return func()

    ddl_snippet = ddl[:800] + "..." if len(ddl) > 800 else ddl
    trace = _langfuse.trace(
        name="synthetic_data_generation",
        input={"ddl": ddl_snippet, "user_prompt": prompt},
        metadata={"component": "data_factory"},
    )

    gen = trace.generation(
        name="build_dataset",
        model="gemini-2.5-flash",
        input={"ddl_preview": ddl_snippet},
    )

    try:
        result = func()
        gen.end(
            output={
                "tables": list(result.keys()),
                "total_rows": sum(len(df) for df in result.values()),
            }
        )
        trace.update(
            output={
                "status": "success",
                "tables": list(result.keys()),
            }
        )
        return result
    except Exception as exc:
        gen.end(output={"status": "error", "error": str(exc)})
        trace.update(output={"status": "error", "error": str(exc)})
        logger.error("Error during traced data generation: %s", exc)
        raise


def trace_nl_query(func: Callable[[], Any], nl_query: str) -> Any:
    """
    Trace the NL→SQL→execution pipeline.
    """
    if _langfuse is None:
        return func()

    trace = _langfuse.trace(
        name="natural_language_query",
        input={"question": nl_query},
        metadata={"component": "sql_agent"},
    )

    gen = trace.generation(
        name="nl_to_sql_and_execution",
        model="gemini-2.5-flash",
        input={"question": nl_query},
    )

    try:
        result = func()
        sql = getattr(result, "sql", None) or result.get("sql") if isinstance(result, dict) else None

        gen.end(
            output={
                "sql": sql,
                "rows": getattr(result, "data", None).shape[0]
                if hasattr(result, "data")
                else result.get("rows_returned", None)
                if isinstance(result, dict)
                else None,
            }
        )
        trace.update(output={"status": "success"})
        return result
    except Exception as exc:
        gen.end(output={"status": "error", "error": str(exc)})
        trace.update(output={"status": "error", "error": str(exc)})
        logger.error("Error during traced NL→SQL query: %s", exc)
        raise


def langfuse_traced(operation_name: str) -> Callable[[Callable], Callable]:
    """
    Generic decorator for tracing arbitrary functions.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _langfuse is None:
                return func(*args, **kwargs)

            trace = _langfuse.trace(
                name=operation_name,
                input={"args": str(args)[:200], "kwargs": str(kwargs)[:200]},
                metadata={"component": "custom"},
            )
            try:
                result = func(*args, **kwargs)
                trace.update(output={"status": "success"})
                return result
            except Exception as exc:
                trace.update(output={"status": "error", "error": str(exc)})
                raise

        return wrapper

    return decorator


def record_user_feedback(trace_id: str, score: float, comment: str = "") -> None:
    """
    Record explicit user feedback for a Langfuse trace.
    """
    if _langfuse is None:
        logger.warning("Langfuse is not configured; feedback is ignored")
        return

    try:
        _langfuse.score(
            trace_id=trace_id,
            name="user_feedback",
            value=score,
            comment=comment,
        )
        logger.info("User feedback recorded for trace %s", trace_id)
    except Exception as exc:
        logger.error("Failed to record user feedback: %s", exc)


def flush_events() -> None:
    """
    Explicitly flush buffered events to Langfuse.
    """
    if _langfuse is None:
        return
    try:
        _langfuse.flush()
    except Exception as exc:
        logger.error("Failed to flush Langfuse events: %s", exc)
