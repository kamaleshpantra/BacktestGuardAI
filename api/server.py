"""
api/server.py
-------------
FastAPI service exposing validation and backtesting workflows.
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from inspection.ast_inspector import inspect_strategy_source
from storage.db import RunDatabase
from validate_ohlcv import validate_ohlcv

app = FastAPI(
    title="BacktestGuard AI API",
    description="Deterministic validation & backtesting API for quantitative trading strategies",
    version="0.1.0",
)

db = RunDatabase()


class CodeInspectRequest(BaseModel):
    source_code: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "BacktestGuard AI"}


@app.post("/inspect/code")
def inspect_code(payload: CodeInspectRequest):
    """Run AST static inspection on provided strategy Python code."""
    res = inspect_strategy_source(payload.source_code)
    return {
        "is_parseable": res.is_parseable,
        "has_confirmed_violations": res.has_confirmed_violations,
        "findings": [
            {
                "code": f.code,
                "certainty": f.certainty,
                "line": f.line_number,
                "description": f.description,
                "evidence": f.evidence,
                "recommendation": f.recommendation,
            }
            for f in res.findings
        ],
    }


@app.post("/validate/dataset")
async def validate_dataset(file: UploadFile = File(...)):
    """Validate uploaded CSV file against OHLCV rules."""
    try:
        contents = await file.read()
        from data.loader import load_csv
        df = load_csv(contents)
        res = validate_ohlcv(df)
        return {
            "passed": res.passed,
            "row_count": res.row_count,
            "errors": [f.__dict__ for f in res.errors],
            "warnings": [f.__dict__ for f in res.warnings],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(exc)}")


@app.get("/history")
def get_history(limit: int = 20):
    """Retrieve historical validation logs."""
    return db.list_runs(limit=limit)
