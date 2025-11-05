from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import io
import json
import asyncio

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# ---------- App ----------
app = FastAPI(title="ODHF LODE MCP Server (FastAPI)")

# Public CORS (no auth)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Data ----------
CSV_FILE = Path(os.getenv("CSV_PATH", "odhf_v1.1.csv"))
CSV_URL = os.getenv("CSV_URL", "").strip()  # optional: direct link to the CSV
DF: Optional[pd.DataFrame] = None
COLUMNS: Optional[List[str]] = None

ALIAS_MAP = {
    "province": {
        "province", "Province", "Province or Territory", "Province/Territory",
        "prov", "province_or_territory"
    },
    "odhf_facility_type": {
        "odhf_facility_type", "ODHF Facility Type", "Facility Type", "facility_type",
        "odhf facility type"
    },
}

def load_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    """Robust loader for odd encodings."""
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="cp1252", errors="replace", low_memory=False)

def try_download_csv(url: str, dest: Path) -> bool:
    """Download the CSV if a URL is provided (first deploy convenience)."""
    if not url:
        return False
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        return True
    except Exception as e:
        print(f"[WARN] CSV download failed: {e}")
        return False

async def ensure_loaded() -> None:
    """Lazy, single-shot load (keeps /sse_once super fast)."""
    global DF, COLUMNS
    if DF is not None:
        return
    # If file not present but URL provided → download once
    if not CSV_FILE.exists() and CSV_URL:
        ok = try_download_csv(CSV_URL, CSV_FILE)
        if not ok:
            raise HTTPException(status_code=400, detail="CSV_URL provided but download failed.")

    df = load_csv_safely(CSV_FILE)
    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"CSV not found or empty at {CSV_FILE.resolve()}. "
                   f"Upload odhf_v1.1.csv or set CSV_URL."
        )
    DF = df
    COLUMNS = list(df.columns)

def find_col(candidates: set[str]) -> Optional[str]:
    if not COLUMNS:
        return None
    lower = {c.lower(): c for c in COLUMNS}
    for want in candidates:
        if want in COLUMNS:
            return want
        v = lower.get(want.lower())
        if v:
            return v
    return None

def df_to_records_clean(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """JSON-safe (NaN/inf→None)."""
    safe = frame.replace([np.inf, -np.inf], np.nan)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")

# ---------- Health ----------
@app.get("/", response_class=PlainTextResponse)
async def root():
    rows = None if DF is None else int(len(DF))
    return f"ODHF MCP Server is running! csv_found={CSV_FILE.exists()} rows={rows}"

# ---------- HTTP tools (non-MCP) ----------
@app.get("/list_fields")
async def list_fields():
    await ensure_loaded()
    return {"columns": COLUMNS}

@app.get("/search_facilities")
async def search_facilities(
    province: str = Query(None, description="Province or territory (e.g., Quebec, QC)"),
    facility_type: str = Query(None, description="ODHF facility type (e.g., Hospitals)"),
    limit: int = Query(25, ge=1, le=200),
):
    await ensure_loaded()
    col_prov = find_col(ALIAS_MAP["province"])
    col_type = find_col(ALIAS_MAP["odhf_facility_type"])
    if col_prov is None or col_type is None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Expected columns not found.",
                "have": COLUMNS,
                "need_any_of": {
                    "province": list(ALIAS_MAP["province"]),
                    "odhf_facility_type": list(ALIAS_MAP["odhf_facility_type"]),
                },
            },
        )

    df = DF
    filt = df
    if province:
        filt = filt[filt[col_prov].astype(str).str.contains(province, case=False, na=False)]
    if facility_type:
        filt = filt[filt[col_type].astype(str).str.contains(facility_type, case=False, na=False)]

    if filt.empty:
        return {"message": "No results. Try another province (e.g., QC/Quebec) or facility_type."}

    preferred_cols = ["Facility Name", "City", col_prov, col_type, "Postal Code", "Latitude", "Longitude"]
    use_cols = [c for c in preferred_cols if c in filt.columns]
    if use_cols:
        filt = filt[use_cols]

    return df_to_records_clean(filt.head(limit))

# ---------- MCP: tools & SSE discovery ----------
TOOLS_MANIFEST = [
    {
        "name": "list_fields",
        "description": "List dataset columns.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_facilities",
        "description": "Search facilities by province and/or ODHF facility type.",
        "input_schema": {
            "type": "object",
            "properties": {
                "province": {"type": "string"},
                "facility_type": {"type": "string"},
                "limit": {"type": "integer"},
            },
        },
    },
]

@app.post("/tools/list_fields")
async def mcp_tool_list_fields(_: dict = {}):
    await ensure_loaded()
    return {"ok": True, "tool": "list_fields", "data": COLUMNS}

@app.post("/tools/search_facilities")
async def mcp_tool_search_facilities(args: dict):
    province = args.get("province")
    facility_type = args.get("facility_type")
    limit = args.get("limit", 25)
    return await search_facilities(province=province, facility_type=facility_type, limit=limit)

from fastapi.responses import StreamingResponse

@app.get("/sse_once")
async def sse_once(_: Request):
    """
    One-shot SSE discovery for ChatGPT Custom Connector.
    Sends a single 'message' event with {event:'list_tools', data:{tools:[...]}} and closes.
    """
    async def event_stream():
        payload = {"event": "list_tools", "data": {"tools": TOOLS_MANIFEST}}
        yield "event: message\n"
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    headers = {
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "Content-Type": "text/event-stream; charset=utf-8",
    }

    return StreamingResponse(event_stream(), headers=headers, media_type="text/event-stream")
