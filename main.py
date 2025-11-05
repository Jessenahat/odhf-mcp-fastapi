from pathlib import Path
from typing import Optional, List, Dict, Any
import os
import json
import asyncio

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

# --- Setup ---
app = FastAPI(title="ODHF LODE MCP Server (MCP-compatible)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_FILE = Path(os.getenv("CSV_PATH", "odhf_v1.1.csv"))
DF: Optional[pd.DataFrame] = None
COLUMNS: Optional[List[str]] = None

def load_csv_safely(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, encoding="cp1252", errors="replace", low_memory=False)

async def ensure_loaded() -> None:
    global DF, COLUMNS
    if DF is not None:
        return
    df = load_csv_safely(CSV_FILE)
    if df is None or df.empty:
        raise HTTPException(
            status_code=400,
            detail=f"CSV not found or empty at {CSV_FILE.resolve()}. Upload odhf_v1.1.csv."
        )
    DF = df
    COLUMNS = list(df.columns)

def df_to_records_clean(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    safe = frame.replace([np.inf, -np.inf], np.nan)
    safe = safe.where(pd.notna(safe), None)
    return safe.to_dict(orient="records")

# --- Health ---
@app.get("/", response_class=PlainTextResponse)
async def root():
    rows = None if DF is None else int(len(DF))
    return f"ODHF MCP Server running! csv_found={CSV_FILE.exists()} rows={rows}"

# --- MCP tools: search & fetch ---
TOOLS_MANIFEST = [
    {
        "name": "search",
        "description": "Full-text facility search (returns array of results).",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
    },
    {
        "name": "fetch",
        "description": "Fetch details by document id from search.",
        "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}}
    },
]

@app.post("/tools/search")
async def mcp_tool_search(args: dict):
    await ensure_loaded()
    query = args.get("query", "")
    results = []
    if query:
        mask = DF["Facility Name"].astype(str).str.contains(query, case=False, na=False)
        subset = DF[mask].head(10)
        for _, row in subset.iterrows():
            results.append({
                "id": str(row.get("ID", row.name)),  # unique identifier: ID or fallback to row
                "title": str(row.get("Facility Name", "")),
                "url": f"https://your-mcp-server.com/detail/{row.get('ID', row.name)}"
            })
    # MCP content array response
    return {
        "content": [{
            "type": "text",
            "text": json.dumps({"results": results}, ensure_ascii=False)
        }]
    }

@app.post("/tools/fetch")
async def mcp_tool_fetch(args: dict):
    await ensure_loaded()
    doc_id = args.get("id")
    if not doc_id:
        raise HTTPException(status_code=400, detail="Must provide id")

    found = DF.loc[DF["ID"].astype(str) == str(doc_id)]
    if found.empty:
        try:
            found = DF.iloc[[int(doc_id)]]
        except Exception:
            return {"content": [{"type": "text", "text": "{}"}]}
    row = found.iloc[0]
    doc = {
        "id": str(row.get("ID", doc_id)),
        "title": str(row.get("Facility Name", "")),
        "text": str(row.to_json()),
        "url": f"https://your-mcp-server.com/detail/{row.get('ID', doc_id)}",
        "metadata": {col: str(row[col]) for col in DF.columns}
    }
    return {
        "content": [{
            "type": "text",
            "text": json.dumps(doc, ensure_ascii=False)
        }]
    }

# --- MCP discovery SSE ---
@app.get("/sse_once")
async def sse_once(_: Request):
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
