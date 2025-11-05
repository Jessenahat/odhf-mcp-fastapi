# ODHF LODE MCP Server (FastAPI)

Public, no-auth **MCP server** exposing tools for the **Open Database of Healthcare Facilities (ODHF)** dataset
from Statistics Canada (a LODE dataset). Built for ChatGPT’s **Custom Connector** using the HTTP/SSE transport.

- **SSE discovery**: `/sse_once`
- **Tools**:
  - `list_fields` – list dataset columns
  - `search_facilities` – filter by province/territory and/or ODHF facility type

## Run locally

```bash
python -m venv .venv && . .venv/Scripts/activate  # (Windows)
# source .venv/bin/activate                       # (macOS/Linux)
pip install -r requirements.txt

# Option A) Put CSV in project root:
#   odhf_v1.1.csv
# Option B) Or set a download URL (first run will cache it):
#   set CSV_URL=https://<direct-statcan-csv-link>   # Windows PowerShell: $env:CSV_URL="..."
#   export CSV_URL="https://..."

uvicorn main:app --host 127.0.0.1 --port 8888 --reload
