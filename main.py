import os
import sys
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI()

# ✅ CORS enabled (required for grader)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- Health Check ----------------
@app.get("/")
def health():
    return {"status": "ok"}


# ---------------- Models ----------------

class CodeRequest(BaseModel):
    code: str


class CodeResponse(BaseModel):
    error: List[int]
    result: str


# ---------------- Tool Function ----------------

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout


# ---------------- AI Error Analyzer ----------------

def analyze_error_with_ai(code: str, tb: str) -> List[int]:

    prompt = f"""
Analyze the Python code and traceback below.
Return ONLY a JSON object in this format:

{{
  "error_lines": [line_numbers]
}}

CODE:
{code}

TRACEBACK:
{tb}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "error_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "error_lines": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    },
                    "required": ["error_lines"],
                    "additionalProperties": False
                }
            }
        }
    )

    return response.output_parsed["error_lines"]


# ---------------- Main Endpoint ----------------

@app.post("/code-interpreter", response_model=CodeResponse)
async def code_interpreter(request: CodeRequest):

    execution = execute_python_code(request.code)

    # ✅ If success → don't call AI
    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"]
        }

    # ❌ If error → call AI
    error_lines = analyze_error_with_ai(
        request.code,
        execution["output"]
    )

    return {
        "error": error_lines,
        "result": execution["output"]
    }


# ---------------- Render Safe Startup ----------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)