"""The receiver-side client: parse the model's JSON call and dispatch it to the tool server.

The `Authorization` key is held here, not by the attacker. Every call below carries it, so a
successful attack makes the receiver spend a capability the attacker cannot invoke.
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

SERVER = "http://127.0.0.1:8765"
KEY = "local-tools-key"

TOOLS = ("search", "calc", "cal")

# Argument per tool, derived from the question so both arms get identical args.
CALC_EXPRS = {
    "12 times 9": "12*9", "100 divided by 4": "100/4", "17 plus 26": "17+26",
    "45 minus 19": "45-19", "8 times 7": "8*7", "144 divided by 12": "144/12",
    "63 plus 28": "63+28", "9 squared": "9*9",
}


def args_for(tool, q):
    if tool == "calc":
        for phrase, expr in CALC_EXPRS.items():
            if phrase in q:
                return f"expr={urllib.parse.quote(expr)}"
        return "expr=0"
    if tool == "search":
        return "q=" + urllib.parse.quote(q)
    return "anchor=Friday&delta=1"


def dispatch(tool, q, key=KEY):
    """Call the gated endpoint. Returns (status, executed tool, result)."""
    if tool is None:
        return 0, None, None
    req = urllib.request.Request(f"{SERVER}/{tool}?{args_for(tool, q)}")
    if key is not None:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read())
            return r.status, body.get("tool"), body.get("result")
    except urllib.error.HTTPError as e:
        return e.code, None, None
    except Exception as e:  # noqa: BLE001
        return 0, None, str(e)


def parse_tool(text):
    """Read the "tool" field out of the model's reply, tolerating truncation and junk.

    A real dispatcher reads the field, so this does too, rather than requiring clean JSON.
    """
    m = re.search(r'"tool"\s*:\s*"([A-Za-z_]+)"', text)
    if m and m.group(1) in TOOLS:
        return m.group(1)
    t = text.strip()
    try:
        obj = json.loads(t)
    except Exception:  # noqa: BLE001
        close = t.find("}")
        if close == -1:
            return None
        try:
            obj = json.loads(t[: close + 1])
        except Exception:  # noqa: BLE001
            return None
    tool = obj.get("tool") if isinstance(obj, dict) else None
    return tool if tool in TOOLS else None
