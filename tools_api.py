"""
tools_api.py — Tool definitions for Mistral function calling
Expanded: web_search, URL fetch, unit conversion, language detection
"""
import json
import urllib.request
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo
from simpleeval import simple_eval, InvalidExpression

TZ = ZoneInfo("Asia/Taipei")


def get_current_time() -> str:
    """Return current time in Asia/Taipei timezone."""
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S (Taipei)")


def calculate_math(expression: str) -> str:
    """Safe math evaluation using simpleeval — no eval() or exec()."""
    try:
        result = simple_eval(expression)
        return str(result)
    except InvalidExpression as e:
        return f"Lỗi: Biểu thức không hợp lệ — {str(e)}"
    except Exception as e:
        return f"Lỗi tính toán: {str(e)}"


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units."""
    conversions = {
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("celsius", "fahrenheit"): lambda v: v * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda v: (v - 32) * 5/9,
        ("twd", "vnd"): 780,  # Approximate TWD→VND
        ("vnd", "twd"): 1/780,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        return f"Không hỗ trợ chuyển đổi {from_unit} → {to_unit}"
    factor = conversions[key]
    if callable(factor):
        result = factor(value)
    else:
        result = value * factor
    return f"{value} {from_unit} = {result:.2f} {to_unit}"


# ═══════════════════════════════════════════════════════════════
# TOOL REGISTRY — Python functions mapped by name
# ═══════════════════════════════════════════════════════════════
AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "calculate_math": calculate_math,
    "convert_units": convert_units,
}

# ═══════════════════════════════════════════════════════════════
# AGENT_TOOLS — OpenAI-compatible function calling schema
# Used by agents_workflow.py for Mistral tool calling
# ═══════════════════════════════════════════════════════════════
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get current date and time in Asia/Taipei timezone.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Calculate a math expression (e.g. 15 * 24 + 100). Returns exact result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert between units (km/miles, kg/lbs, celsius/fahrenheit, TWD/VND).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Numeric value to convert"},
                    "from_unit": {"type": "string", "description": "Source unit"},
                    "to_unit": {"type": "string", "description": "Target unit"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]
