from datetime import datetime
from zoneinfo import ZoneInfo
TZ = ZoneInfo("Asia/Taipei")

def get_current_time() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d %H:%M:%S (Taipei)")

def calculate_math(expression: str) -> str:
    try:
        if any(c not in "0123456789+-*/().% " for c in expression): return "Ky tu khong hop le."
        return str(eval(expression))
    except Exception as e: return f"Loi: {e}"

def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    conv = {("km","miles"):0.621371,("miles","km"):1.60934,("kg","lbs"):2.20462,("lbs","kg"):0.453592,
        ("celsius","fahrenheit"):lambda v:v*9/5+32,("fahrenheit","celsius"):lambda v:(v-32)*5/9,("twd","vnd"):780,("vnd","twd"):1/780}
    k = (from_unit.lower(), to_unit.lower())
    if k not in conv: return f"Khong ho tro {from_unit} -> {to_unit}"
    r = conv[k](value) if callable(conv[k]) else value * conv[k]
    return f"{value} {from_unit} = {r:.2f} {to_unit}"

AVAILABLE_TOOLS = {"get_current_time":get_current_time,"calculate_math":calculate_math,"convert_units":convert_units}

AGENT_TOOLS = [
    {"type":"function","function":{"name":"get_current_time","description":"Lay ngay gio hien tai (Taipei).","parameters":{"type":"object","properties":{},"required":[]}}},
    {"type":"function","function":{"name":"calculate_math","description":"Tinh toan hoc.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}},
    {"type":"function","function":{"name":"convert_units","description":"Doi don vi (km/miles, kg/lbs, TWD/VND).","parameters":{"type":"object","properties":{"value":{"type":"number"},"from_unit":{"type":"string"},"to_unit":{"type":"string"}},"required":["value","from_unit","to_unit"]}}}
]
