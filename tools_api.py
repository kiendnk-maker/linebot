import json
import urllib.request
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo
TZ = ZoneInfo('Asia/Taipei')

def get_current_time() -> str:
    """Trả về thời gian hiện tại của hệ thống."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculate_math(expression: str) -> str:
    """Tính toán biểu thức toán học an toàn."""
    try:
        # Chỉ cho phép tính toán cơ bản để tránh rủi ro bảo mật
        allowed_chars = "0123456789+-*/(). "
        if any(char not in allowed_chars for char in expression):
            return "Lỗi: Biểu thức chứa ký tự không hợp lệ."
        return str(eval(expression))
    except Exception as e:
        return f"Lỗi tính toán: {str(e)}"

AVAILABLE_TOOLS = {
    "get_current_time": get_current_time,
    "calculate_math": calculate_math
}

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Lấy ngày và giờ hiện tại của hệ thống.",
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_math",
            "description": "Tính toán một biểu thức toán học (ví dụ: 15 * 24 + 100). Trả về kết quả chính xác.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Biểu thức toán học cần tính"}
                },
                "required": ["expression"]
            }
        }
    }
]

