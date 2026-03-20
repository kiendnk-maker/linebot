"""
agents_workflow.py — AI Agent workflows using Mistral API
Updated: Magistral for reasoning, Codestral for coding, proper tool calling
"""
import os
import json
import logging
from openai import AsyncOpenAI
from tools_api import AVAILABLE_TOOLS, AGENT_TOOLS

logger = logging.getLogger(__name__)

# Reuse global client
_client = AsyncOpenAI(
    api_key=os.environ.get("MISTRAL_API_KEY", ""),
    base_url="https://api.mistral.ai/v1",
)


async def _call(model_id: str, messages: list[dict], max_tokens: int = 1500, **kwargs) -> str:
    """Helper: single Mistral API call with error handling."""
    try:
        resp = await _client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
            **kwargs,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ Lỗi [{model_id}]: {str(e)[:150]}"


async def run_multi_agent_workflow(user_id: str, task: str) -> str:
    """
    /coder — 3-agent pipeline: Planner → Coder → Reviewer
    Uses Codestral for code generation (specialized coding model).
    """
    # Import here to avoid circular dependency
    import main

    # Step 1: Planner (Small — fast, cheap)
    plan = await _call(
        "mistral-small-latest",
        [{"role": "user", "content": (
            f"Bạn là System Architect. Phân tích và lập kế hoạch từng bước "
            f"để giải quyết yêu cầu sau. Chỉ trả về các bước, không viết code.\n\n"
            f"YÊU CẦU: {task}"
        )}],
        max_tokens=800,
        temperature=0.3,
    )

    # Step 2: Coder (Codestral — specialized for code)
    code = await _call(
        "codestral-latest",
        [{"role": "user", "content": (
            f"Bạn là Senior Developer. Dựa vào bản thiết kế, viết mã nguồn "
            f"hoàn chỉnh và tối ưu.\n\n"
            f"[BẢN THIẾT KẾ]\n{plan}\n\n[YÊU CẦU GỐC]\n{task}"
        )}],
        max_tokens=2000,
        temperature=0.2,
    )

    # Step 3: Reviewer (Large — broad knowledge for review)
    review = await _call(
        "mistral-large-latest",
        [{"role": "user", "content": (
            f"Bạn là QA & Security Engineer. Kiểm tra đoạn mã sau: "
            f"lỗi logic, lỗ hổng bảo mật, điểm tối ưu. "
            f"Đưa ra nhận xét và cách sửa nếu có.\n\n"
            f"[MÃ NGUỒN]\n{code}"
        )}],
        max_tokens=1000,
        temperature=0.3,
    )

    return (
        f"🚀 [MULTI-AGENT WORKFLOW]\n"
        f"Tác vụ: {task}\n"
        f"──────────────\n"
        f"📝 PLANNER (Small 4):\n{plan[:300]}...\n\n"
        f"💻 CODER (Codestral):\n{code}\n\n"
        f"🔎 REVIEWER (Large 3):\n{review}"
    )


async def run_pro_workflow(user_id: str, task: str) -> str:
    """
    /pro — Deep thinking: Magistral reasoning → Large synthesis
    Uses Magistral for deep analysis, Large for final polished answer.
    """
    # Step 1: Deep thinking with Magistral (reasoning model)
    thought = await _call(
        "magistral-medium-latest",
        [{"role": "user", "content": (
            f"Phân tích yêu cầu sau đa chiều (logic, sáng tạo, kỹ thuật, "
            f"văn hóa). Đưa ra luồng suy nghĩ chi tiết và dàn ý tốt nhất.\n\n"
            f"YÊU CẦU: {task}"
        )}],
        max_tokens=1500,
        temperature=0.7,
    )

    # Step 2: Synthesis with Large (polished output)
    final = await _call(
        "mistral-large-latest",
        [{"role": "user", "content": (
            f"Dựa trên phân tích dưới đây, viết câu trả lời cuối cùng "
            f"cho người dùng. Tự điều chỉnh văn phong phù hợp.\n\n"
            f"[PHÂN TÍCH]\n{thought}\n\n[YÊU CẦU GỐC]\n{task}"
        )}],
        max_tokens=1500,
        temperature=0.6,
    )

    return f"🧠 [PRO — DEEP THINKING]\n──────────────\n{final}"


async def run_agentic_loop(user_id: str, prompt: str) -> str:
    """
    /agent — Autonomous agent with tool calling loop.
    Uses Mistral Large for best tool-calling support.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là một Agent tự trị. Sử dụng các công cụ được cung cấp "
                "khi cần tìm thông tin hoặc tính toán. Suy nghĩ từng bước."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    max_iterations = 5

    for _iteration in range(max_iterations):
        try:
            resp = await _client.chat.completions.create(
                model="mistral-large-latest",
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto",
            )
            response_message = resp.choices[0].message

            # No tool calls → final answer
            if not response_message.tool_calls:
                return f"🤖 [AGENTIC MODE]\n──────────────\n{response_message.content}"

            # Execute tool calls
            messages.append(response_message)
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                func_to_call = AVAILABLE_TOOLS.get(func_name)
                if func_to_call:
                    # Dynamic argument passing
                    import inspect
                    sig = inspect.signature(func_to_call)
                    if len(sig.parameters) == 0:
                        func_result = func_to_call()
                    else:
                        func_result = func_to_call(**func_args)
                else:
                    func_result = f"Error: Tool {func_name} not found."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(func_result),
                })

        except Exception as e:
            return f"⚠️ Lỗi Agent: {str(e)[:200]}"

    return "⚠️ Agent đã vượt quá số vòng lặp tối đa."


async def run_debate(user_id: str, topic: str, rounds: int = 2) -> str:
    from llm_core import call_mistral_text, MODEL_REGISTRY
    
    # AI 1 dùng Large, AI 2 dùng Magistral (nếu có, không thì dùng Large)
    model_a = MODEL_REGISTRY.get("large", {}).get("model_id", "mistral-large-latest")
    model_b = MODEL_REGISTRY.get("reason", {}).get("model_id", "mistral-large-latest")
    
    transcript = f"⚔️ CHỦ ĐỀ TRANH LUẬN: {topic}\n⚖️ Số vòng: {rounds}\n\n"
    
    try:
        for i in range(rounds):
            prompt_a = f"Bạn là AI Ủng Hộ (Proponent). Chủ đề: {topic}. Hãy đưa ra luận điểm sắc bén. Nếu đối thủ đã nói, hãy phản bác mạnh mẽ. Lịch sử tranh luận:\n{transcript}"
            reply_a = await call_mistral_text(prompt_a, model=model_a, max_tokens=600)
            transcript += f"🟢 [AI Ủng Hộ]:\n{reply_a}\n\n"
            
            prompt_b = f"Bạn là AI Phản Đối (Opponent). Chủ đề: {topic}. Hãy phản bác gay gắt luận điểm của AI Ủng Hộ và đưa ra góc nhìn trái chiều. Lịch sử tranh luận:\n{transcript}"
            reply_b = await call_mistral_text(prompt_b, model=model_b, max_tokens=600)
            transcript += f"🔴 [AI Phản Đối]:\n{reply_b}\n\n"
            
        transcript += "🏁 KẾT THÚC TRANH LUẬN."
        return transcript
    except Exception as e:
        return f"Lỗi trong quá trình tranh luận: {str(e)}"
