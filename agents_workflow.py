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
    import asyncio
    # Phản hồi ngay cho LINE để né Timeout, đẩy phần việc nặng vào background
    asyncio.create_task(_background_debate(user_id, topic, rounds))
    return f"⚔️ VÕ ĐÀI AI ĐÃ MỞ (Chế độ Trọng tài)!\n⚖️ Chủ đề: {topic}\n🔄 Số hiệp: {rounds}\n\n🤖 Các chuyên gia đang vào vị trí. Bạn sẽ nhận được từng hiệp đấu ngay bây giờ..."

async def _background_debate(user_id: str, topic: str, rounds: int):
    import os
    import httpx
    from openai import AsyncOpenAI
    from llm_core import MODEL_REGISTRY
    
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    
    # Hàm bắn tin nhắn Push trực tiếp tới user
    async def push_msg(text):
        if not token: return
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        data = {"to": user_id, "messages": [{"type": "text", "text": text[:5000]}]}
        async with httpx.AsyncClient() as client:
            await client.post(url, headers=headers, json=data)

    client = AsyncOpenAI(
        api_key=os.environ.get("MISTRAL_API_KEY", ""),
        base_url="https://api.mistral.ai/v1"
    )
    
    model_a = MODEL_REGISTRY.get("large", {}).get("model_id", "mistral-large-latest")
    model_b = MODEL_REGISTRY.get("reason", {}).get("model_id", "mistral-large-latest")
    
    a_history = []
    b_history = []
    
    try:
        a_says = ""
        b_says = ""
        for i in range(rounds):
            round_num = i + 1
            
            # ── Model A: Advocate (Large) ──
            if i == 0:
                a_prompt = f"Bạn là chuyên gia phân tích (Advocate). Đưa ra quan điểm rõ ràng, có luận cứ cho câu hỏi sau:\n\n{topic}"
            else:
                a_prompt = f"Đối phương (Critic) vừa phản biện:\n{b_says}\n\nHãy bảo vệ hoặc điều chỉnh quan điểm của bạn. Thừa nhận điểm đúng của đối phương nếu có."
                
            resp_a = await client.chat.completions.create(model=model_a, messages=[{"role": "user", "content": a_prompt}], temperature=0.7)
            a_says = resp_a.choices[0].message.content
            a_history.append(a_says)
            await push_msg(f"🔵 [Hiệp {round_num}] Advocate (Ủng hộ):\n\n{a_says}")
            
            # ── Model B: Critic (Magistral/Reason) ──
            b_prompt = f"Bạn là chuyên gia phản biện (Critic). Phân tích quan điểm sau, tìm điểm yếu, thiếu sót, hoặc góc nhìn bị bỏ qua. Đưa ra phản biện có logic.\n\nQuan điểm của Advocate:\n{a_says}"
            
            resp_b = await client.chat.completions.create(model=model_b, messages=[{"role": "user", "content": b_prompt}], temperature=0.7)
            b_says = resp_b.choices[0].message.content
            b_history.append(b_says)
            await push_msg(f"🔴 [Hiệp {round_num}] Critic (Phản biện):\n\n{b_says}")
            
        # ── Judge: Synthesize (Large) ──
        await push_msg("⚖️ Trọng tài đang tổng hợp kết luận...")
        
        judge_prompt = f"Bạn là trọng tài trung lập. Đọc cuộc tranh luận sau và đưa ra kết luận cuối cùng, tổng hợp điểm mạnh của cả hai bên.\n\nCâu hỏi gốc: {topic}\n\n"
        for i in range(rounds):
            judge_prompt += f"--- Hiệp {i+1} ---\nAdvocate: {a_history[i][:800]}\nCritic: {b_history[i][:800]}\n\n"
        judge_prompt += "Kết luận cuối cùng (tổng hợp cả hai bên):"
        
        resp_judge = await client.chat.completions.create(model=model_a, messages=[{"role": "user", "content": judge_prompt}], temperature=0.5)
        verdict = resp_judge.choices[0].message.content
        
        await push_msg(f"🏆 KẾT LUẬN TỪ TRỌNG TÀI:\n\n{verdict}\n\n🏁 KẾT THÚC TRANH LUẬN.")
        
    except Exception as e:
        await push_msg(f"⚠️ Lỗi hệ thống tranh luận: {str(e)}")
