from groq import AsyncGroq
import os
import json
import logging
# Import không gian bộ nhớ của ứng dụng gốc
import main
from tools_api import AVAILABLE_TOOLS, AGENT_TOOLS

logger = logging.getLogger(__name__)

async def run_multi_agent_workflow(user_id: str, task: str) -> str:
    planner_prompt = f"Bạn là một System Architect (Kiến trúc sư phần mềm). Nhiệm vụ của bạn là phân tích và lập kế hoạch từng bước rõ ràng để giải quyết yêu cầu sau:\n\nYÊU CẦU: {task}\n\nChỉ trả về các bước lập trình, không cần viết code."
    plan = await main.call_groq_text(
        history=[{"role": "user", "content": planner_prompt}],
        model_id=main.MODEL_REGISTRY["qwen"]["model_id"],
        model_key="qwen",
        user_id=user_id
    )

    coder_prompt = f"Bạn là một Senior Developer. Dựa vào bản thiết kế dưới đây, hãy viết mã nguồn hoàn chỉnh và tối ưu nhất:\n\n[BẢN THIẾT KẾ]\n{plan}\n\n[YÊU CẦU GỐC]\n{task}"
    code = await main.call_groq_text(
        history=[{"role": "user", "content": coder_prompt}],
        model_id=main.MODEL_REGISTRY["llama70b"]["model_id"],
        model_key="llama70b",
        user_id=user_id
    )

    reviewer_prompt = f"Bạn là một QA & Security Engineer khắt khe. Hãy kiểm tra đoạn mã sau xem có lỗi logic, lỗ hổng bảo mật, hoặc điểm nào cần tối ưu hiệu năng không. Đưa ra nhận xét và cách sửa (nếu có):\n\n[MÃ NGUỒN CẦN DUYỆT]\n{code}"
    review = await main.call_groq_text(
        history=[{"role": "user", "content": reviewer_prompt}],
        model_id=main.MODEL_REGISTRY["gpt120b"]["model_id"],
        model_key="gpt120b",
        user_id=user_id
    )

    final_output = (
        f"🚀 [MULTI-AGENT WORKFLOW THÀNH CÔNG]\n"
        f"Tác vụ: {task}\n"
        f"──────────────\n"
        f"📝 1. PLANNER (Qwen):\nĐã lập kế hoạch {len(plan.split())} từ.\n\n"
        f"💻 2. CODER (LLaMA 70B):\n{code}\n\n"
        f"🔎 3. REVIEWER (GPT-120B):\n{review}"
    )
    return final_output


async def run_pro_workflow(user_id: str, task: str) -> str:
    # Bước 1: Tư duy sâu đa chiều (Reasoning)
    thinker_prompt = (
        f"Bạn là một bộ não siêu việt chuyên phân tích vấn đề. "
        f"Hãy phân tích yêu cầu dưới đây đa chiều (logic, sáng tạo, kỹ thuật, cảm xúc, văn hóa... tùy bối cảnh). "
        f"Đưa ra luồng suy nghĩ chi tiết, các góc nhìn cần lưu ý và lập dàn ý nội dung tốt nhất để giải quyết:\n\n"
        f"YÊU CẦU: {task}"
    )
    thought_process = await main.call_groq_text(
        history=[{"role": "user", "content": thinker_prompt}],
        model_id=main.MODEL_REGISTRY["qwen"]["model_id"],
        model_key="qwen",
        user_id=user_id
    )

    # Bước 2: Chắp bút và hoàn thiện (Synthesizing)
    writer_prompt = (
        f"Bạn là một chuyên gia giao tiếp và học giả ngôn ngữ xuất chúng (như Claude). "
        f"Dựa trên luồng suy nghĩ và dàn ý dưới đây, hãy viết câu trả lời cuối cùng trực tiếp cho người dùng. "
        f"Tự động điều chỉnh văn phong (code, học thuật, hài hước, trang trọng) cho phù hợp nhất với bản chất yêu cầu gốc:\n\n"
        f"[PHÂN TÍCH & DÀN Ý CỦA BỘ NÃO]\n{thought_process}\n\n"
        f"[YÊU CẦU GỐC CỦA NGƯỜI DÙNG]\n{task}"
    )
    final_answer = await main.call_groq_text(
        history=[{"role": "user", "content": writer_prompt}],
        model_id=main.MODEL_REGISTRY["gpt120b"]["model_id"],
        model_key="gpt120b",
        user_id=user_id
    )

    return f"🧠 [CHẾ ĐỘ PRO - DEEP THINKING]\n──────────────\n{final_answer}"


async def run_agentic_loop(user_id: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": "Bạn là một Agent tự trị. Bạn CÓ THỂ sử dụng các công cụ được cung cấp để tìm kiếm thông tin hoặc tính toán trước khi trả lời. Hãy suy nghĩ từng bước."},
        {"role": "user", "content": prompt}
    ]
    
    max_iterations = 5 # Giới hạn số vòng lặp để tránh treo hệ thống
    client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY", ""))
    
    for iteration in range(max_iterations):
        try:
            # Gọi Groq API (sử dụng Llama 3.3 70B vì nó hỗ trợ Tool Calling rất tốt)
            resp = await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=AGENT_TOOLS,
                tool_choice="auto"
            )
            
            response_message = resp.choices[0].message
            
            # Nếu LLM không gọi công cụ nào -> Nó đã có câu trả lời cuối cùng
            if not response_message.tool_calls:
                return f"🤖 [AGENTIC MODE]\n──────────────\n{response_message.content}"
            
            # Nếu LLM quyết định gọi công cụ -> Thực thi vòng lặp
            messages.append(response_message) # Lưu lại quyết định của AI
            
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                # Thực thi hàm Python tương ứng
                func_to_call = AVAILABLE_TOOLS.get(func_name)
                if func_to_call:
                    if func_name == "calculate_math":
                        func_result = func_to_call(func_args.get("expression"))
                    else:
                        func_result = func_to_call()
                else:
                    func_result = f"Error: Tool {func_name} not found."
                
                # Gửi kết quả của công cụ ngược lại cho AI
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(func_result)
                })
                
        except Exception as e:
            return f"⚠️ Lỗi Agent: {str(e)}"
            
    return "⚠️ Lỗi: Agent đã vượt quá số vòng lặp tối đa mà không thể hoàn thành nhiệm vụ."
