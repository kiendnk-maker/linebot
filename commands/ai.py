"""
commands/ai.py — AI workflow commands: /pro, /agent, /coder, /debate
"""
import asyncio
from agents_workflow import run_pro_workflow, run_agentic_loop, run_multi_agent_workflow, run_debate

# LINE's webhook timeout is 30s; keep debate under that.
_DEBATE_TIMEOUT = 25


async def handle_ai_command(user_id: str, cmd: str, arg: str) -> str | None:
    """Return response string or None if command not handled here."""

    if cmd == "pro":
        if not arg:
            return "⚠️ Vui lòng nhập yêu cầu phức tạp. Ví dụ: /pro Phân tích ưu nhược điểm của việc học Thạc sĩ tại Đài Loan"
        return await run_pro_workflow(user_id, arg)

    if cmd == "agent":
        if not arg:
            return "⚠️ Vui lòng nhập nhiệm vụ. Ví dụ: /agent Bây giờ là mấy giờ? Tính giúp tôi 12345 * 6789"
        return await run_agentic_loop(user_id, arg)

    if cmd == "coder":
        if not arg:
            return "⚠️ Vui lòng nhập yêu cầu. Ví dụ: /coder Viết hàm Python tính dãy Fibonacci"
        return await run_multi_agent_workflow(user_id, arg)

    if cmd == "debate":
        if not arg:
            return (
                "⚔️ Chế độ tranh luận — 2 AI đấu trí rồi trọng tài kết luận\n\n"
                "Cách dùng:\n"
                "/debate Nên học Thạc sĩ hay đi làm ngay?\n"
                "/debate 3 Đài Loan hay Nhật Bản để du học?\n\n"
                "Số đầu = số vòng tranh luận (mặc định 2)"
            )
        parts_d = arg.split(maxsplit=1)
        if parts_d[0].isdigit() and len(parts_d) > 1:
            rounds = min(int(parts_d[0]), 4)
            question = parts_d[1]
        else:
            rounds = 2
            question = arg
        try:
            return await asyncio.wait_for(
                run_debate(user_id, question, rounds),
                timeout=_DEBATE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            return "⚠️ Debate mất quá 25 giây. Thử giảm số vòng (mặc định /debate 2 <câu hỏi>)."

    return None
