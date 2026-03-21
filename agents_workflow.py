"""agents_workflow.py — /pro /coder /agent /debate using dual provider and Real-time Push"""
import os, json, inspect, asyncio, httpx
from openai import AsyncOpenAI
from tools_api import AVAILABLE_TOOLS, AGENT_TOOLS

_mistral = AsyncOpenAI(api_key=os.environ.get("MISTRAL_API_KEY",""), base_url="https://api.mistral.ai/v1")
_groq = AsyncOpenAI(api_key=os.environ.get("GROQ_API_KEY",""), base_url="https://api.groq.com/openai/v1")

async def _push_msg(user_id: str, text: str):
    token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "")
    if not token: return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    data = {"to": user_id, "messages": [{"type": "text", "text": text[:5000]}]}
    async with httpx.AsyncClient() as client:
        await client.post(url, headers=headers, json=data)

async def _call(client, model, prompt, max_tokens=1500, **kw):
    try:
        r = await client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=max_tokens, **kw)
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return f"⚠️ [{model}]: {str(e)[:120]}"

# ── PRO ──
async def run_pro_workflow(user_id: str, task: str) -> str:
    asyncio.create_task(_bg_pro(user_id, task))
    return "🧠 [PRO] Đang phân tích chuyên sâu..."

async def _bg_pro(user_id: str, task: str):
    await _push_msg(user_id, "⏳ Small 4 (High Reasoning) đang lập luận...")
    thought = await _call(_mistral, "mistral-small-latest", f"Phan tich da chieu. YEU CAU: {task}", temperature=0.7, extra_body={"reasoning_effort":"high"})
    await _push_msg(user_id, "⏳ Large đang tổng hợp kết quả...")
    final = await _call(_mistral, "mistral-large-latest", f"Dua tren phan tich, viet cau tra loi.\n[PHAN TICH]\n{thought}\n[YEU CAU]\n{task}", temperature=0.6)
    await _push_msg(user_id, f"🧠 [PRO — DEEP THINKING]\n──────────────\n{final}")

# ── CODER ──
async def run_multi_agent_workflow(user_id: str, task: str) -> str:
    asyncio.create_task(_bg_coder(user_id, task))
    return "🚀 [CODER] Đang khởi động quy trình phát triển..."

async def _bg_coder(user_id: str, task: str):
    await _push_msg(user_id, "⏳ System Architect đang lên kế hoạch...")
    plan = await _call(_mistral, "mistral-small-latest", f"Ban la System Architect. Lap ke hoach.\nYEU CAU: {task}", temperature=0.3)
    await _push_msg(user_id, "⏳ Senior Dev đang viết code (High Reasoning)...")
    code = await _call(_mistral, "mistral-small-latest", f"Ban la Senior Dev. Viet code.\n[THIET KE]\n{plan}\n[YEU CAU]\n{task}", temperature=0.2, extra_body={"reasoning_effort":"high"})
    await _push_msg(user_id, "⏳ QA Engineer đang review code...")
    review = await _call(_mistral, "mistral-large-latest", f"Ban la QA. Kiem tra code.\n[CODE]\n{code}", temperature=0.3)
    await _push_msg(user_id, f"💻 CODE:\n{code}\n\n🔎 REVIEW:\n{review}")

# ── AGENT ──
async def run_agentic_loop(user_id: str, prompt: str) -> str:
    asyncio.create_task(_bg_agent(user_id, prompt))
    return "🤖 [AGENT] Qwen3 đang xử lý..."

async def _bg_agent(user_id: str, prompt: str):
    messages = [{"role":"system","content":"Ban la Agent tu tri. Dung tools khi can."}, {"role":"user","content":prompt}]
    for _ in range(5):
        try:
            r = await _groq.chat.completions.create(model="qwen-2.5-32b", messages=messages, tools=AGENT_TOOLS, tool_choice="auto")
            msg = r.choices[0].message
            if not msg.tool_calls:
                await _push_msg(user_id, f"🤖 [AGENT — Qwen3]\n──────────────\n{msg.content}")
                return
            messages.append(msg)
            for tc in msg.tool_calls:
                fn = AVAILABLE_TOOLS.get(tc.function.name)
                args = json.loads(tc.function.arguments)
                result = fn() if len(inspect.signature(fn).parameters)==0 else fn(**args) if fn else f"Tool not found"
                messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.function.name,"content":str(result)})
                await _push_msg(user_id, f"🛠️ Đang dùng tool: {tc.function.name}...")
        except Exception:
            try:
                r = await _mistral.chat.completions.create(model="mistral-small-latest", messages=messages)
                await _push_msg(user_id, f"🤖 [AGENT — Fallback Small]\n──────────────\n{r.choices[0].message.content}")
            except Exception as e2:
                await _push_msg(user_id, f"⚠️ Agent error: {str(e2)[:120]}")
            return
    await _push_msg(user_id, "⚠️ Agent vượt quá số vòng lặp.")

# ── DEBATE ──
async def run_debate(user_id: str, question: str, rounds: int = 2) -> str:
    asyncio.create_task(_bg_debate(user_id, question, rounds))
    return f"⚔️ VÕ ĐÀI AI ĐÃ MỞ!\nChủ đề: {question}\nSố hiệp: {rounds}"

async def _bg_debate(user_id: str, question: str, rounds: int):
    a_hist, b_hist = [], []
    for i in range(rounds):
        rn = i + 1
        a_prompt = f"Phan tich quan diem:\n{question}" if i==0 else f"Doi phuong noi:\n{b_says}\nBao ve hoac dieu chinh quan diem."
        a_says = await _call(_mistral, "mistral-large-latest", a_prompt, temperature=0.7)
        a_hist.append(a_says)
        await _push_msg(user_id, f"🔵 [Hiệp {rn}] Advocate (Mistral Large):\n\n{a_says}")
        
        b_says = await _call(_groq, "qwen-2.5-32b", f"Phan bien quan diem, tim diem yeu:\n{a_says}", temperature=0.7)
        b_hist.append(b_says)
        await _push_msg(user_id, f"🔴 [Hiệp {rn}] Critic (Qwen3):\n\n{b_says}")
        
    await _push_msg(user_id, "⚖️ Trọng tài đang tổng hợp...")
    judge_ctx = "\n".join(f"R{i+1} A:{a_hist[i][:400]} B:{b_hist[i][:400]}" for i in range(rounds))
    verdict = await _call(_mistral, "mistral-small-latest", f"Tong hop 2 quan diem.\nCau hoi: {question}\n{judge_ctx}", extra_body={"reasoning_effort":"high"})
    await _push_msg(user_id, f"🏆 KẾT LUẬN:\n\n{verdict}\n\n🏁 KẾT THÚC.")
