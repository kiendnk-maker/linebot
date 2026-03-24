"""
command_handler.py — Dispatcher: routes /commands to submodule handlers.

Submodules:
  commands/ai.py       — /pro, /agent, /coder, /debate
  commands/data.py     — /mn, /rag, /audio
  commands/settings.py — /vi, /tw, /usage, /nuke, /clear, /auto, /model,
                          /models, /long, /short, /tokens, /profile, /remind
  google_workspace.py  — /login, /cal, /mail, /ls, /block, /unblock, /wedding
"""
from llm_core import MODEL_REGISTRY, call_mistral_text
from database import set_user_model
from google_workspace import handle_workspace_command
from commands.ai import handle_ai_command
from commands.data import handle_data_command
from commands.settings import handle_settings_command


async def handle_command(user_id: str, text: str) -> str | None:
    if not text.startswith("/"):
        return None

    parts = text[1:].strip().split(maxsplit=1)
    cmd   = parts[0].lower()
    arg   = parts[1].strip() if len(parts) > 1 else ""

    # Google Workspace commands (/login, /cal, /mail, /ls, /block, /unblock, /wedding)
    ws_reply = await handle_workspace_command(cmd, arg, user_id)
    if ws_reply:
        return ws_reply

    # AI workflow commands
    result = await handle_ai_command(user_id, cmd, arg)
    if result is not None:
        return result

    # Data management commands
    result = await handle_data_command(user_id, cmd, arg)
    if result is not None:
        return result

    # Settings & system commands
    result = await handle_settings_command(user_id, cmd, arg)
    if result is not None:
        return result

    # Model shortcut: /small, /large, /coder, /vision, /qwen3, etc.
    if cmd in MODEL_REGISTRY:
        await set_user_model(user_id, cmd)
        cfg = MODEL_REGISTRY[cmd]
        if arg:
            answer = await call_mistral_text(
                [{"role": "user", "content": arg}],
                cfg["model_id"],
                model_key=cmd,
            )
            return f"[{cfg['display']}]\n{answer}"
        return f"✅ 已切換至 {cfg['display']}。\n輸入 /auto 返回自動模式。"

    return f"❓ 指令 /{cmd} 無效。請輸入 /models 查看。"
