content = open("main.py").read()

old = (
    "                    else:\n"
    "                        model_key, model_id = await resolve_model(user_id, user_text)\n"
    "                logger.info(f'TEXT | user={user_id} | model={model_key} | text={user_text[:50]!r}')\n"
    "\n"
    "                        # Text dài không có câu hỏi → inject tóm tắt instruction\n"
    "                        if len(user_text) > 500 and \"?\" not in user_text and \"？\" not in user_text:\n"
    "                            await save_message(user_id, \"user\", user_text)\n"
    "                            history = [{\"role\": \"user\", \"content\": f\"Hãy tóm tắt nội dung sau:\\n{user_text}\"}]\n"
    "                        else:\n"
    "                            await save_message(user_id, \"user\", user_text)\n"
    "                            history = await get_history_with_summary(user_id)\n"
    "\n"
    "                        answer = await call_groq_text(history, model_id, model_key=model_key, user_id=user_id)\n"
    "                        await save_message(user_id, \"assistant\", answer)\n"
    "                        await maybe_summarize(user_id)\n"
    "                        reply  = answer"
)

new = (
    "                    else:\n"
    "                        model_key, model_id = await resolve_model(user_id, user_text)\n"
    "                        logger.info(f'TEXT | user={user_id} | model={model_key} | text={user_text[:50]!r}')\n"
    "\n"
    "                        # Text dài không có câu hỏi → inject tóm tắt instruction\n"
    "                        if len(user_text) > 500 and \"?\" not in user_text and \"？\" not in user_text:\n"
    "                            await save_message(user_id, \"user\", user_text)\n"
    "                            history = [{\"role\": \"user\", \"content\": f\"Hãy tóm tắt nội dung sau:\\n{user_text}\"}]\n"
    "                        else:\n"
    "                            await save_message(user_id, \"user\", user_text)\n"
    "                            history = await get_history_with_summary(user_id)\n"
    "\n"
    "                        answer = await call_groq_text(history, model_id, model_key=model_key, user_id=user_id)\n"
    "                        await save_message(user_id, \"assistant\", answer)\n"
    "                        await maybe_summarize(user_id)\n"
    "                        reply  = answer"
)

if old in content:
    content = content.replace(old, new)
    open("main.py", "w").write(content)
    print("OK - indentation fixed")
else:
    print("ERROR - pattern not found")
