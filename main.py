# --- START OF FILE main.py ---

import asyncio
import os
import logging
from dotenv import load_dotenv
from datetime import datetime

from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command

from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- Константы ---
DB_FILE = 'qa.db'
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LOG_FILE = "llm_requests.log"

if not ANTHROPIC_API_KEY or not TELEGRAM_BOT_TOKEN:
    raise ValueError("Необходимо установить ANTHROPIC_API_KEY и TELEGRAM_BOT_TOKEN в .env файле")

# ... (Класс FileCallbackHandler остается без изменений) ...
class FileCallbackHandler(BaseCallbackHandler):
    """Логирует полный цикл работы агента, уделяя особое внимание запросам к LLM."""
    def __init__(self, filename: str = LOG_FILE):
        self.file = open(filename, 'a', encoding='utf-8')

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        self.file.write(f"\n\n---\n\n## 🚀 Новая сессия: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        user_input = "Не удалось определить входной запрос"
        if isinstance(inputs, dict):
            user_input = inputs.get('input', user_input)
        self.file.write(f"**Входной запрос пользователя:**\n```\n{user_input}\n```\n")
        self.file.flush()

    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        self.file.write(f"\n### ➡️ Запрос к LLM\n\n")
        for i, prompt in enumerate(prompts):
            self.file.write(f"**Промпт {i+1}:**\n```text\n{prompt}\n```\n")
        self.file.flush()

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.file.write(f"\n### ⬅️ Ответ от LLM\n\n")
        try:
            raw_response = response.generations[0][0].text
            self.file.write(f"**Ответ модели (raw):**\n```text\n{raw_response}\n```\n")
        except (AttributeError, IndexError):
            self.file.write(f"**Не удалось извлечь ответ из объекта LLMResult.**\n")
        self.file.flush()

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        self.file.write(f"\n### 🛠️ Действие Агента\n\n")
        self.file.write(f"**Инструмент:** `{action.tool}`\n")
        self.file.write(f"**Входные данные (SQL):**\n```sql\n{action.tool_input}\n```\n")
        self.file.flush()

    def on_tool_end(self, output: str, **kwargs) -> None:
        self.file.write(f"\n### 📊 Результат Инструмента\n\n")
        self.file.write(f"**Полученные данные из БД:**\n```\n{output}\n```\n")
        self.file.flush()

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        self.file.write(f"\n### ✅ Финальный Ответ Агента\n\n")
        self.file.write(f"**Итоговый ответ для пользователя:**\n```\n{finish.return_values.get('output')}\n```\n")
        self.file.flush()

    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        self.file.write(f"\n--- 🏁 Сессия завершена ---\n")
        self.file.flush()

try:
    logging.info("Инициализация 'двигателя' бота...")
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)

    # --- ИЗМЕНЕНИЕ: Добавляем требование цитирования ---
    CUSTOM_PROMPT_TEMPLATE = """
Ты — агент, созданный для взаимодействия с SQL базой данных. Твоя задача - отвечать на вопросы пользователя на РУССКОМ ЯЗЫКЕ.
База данных, к которой ты обращаешься, — это лог чата в Telegram (Валенсия, Испания).
Твоя основная функция — запрашивать базу данных. Ты должен в первую очередь пытаться найти ответ в ней.
Если после выполнения SQL-запроса ты получаешь пустой результат, твой финальный ответ должен быть: "К сожалению, в истории чата не нашлось информации по вашему запросу."
Не придумывай ответ на основе своих общих знаний, если база данных ничего не вернула.

**ВАЖНОЕ ПРАВИЛО ФОРМАТИРОВАНИЯ ОТВЕТА:**
Твой финальный ответ должен состоять из двух частей:
1.  **Сводка:** Краткий, лаконичный вывод на русском языке.
2.  **Источники:** Под заголовком "**Исходные сообщения:**" приведи 2-3 наиболее релевантных сообщения из базы данных, на основе которых сделана сводка. Каждое сообщение должно быть отформатировано так: `Автор (Дата): Текст сообщения`.

У тебя есть доступ к следующим инструментам:
{tools}

Используй СТРОГО следующий формат:

Question: the input question you must answer
Thought: твои рассуждения о том, что делать дальше, на русском языке.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (Эта последовательность Thought/Action/Action Input/Observation может повторяться N раз)
Thought: Теперь я знаю финальный ответ и сформулирую его на русском языке, следуя правилу форматирования.
Final Answer: финальный ответ, состоящий из сводки и источников.

Начинай!

Question: {input}
Thought:{agent_scratchpad}
"""
    
    prompt = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        prompt=prompt,
        verbose=True,
        handle_parsing_errors=True
    )

    logging.info("'Двигатель' бота (SQL-агент с Claude и гибридным промптом) успешно инициализирован.")
except Exception as e:
    logging.critical(f"Не удалось инициализировать SQL-агента! Ошибка: {e}")
    agent_executor = None

bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.answer(
        "Привет! Я — ChatGeist, аналитик истории этого чата.\n"
        "Задайте мне вопрос, и я постараюсь найти ответ в базе данных сообщений.\n\n"
        "Например: 'Кто самый активный участник?' или 'Покажи последние 5 сообщений от Ивана'."
    )

@dp.message()
async def handle_message(message: types.Message):
    if not agent_executor:
        await message.answer("Извините, аналитический модуль не запущен из-за ошибки. Проверьте логи сервера.")
        return
    if not message.text:
        return

    thinking_message = await message.answer("Анализирую ваш запрос...")

    try:
        file_logger = FileCallbackHandler()
        response = await agent_executor.ainvoke(
            {"input": message.text},
            {"callbacks": [file_logger]}
        )
        final_answer = response.get('output', "Не удалось извлечь ответ.")

    except Exception as e:
        # ... (обработка ошибок остается прежней)
        logging.error(f"Ошибка при выполнении запроса агентом: {e}")
        error_text = str(e)
        if "Could not parse LLM output:" in error_text:
            logging.warning("Произошла ошибка парсинга, извлекаю ответ из текста ошибки...")
            raw_answer = error_text.split("Could not parse LLM output:")[1]
            final_answer = raw_answer.split("For troubleshooting, visit:")[0].strip()
        elif "quota" in error_text.lower() or "limit" in error_text.lower():
            final_answer = "К сожалению, я сейчас перегружен запросами (превышен лимит API). Пожалуйста, повторите попытку позже."
        else:
            final_answer = "К сожалению, при обработке вашего запроса произошла серьезная ошибка."

    await thinking_message.edit_text(final_answer)

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    if agent_executor is None:
        logging.critical("Запуск бота отменен, так как аналитический модуль не был инициализирован.")
    else:
        logging.info("Запуск Telegram-бота...")
        asyncio.run(main())

# --- END OF FILE main.py ---