# --- START OF FILE main.py ---

import asyncio
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import sqlite3
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

from aiogram import Bot, Dispatcher, types
from aiogram.filters.command import Command

from langchain_community.utilities import SQLDatabase
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_community.tools import Tool
from langchain.tools.render import render_text_description

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

load_dotenv()

# --- Константы ---
DB_FILE = 'qa.db'
FAISS_INDEX_FILE = 'qa.index'
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DETAILED_LOG_FILE = "llm_requests.log"
VERBOSE_LOG_FILE = "agent_verbose.md"
MODEL_NAME_SEMANTIC = 'paraphrase-multilingual-mpnet-base-v2'

# --- 1. Ручная настройка логирования ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

file_handler = logging.FileHandler(VERBOSE_LOG_FILE, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# --- 2. Инициализация ---
try:
    logging.info("Инициализация компонентов...")
    db = SQLDatabase.from_uri(f"sqlite:///{DB_FILE}")
    llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
    semantic_model = SentenceTransformer(MODEL_NAME_SEMANTIC)
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    logging.info("Все компоненты успешно инициализированы.")
except Exception as e:
    logging.critical(f"Критическая ошибка при инициализации: {e}")
    db = llm = semantic_model = faiss_index = None

# --- 3. Инструменты ---
async def run_smart_semantic_search(query: str) -> str:
    try:
        k = 5
        logging.info(f"Запуск умного семантического поиска для запроса: '{query}'")
        expansion_prompt = PromptTemplate.from_template(
            "Пользователь ищет в истории чата информацию по теме: '{query}'. "
            "Сгенеририруй 3 альтернативных поисковых запроса на русском языке. Выведи каждый с новой строки."
        )
        expansion_chain = expansion_prompt | llm
        response = await expansion_chain.ainvoke({"query": query})
        expanded_queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        search_queries = [query] + expanded_queries
        logging.info(f"Сгенерированы расширенные запросы: {search_queries}")

        query_vectors = semantic_model.encode(search_queries)
        
        # --- ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ: Явно преобразуем векторы в тип float32 ---
        all_distances, all_ids = faiss_index.search(query_vectors.astype(np.float32), k)
        
        unique_ids = list(dict.fromkeys(all_ids.flatten()))
        unique_ids = [int(id_val) for id_val in unique_ids if id_val != -1]
        if not unique_ids:
            return "Семантический поиск не нашел релевантных сообщений."

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        placeholders = ','.join('?' for _ in unique_ids)
        cursor.execute(f"SELECT author_name, timestamp, message_text FROM messages WHERE message_id IN ({placeholders})", tuple(unique_ids))
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "Семантический поиск не нашел релевантных сообщений."
        
        return "\n---\n".join([f"Автор: {row[0]} ({row[1]})\nСообщение: {row[2]}" for row in results])
    except Exception as e:
        logging.error(f"Ошибка в run_smart_semantic_search: {e}")
        return f"Ошибка при семантическом поиске: {e}"

tools = [
    Tool(
        name="smart_semantic_search_chat",
        func=run_smart_semantic_search,
        description="""Используй для открытых, концептуальных вопросов (рекомендации, обсуждения), когда нужно найти сообщения по смыслу. Например: "посоветуйте стоматолога", "обсуждения ресторанов", "что говорили про ремонт". """,
        coroutine=run_smart_semantic_search
    ),
    Tool(
        name="sql_database_query",
        func=db.run,
        description="""Используй для выполнения SQL-запросов. Полезен для точных вопросов (подсчет, поиск по автору/дате). Входные данные - корректный SQLite запрос. Ты можешь использовать этот инструмент для проверки схемы БД."""
    )
]

# --- 4. Логгеры и Промпт ---
class DetailedFileCallbackHandler(BaseCallbackHandler):
    def __init__(self, filename: str = DETAILED_LOG_FILE):
        self.file = open(filename, 'a', encoding='utf-8')
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        self.file.write(f"\n\n---\n\n## 🚀 Новая сессия: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        self.file.write(f"**Входной запрос пользователя:**\n```\n{inputs.get('input', 'N/A')}\n```\n")
        self.file.flush()
    def on_llm_start(self, serialized: dict, prompts: list[str], **kwargs) -> None:
        self.file.write(f"\n### ➡️ Запрос к LLM (Раунд размышлений)\n\n")
        self.file.write(f"**Полный промпт, отправленный модели:**\n```text\n{prompts}\n```\n")
        self.file.flush()
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        self.file.write(f"\n### 🛠️ Выбранное Действие\n\n**Инструмент:** `{action.tool}`\n**Входные данные:**\n```\n{action.tool_input}\n```\n")
        self.file.flush()
    def on_tool_end(self, output: str, **kwargs) -> None:
        self.file.write(f"\n### 📊 Результат Инструмента\n\n```\n{output}\n```\n")
        self.file.flush()
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        self.file.write(f"\n### ✅ Финальный Ответ\n\n```\n{finish.return_values.get('output')}\n```\n")
        self.file.flush()
    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        self.file.write(f"\n--- 🏁 Сессия завершена ---\n")
        self.file.flush()

class VerboseFileCallbackHandler(BaseCallbackHandler):
    """Callback-обработчик, который пишет 'мысли' агента через стандартный модуль logging."""
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs) -> None:
        logging.info("> Entering new AgentExecutor chain...")
    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        logging.info(action.log.strip())
    def on_tool_end(self, output: str, **kwargs) -> None:
        logging.info(f"Observation: {output}")
    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        logging.info(finish.log.strip())
    def on_chain_end(self, outputs: dict, **kwargs) -> None:
        logging.info("> Finished chain.")

CUSTOM_PROMPT_TEMPLATE = """
Твоя задача - отвечать на вопросы пользователя на РУССКОМ ЯЗЫКЕ, анализируя историю чата.
Сначала проанализируй вопрос и выбери наиболее подходящий инструмент.
- Для концептуальных вопросов (рекомендации, обсуждения) используй `smart_semantic_search_chat`.
- Для точных вопросов (статистика, поиск по автору/дате) используй `sql_database_query`.
Перед использованием `sql_database_query`, ты МОЖЕШЬ проверить схему таблицы, если не уверен в именах колонок.

**Схема базы данных:**
{db_schema}

У тебя есть доступ к следующим инструментам:
{tools}

**ВАЖНОЕ ПРАВИЛО ФОРМАТИРОВАНИЯ ОТВЕТА:**
Твой финальный ответ должен состоять из двух частей:
1.  **Сводка:** Краткий, лаконичный вывод на русском языке.
2.  **Источники:** Под загоголовком "**Исходные сообщения:**" приведи 2-3 наиболее релевантных сообщения из базы данных, на основе которых сделана сводка. Каждое сообщение должно быть отформатировано так: `Автор (Дата): Текст сообщения`.

Используй СТРОГО следующий формат для своих мыслей:
Question: the input question you must answer
Thought: твои рассуждения на русском языке о том, какой инструмент выбрать и почему.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (Эта последовательность может повторяться)
Thought: Теперь я знаю финальный ответ и сформулирую его на русском языке, следуя правилу форматирования.
Final Answer: финальный ответ, состоящий из сводки и источников.

Начинай!

Question: {input}
Thought:{agent_scratchpad}
"""

try:
    rendered_tools = render_text_description(tools)
    tool_names = ", ".join([t.name for t in tools])
    db_schema = db.get_table_info()
    prompt = PromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE).partial(
        tools=rendered_tools,
        tool_names=tool_names,
        db_schema=db_schema
    )
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)
    logging.info("'Двигатель' бота (Гибридный агент) успешно инициализирован.")
except Exception as e:
    logging.critical(f"Не удалось инициализировать гибридного агента! Ошибка: {e}")
    agent_executor = None

# --- 5. Логика Telegram-бота ---
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.answer("Привет! Я — ChatGeist, гибридный аналитик истории этого чата. Задайте мне вопрос.")

@dp.message()
async def handle_message(message: types.Message):
    if not agent_executor:
        await message.answer("Извините, аналитический модуль не запущен.")
        return
    if not message.text: return

    thinking_message = await message.answer("Анализирую ваш запрос...")
    try:
        detailed_logger = DetailedFileCallbackHandler()
        verbose_logger = VerboseFileCallbackHandler()
        response = await agent_executor.ainvoke(
            {"input": message.text},
            {"callbacks": [detailed_logger, verbose_logger]}
        )
        final_answer = response.get('output', "Не удалось извлечь ответ.")
    except Exception as e:
        logging.error(f"Ошибка при выполнении запроса агентом: {e}")
        final_answer = "К сожалению, при обработке вашего запроса произошла ошибка."

    await thinking_message.edit_text(final_answer)

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    if all([db, llm, semantic_model, faiss_index, agent_executor]):
        logging.info("Запуск Telegram-бота...")
        asyncio.run(main())
    else:
        logging.critical("Запуск бота отменен.")