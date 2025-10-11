import json
import sqlite3
import logging
import re
from datetime import datetime

# --- Настройки ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
JSON_FILE = 'chat_history.json'
DB_FILE = 'qa.db'

def has_link(text_or_entities):
    """Проверяет, содержит ли текст или его сущности ссылку."""
    # Если на входе строка, ищем в ней
    if isinstance(text_or_entities, str):
        return bool(re.search(r'https?://\S+', text_or_entities))
    # Если список (сущности), ищем сущности типа 'link' или 'text_link'
    if isinstance(text_or_entities, list):
        for entity in text_or_entities:
            if isinstance(entity, dict) and entity.get('type') in ['link', 'text_link']:
                return True
    return False

def create_rich_db_and_table():
    """Создает или пересоздает базу данных с новой, богатой структурой."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS messages")
    
    # Новая, детализированная схема таблицы
    cursor.execute("""
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY,
            thread_id INTEGER,
            author_id TEXT,
            author_name TEXT,
            message_text TEXT,
            text_length INTEGER,
            timestamp TEXT,
            reply_to_message_id INTEGER,
            has_media BOOLEAN,
            has_links BOOLEAN
        )
    """)
    conn.commit()
    conn.close()
    logging.info(f"База данных '{DB_FILE}' с новой таблицей 'messages' создана.")

def get_full_text(text_data):
    """Извлекает полный текст из поля 'text', которое может быть строкой или списком."""
    if isinstance(text_data, str):
        return text_data
    if isinstance(text_data, list):
        full_text = ""
        for part in text_data:
            if isinstance(part, str):
                full_text += part
            elif isinstance(part, dict) and 'text' in part:
                full_text += str(part['text']) # Убедимся, что это строка
        return full_text
    return "" # Возвращаем пустую строку, если формат неизвестен или текст отсутствует

def main():
    """Основная функция для парсинга JSON и наполнения детализированной БД."""
    logging.info("Запуск процесса создания структурированной базы данных...")

    create_rich_db_and_table()

    logging.info(f"Загрузка данных из {JSON_FILE}...")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            # Используем потоковое чтение для большого файла
            chat_data = json.load(f)
        messages = chat_data.get('messages', [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Ошибка при чтении JSON-файла: {e}")
        return

    logging.info(f"Найдено {len(messages)} сообщений. Начинаю обработку...")
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    processed_count = 0
    for msg in messages:
        # Пропускаем сервисные сообщения, так как у них другая структура
        if msg.get('type') != 'message':
            continue

        full_text = get_full_text(msg.get('text', ''))
        
        # Определяем ID автора, обрабатывая случаи, когда его нет
        author_id = str(msg.get('from_id')) if msg.get('from_id') else "unknown"
        
        # Определяем thread_id
        thread_id = msg.get('reply_to_message_id') or msg.get('id')

        # Проверяем наличие медиа
        media_keys = ['photo', 'video', 'animation', 'sticker', 'file', 'voice_message', 'video_message']
        has_media_flag = any(key in msg for key in media_keys)

        # Проверяем наличие ссылок
        has_links_flag = has_link(msg.get('text_entities', [])) or has_link(full_text)

        cursor.execute(
            """
            INSERT INTO messages (
                message_id, thread_id, author_id, author_name, message_text, 
                text_length, timestamp, reply_to_message_id, has_media, has_links
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg.get('id'),
                thread_id,
                author_id,
                msg.get('from'),
                full_text,
                len(full_text),
                msg.get('date'),
                msg.get('reply_to_message_id'),
                has_media_flag,
                has_links_flag
            )
        )
        processed_count += 1

    conn.commit()
    conn.close()
    
    logging.info(f"Обработка завершена. В базу данных добавлено {processed_count} сообщений.")

if __name__ == '__main__':
    main()