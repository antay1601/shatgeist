# --- START OF FILE import_to_db.py ---

import json
import sqlite3
import logging
import re
from datetime import datetime
import os
import faiss
from sentence_transformers import SentenceTransformer
# --- ИЗМЕНЕНИЕ: Импортируем numpy ---
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Константы ---
JSON_FILE = 'chat_history.json'
DB_FILE = 'qa.db'
FAISS_INDEX_FILE = 'qa.index'
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

def get_full_text(text_data):
    if isinstance(text_data, str): return text_data
    if isinstance(text_data, list):
        return "".join(str(part.get('text', '')) if isinstance(part, dict) else str(part) for part in text_data)
    return ""

def main():
    logging.info("Запуск процесса создания структурированной БД и семантического индекса...")

    # Удаляем старые файлы для полной перестройки
    for f in [DB_FILE, FAISS_INDEX_FILE]:
        if os.path.exists(f):
            os.remove(f)
            logging.info(f"Удален старый файл: {f}")

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS messages")
    cursor.execute("""
        CREATE TABLE messages (
            message_id INTEGER PRIMARY KEY,
            author_name TEXT,
            message_text TEXT,
            timestamp TEXT
        )
    """)

    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        all_messages = chat_data.get('messages', [])
    except Exception as e:
        logging.error(f"Ошибка при чтении JSON-файла: {e}")
        return

    logging.info(f"Найдено {len(all_messages)} сообщений. Начинаю обработку...")
    
    messages_for_indexing = []
    for msg in all_messages:
        if msg.get('type') != 'message': continue
        
        full_text = get_full_text(msg.get('text', ''))
        if not full_text: continue

        author_name = msg.get('from') or f"user_{msg.get('from_id')}"
        
        cursor.execute(
            "INSERT INTO messages (message_id, author_name, message_text, timestamp) VALUES (?, ?, ?, ?)",
            (msg['id'], author_name, full_text, msg['date'])
        )
        messages_for_indexing.append({'id': msg['id'], 'text': full_text})

    conn.commit()
    conn.close()
    logging.info(f"Структурированная база данных '{DB_FILE}' успешно создана.")

    if not messages_for_indexing:
        logging.warning("Нет сообщений для создания семантического индекса.")
        return

    logging.info(f"Загрузка модели '{MODEL_NAME}' для векторизации...")
    model = SentenceTransformer(MODEL_NAME)
    
    texts_to_encode = [msg['text'] for msg in messages_for_indexing]
    message_ids_list = [msg['id'] for msg in messages_for_indexing] # Переименуем для ясности

    logging.info(f"Создание векторов для {len(texts_to_encode)} сообщений...")
    embeddings = model.encode(texts_to_encode, show_progress_bar=True, convert_to_numpy=True)
    
    # --- ИЗМЕНЕНИЕ: Преобразуем список ID в NumPy массив ---
    message_ids_np = np.array(message_ids_list, dtype='int64')

    index = faiss.IndexIDMap(faiss.IndexFlatL2(embeddings.shape[1]))
    # --- ИЗМЕНЕНИЕ: Передаем в Faiss NumPy массив ---
    index.add_with_ids(embeddings, message_ids_np)
    
    faiss.write_index(index, FAISS_INDEX_FILE)
    logging.info(f"Семантический индекс '{FAISS_INDEX_FILE}' с {index.ntotal} векторами успешно создан.")
    logging.info("Процесс успешно завершен.")

if __name__ == '__main__':
    main()

# --- END OF FILE import_to_db.py ---