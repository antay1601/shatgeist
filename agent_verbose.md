2025-10-19 10:05:22,429 - INFO - Инициализация компонентов...
2025-10-19 10:05:22,475 - INFO - Use pytorch device_name: mps
2025-10-19 10:05:22,475 - INFO - Load pretrained SentenceTransformer: paraphrase-multilingual-mpnet-base-v2
2025-10-19 10:05:29,187 - INFO - Все компоненты успешно инициализированы.
2025-10-19 10:05:29,193 - INFO - 'Двигатель' бота (Гибридный агент) успешно инициализирован.
2025-10-19 10:05:29,245 - INFO - Запуск Telegram-бота...
2025-10-19 10:05:29,246 - INFO - Start polling
2025-10-19 10:05:29,619 - INFO - Run polling for bot @Vasha2024_bot id=7510963021 - 'availability_test'
2025-10-19 10:05:41,469 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:41,472 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:41,476 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:41,477 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:41,480 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:41,482 - INFO - > Finished chain.
2025-10-19 10:05:41,483 - INFO - > Finished chain.
2025-10-19 10:05:41,484 - INFO - > Finished chain.
2025-10-19 10:05:41,486 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:41,487 - INFO - > Finished chain.
2025-10-19 10:05:42,595 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
2025-10-19 10:05:44,283 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:44,283 - WARNING - Error in DetailedFileCallbackHandler.on_chain_start callback: AttributeError("'AIMessageChunk' object has no attribute 'get'")
2025-10-19 10:05:44,285 - INFO - > Finished chain.
2025-10-19 10:05:44,286 - INFO - > Finished chain.
2025-10-19 10:05:44,287 - INFO - Thought: Для ответа на этот вопрос лучше всего подойдет инструмент smart_semantic_search_chat, так как нам нужно найти сообщения, в которых обсуждались места для курения кальяна. Это концептуальный вопрос, требующий поиска по смыслу, а не точных данных.

Action: smart_semantic_search_chat
Action Input: "рекомендации мест для курения кальяна в Валенсии"
2025-10-19 10:05:44,288 - INFO - Запуск умного семантического поиска для запроса: 'рекомендации мест для курения кальяна в Валенсии"
'
2025-10-19 10:05:44,289 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:44,291 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:44,292 - INFO - > Finished chain.
2025-10-19 10:05:46,131 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
2025-10-19 10:05:46,149 - INFO - > Finished chain.
2025-10-19 10:05:46,150 - INFO - Сгенерированы расширенные запросы: ['рекомендации мест для курения кальяна в Валенсии"\n', 'Вот 3 альтернативных поисковых запроса на русском языке:', 'Кальянные бары в Валенсии', 'Лучшие места покурить кальян в Валенсии', 'Популярные кальян-лаунжи Валенсии']
2025-10-19 10:05:47,175 - ERROR - Ошибка в run_smart_semantic_search: 
2025-10-19 10:05:47,176 - INFO - Observation: Ошибка при семантическом поиске: 
2025-10-19 10:05:47,178 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:47,180 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:47,182 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:47,185 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:47,187 - INFO - > Finished chain.
2025-10-19 10:05:47,188 - INFO - > Finished chain.
2025-10-19 10:05:47,189 - INFO - > Finished chain.
2025-10-19 10:05:47,190 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:47,191 - INFO - > Finished chain.
2025-10-19 10:05:47,697 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
2025-10-19 10:05:49,603 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:49,603 - WARNING - Error in DetailedFileCallbackHandler.on_chain_start callback: AttributeError("'AIMessageChunk' object has no attribute 'get'")
2025-10-19 10:05:49,604 - INFO - > Finished chain.
2025-10-19 10:05:49,605 - INFO - > Finished chain.
2025-10-19 10:05:49,606 - INFO - Извините, похоже, что произошла ошибка при попытке использовать инструмент семантического поиска. В таком случае, давайте попробуем использовать SQL-запрос для поиска сообщений, которые могут содержать информацию о местах для курения кальяна.

Action: sql_database_query
Action Input: SELECT author_name, message_text, timestamp FROM messages WHERE message_text LIKE '%кальян%' OR message_text LIKE '%курить%' OR message_text LIKE '%курение%' ORDER BY timestamp DESC LIMIT 3
2025-10-19 10:05:49,861 - INFO - Observation: [('Demetrio Mazur', 'У мене іспанська до 21.00 тому я одразу в кальян на 22', '2024-10-18T18:46:48'), ('Даниил Лесняков', 'На кальян пойдешь? Два месте в ледженте есть.', '2024-10-11T10:49:37'), ('Serhii Qwarz', 'Есть два места на покурить кальян в Legend, если есть желающие пишите в лс, обсудим ваш профиль', '2024-10-11T10:44:06')]
2025-10-19 10:05:49,862 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:49,863 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:49,863 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:49,864 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:49,865 - INFO - > Finished chain.
2025-10-19 10:05:49,866 - INFO - > Finished chain.
2025-10-19 10:05:49,867 - INFO - > Finished chain.
2025-10-19 10:05:49,867 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:49,868 - INFO - > Finished chain.
2025-10-19 10:05:50,641 - INFO - HTTP Request: POST https://api.anthropic.com/v1/messages "HTTP/1.1 200 OK"
2025-10-19 10:05:54,024 - INFO - > Entering new AgentExecutor chain...
2025-10-19 10:05:54,024 - WARNING - Error in DetailedFileCallbackHandler.on_chain_start callback: AttributeError("'AIMessageChunk' object has no attribute 'get'")
2025-10-19 10:05:54,025 - INFO - > Finished chain.
2025-10-19 10:05:54,026 - INFO - > Finished chain.
2025-10-19 10:05:54,027 - INFO - Thought: Теперь у нас есть несколько сообщений, упоминающих кальян. Я могу сформулировать ответ на основе этой информации.

Final Answer: 

**Сводка:** В Валенсии есть несколько мест, где можно покурить кальян. Одно из популярных мест - кальянная под названием "Legend". Люди часто договариваются о совместном посещении кальянных, иногда бронируя места заранее.

**Исходные сообщения:**
Serhii Qwarz (2024-10-11): Есть два места на покурить кальян в Legend, если есть желающие пишите в лс, обсудим ваш профиль
Даниил Лесняков (2024-10-11): На кальян пойдешь? Два месте в ледженте есть.
Demetrio Mazur (2024-10-18): У мене іспанська до 21.00 тому я одразу в кальян на 22
2025-10-19 10:05:54,028 - INFO - > Finished chain.
2025-10-19 10:05:54,147 - INFO - Update id=351851574 is handled. Duration 13089 ms by bot id=7510963021
2025-10-19 10:08:24,105 - WARNING - Received SIGINT signal
2025-10-19 10:08:24,106 - INFO - Polling stopped for bot @Vasha2024_bot id=7510963021 - 'availability_test'
2025-10-19 10:08:24,106 - INFO - Polling stopped
