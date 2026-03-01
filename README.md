# RAG Sissoko - Проект RAG системы для работы с PDF

Этот проект представляет собой реализацию RAG (Retrieval-Augmented Generation) системы для обработки PDF-документов. Основная цель — извлечение, разбиение на чанки и подготовка текста для дальнейшего использования в генеративных моделях.

## Структура проекта

```
rag-pdf-project/
├── src/
│   ├── __init__.py
│   ├── text_splitter.py      # Разбиение текста на чанки
│   ├── models.py             # Модели данных (Document, Chunk)
│   ├── config.py             # Конфигурация
│   └── utils.py              # Вспомогательные функции
├── data/                      # PDF файлы для обработки
├── output/                    # Результаты (chunks.json, answer.json)
├── requirements.txt           # Зависимости
└── README.md                  # Описание проекта
```


## Зависимости

Основные библиотеки, необходимые для работы проекта:

```txt
langchain
pydantic
python-dotenv
