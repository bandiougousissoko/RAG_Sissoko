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
langchain>=0.1.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pypdf>=3.0.0
tiktoken>=0.5.0


## Инструкция по запуску

### Требования
- Python 3.9 или выше
- Git

### Установка

1. **Клонировать репозиторий**
   ```bash
   git clone https://github.com/bandiougousissoko/RAG_Sissoko.git
   cd RAG_Sissoko/rag-pdf-project



## Как сохранить

1. В VS Code откройте файл `README.md`
2. Удалите всё старое содержимое
3. Вставьте новый текст выше
4. Нажмите `Ctrl+S`

Готово! Теперь у вас красивый, профессиональный README для вашего проекта.


