# RecipeNER — Автоматическое распознавание именованных сущностей в рецептах блюд с использованием LSTM

## Цель проекта
Проект направлен на автоматизацию процесса анализа текстовых рецептов блюд путем выявления ключевых компонентов, таких как ингредиенты (`name`), их количество (`qty`), единицы измерения (`unit`) и дополнительные комментарии (`comment`).

## Описание проекта
### Данные
Использование датасета из **50,000** рецептов, представленных в формате **BIO** , где каждому слову соответствует тэг:

`B-<TAG>`: Начало сущности типа `<TAG>`.

`I-<TAG>`: Продолжение сущности типа `<TAG>`.

Данные разделены на обучающую выборку (**40,000** рецептов) и тестовую выборку (**10,000** рецептов).
### Архитектура модели
Модель состоит из нескольких компонентов:
- Эмбеддинговый слой (nn.Embedding) : Преобразует индексы слов в их векторные представления.
- LSTM-слой (nn.LSTM) : Обрабатывает последовательность эмбеддингов, учитывая контекст каждого слова.
- Полносвязный слой (nn.Linear) : Преобразует выход LSTM в пространство тегов.
- Выход модели проходит через функцию F.log_softmax, чтобы получить логарифмы вероятностей для каждого тега.
### Обучение
В качестве функции потерь используется **NLLLoss** (Negative Log-Likelihood Loss).

Для эффективного обновления параметров модели используется оптимизатор **Adam**.
### Оценка результатов

Для оценки качества модели были вычислены следующие метрики:
- Accuracy: 0.89
- Macro-Precision: 0.89
- Micro-Precision: 0.89
- Macro-Recall: 0.88
- Micro-Recall: 0.89
- Micro-F1: 0.88

А также была построена матрица ошибок (**Confusion Matrix**), показывающая, какие тэги чаще всего путаются.
