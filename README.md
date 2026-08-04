# Credit Score System

Пайплайн для предсказания кредитного рейтинга клиента (**Good**, **Standard** или **Poor**) по его финансовым и поведенческим данным. Датасет довольно "грязный", поэтому большая часть работы — это очистка и восстановление пропусков. Дальше обучаются и сравниваются две модели: Random Forest и CatBoost, с подбором гиперпараметров через Optuna.

## Что делает пайплайн

1. Чистит данные и восстанавливает пропуски по `Customer_ID` (у одного клиента может быть несколько записей — пропуски в одной можно закрыть значениями из других).
2. Подбирает гиперпараметры для Random Forest и CatBoost через Optuna.
3. Обучает обе модели с 5-фолдовой стратифицированной кросс-валидацией и считает F1-macro.
4. Сравнивает модели статистически — t-тестом, чтобы понять, значима ли разница между ними или это шум.
5. Строит графики сравнения (по фолдам, распределение метрик, доверительные интервалы, t-распределение).

## Датасет

Нужен файл `train.csv` в директории проекта. Основные столбцы, которые использует пайплайн:

- `Customer_ID`, `Age`, `Annual_Income`, `Monthly_Inhand_Salary`
- `Num_Bank_Accounts`, `Num_Credit_Card`, `Interest_Rate`, `Num_of_Loan`
- `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Changed_Credit_Limit`
- `Num_Credit_Inquiries`, `Outstanding_Debt`, `Credit_Utilization_Ratio`
- `Total_EMI_per_month`, `Amount_invested_monthly`, `Monthly_Balance`
- `Month`, `Credit_History_Age`
- `Credit_Score` — целевая переменная (`Good` / `Standard` / `Poor`)

Датасет - [Credit Score Classification с Kaggle](https://www.kaggle.com/datasets/parisrohan/credit-score-classification).

## Этапы

### 1. Очистка данных (`clean_dataset`)

- Оставляет только строки с валидным `Credit_Score` (`Good`, `Standard`, `Poor`).
- Чистит числовые столбцы от мусора (лишние символы) и приводит к числам.
- Месяцы (`January`, `February`...) переводятся в числа.
- `Credit_History_Age` вида "5 Years and 3 Months" парсится в общее число месяцев.
- Пропуски заполняются по каждому клиенту (`Customer_ID`) через forward/backward fill — логично, что у одного человека возраст, доход и т.п. не должны прыгать между записями.

### 2. Подбор гиперпараметров

- **Random Forest** — тюнится на подвыборке (для скорости), перебираются `n_estimators`, `max_depth`, `min_samples_split`. Стоит `class_weight='balanced'`, т.к. классы несбалансированы. Категориальные признаки кодируются One-Hot.
- **CatBoost** — обучается на GPU, перебираются `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`. Использует свою нативную обработку категориальных признаков плюс `RandomOverSampler` для баланса классов.
- Оба тюнятся через [Optuna](https://optuna.org/), по 10 попыток на модель, метрика — F1-macro.

### 3. Кросс-валидация

Лучшие найденные параметры используются, чтобы переобучить обе модели уже на полном датасете, с 5 фолдами и стратификацией. Метрика — F1-macro по каждому фолду.

### 4. Статистическая значимость

t-тест (`scipy.stats.ttest_ind`) между фолд-скорами двух моделей — проверяем, действительно ли разница в качестве значима (p < 0.05), а не просто случайность.

### 5. ROC-AUC, Gini и бизнес-эффект

Помимо F1-macro, задача сводится к бинарной (`Poor` = дефолт, остальные — нет) и считается ROC-AUC/Gini на той же кросс-валидации — это более стандартная для скоринга метрика, так как не зависит от порога отсечения. Дальше на out-of-fold предсказаниях CatBoost проводится симуляция: что будет с NPL (долей дефолтов), если отсекать худший дециль заёмщиков по PD-скору, и сколько это гипотетически могло бы сэкономить на резервах (при заданных допущениях по среднему чеку, LGD и размеру портфеля).

### 6. Графики

Дашборд 2×2:
- F1-macro по фолдам (линии)
- Boxplot + точки по каждому фолду
- Средние значения с 95% доверительным интервалом
- t-распределение с отмеченной t-статистикой

## Результаты

| Модель | F1-macro (CV) | Параметры |
|---|---|---|
| Random Forest | 0.6495 (± 0.0036) | `n_estimators=79, max_depth=19, min_samples_split=3` |
| CatBoost | 0.7560 (± 0.0029) | `iterations=358, learning_rate=0.0899, depth=8, l2_leaf_reg=0.0575` |

t-statistic: -45.9227, p-value: 0.0000 — разница статистически значима, CatBoost значительно превосходит Random Forest по качеству.

**ROC-AUC / Gini** (бинарная постановка: `Poor` vs остальные, та же 5-фолдовая CV):

| Модель | ROC-AUC (CV) | Gini |
|---|---|---|
| Random Forest | 0.8624 (± 0.0064)  |  0.7248 |
| CatBoost | 0.9248 (± 0.0045)  | 0.8497 |

t-statistic: -15.9355, p-value: 0.0000 - разница статистически значима, CatBoost значительно превосходит Random Forest по качеству.


**Гипотетический бизнес-эффект.** На предсказаниях CatBoost заёмщики разбиваются на 10 децилей по PD-скору, и считается, каким был бы NPL портфеля, если отсечь худший дециль (10% самых рискованных). Дальше это переводится в деньги через Expected Loss (`NPL × средний кредит × LGD × размер портфеля`) при следующих допущениях:

- средний размер кредита — 210 000 руб.;
- LGD (Loss Given Default) — 60%;
- гипотетический портфель — 50 000 заёмщиков.

Базовый NPL по всему портфелю: 29.08%
NPL после отсечения худшего дециля (10% самых рискованных заёмщиков): 23.12%
Относительное снижение NPL: 20.5%

Ожидаемые потери (Expected Loss) до модели: 1841.1 млн руб/год
Ожидаемые потери после отсечения худшего дециля: 1463.2 млн руб/год
Потенциальная экономия резервов: 377.9 млн руб/год

##  🐋 Архитектура API и Docker

Приложение обёрнуто в FastAPI и поставляется в виде Docker-контейнера.

### Основные эндпоинты:
* `GET /health` — проверка статуса сервиса и загрузки модели.
* `POST /predict` — расчет вероятности дефолта (`pd_score`), итогового класса и флага высокого риска (`high_risk_decile`).
* `POST /reload-model` — горячая перезагрузка весов `.cbm` без перезапуска Docker-контейнера.

---

##  Быстрый запуск

### Запуск через Docker 

**Сборка образa:**
   ```bash
   docker build -t credit-score-api .

   docker run -d \
   --name credit_api \
   -p 8000:8000 \
   -v $(pwd)/model:/app/model \
   credit-score-api
  ```
## Проверка работы через curl:

curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
    "Month": "March",
    "Age": 32,
    "Occupation": "Scientist",
    "Annual_Income": 45000,
    "Monthly_Inhand_Salary": 3200,
    "Num_Bank_Accounts": 3,
    "Num_Credit_Card": 4,
    "Interest_Rate": 14,
    "Num_of_Loan": 2,
    "Type_of_Loan": "Auto Loan, Personal Loan",
    "Delay_from_due_date": 8,
    "Num_of_Delayed_Payment": 3,
    "Changed_Credit_Limit": 5.5,
    "Num_Credit_Inquiries": 2,
    "Credit_Mix": "Good",
    "Outstanding_Debt": 1200.0,
    "Credit_Utilization_Ratio": 32.1,
    "Credit_History_Age": "5 Years and 3 Months",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 150.0,
    "Amount_invested_monthly": 100.0,
    "Payment_Behaviour": "High_spent_Small_value_payments",
    "Monthly_Balance": 250.0
  }'



## Установка

```bash
pip install -r requirements.txt
```

Для CatBoost нужен GPU (в коде стоит `task_type='GPU'`) — если GPU нет, поменяйте на `'CPU'` в параметрах модели.

## Запуск

1. Положите `train.csv` рядом с ноутбуком.
2. Откройте `CreditScoring.ipynb` и прогоните ячейки по порядку (Jupyter или Google Colab с GPU).
3. Смотрите на метрики по фолдам, результат t-теста и графики в конце.

## Структура

```
.
├── CreditScoring.ipynb
├── requirements.txt
├── data
├    ├──train.csv   
├    └──test.csv 
└── README.md
```

## Немного нюансов

- Метрика везде F1-macro — она не даёт "большим" классам перевешивать редкие.
- Баланс классов: у RandomForest через `class_weight`, у CatBoost — через oversampling.
- Подбор гиперпараметров идёт на подвыборке ради скорости, финальное обучение — уже на всех данных.
