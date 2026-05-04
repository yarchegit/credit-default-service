
---

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение моделей

```bash
python src/train.py
```

Или используйте notebook: `notebooks/01_train_models.ipynb`

**Результаты обучения:**
- Model v1: F1-score = 0.4679
- Model v2: F1-score = 0.4638
- Разница: -0.88%

### 3. Запуск API

```bash
python app/api.py
```

API будет доступен на `http://localhost:5001`

---

## 🌐 API Endpoints

### Health Check
```bash
curl http://localhost:5001/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "models": {
    "v1_loaded": true,
    "v2_loaded": true
  },
  "ab_test_ratio": 0.5
}
```

### Предсказание (A/B тест)
```bash
curl -X POST http://localhost:5001/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "features": {
      "LIMIT_BAL": 20000,
      "SEX": 2,
      "EDUCATION": 2,
      "MARRIAGE": 1,
      "AGE": 24,
      "PAY_0": 2,
      "PAY_2": 2,
      "PAY_3": -1,
      "PAY_4": -1,
      "PAY_5": -2,
      "PAY_6": -2,
      "BILL_AMT1": 3913,
      "BILL_AMT2": 3102,
      "BILL_AMT3": 689,
      "BILL_AMT4": 0,
      "BILL_AMT5": 0,
      "BILL_AMT6": 0,
      "PAY_AMT1": 0,
      "PAY_AMT2": 689,
      "PAY_AMT3": 0,
      "PAY_AMT4": 0,
      "PAY_AMT5": 0,
      "PAY_AMT6": 0
    }
  }'
```

**Ответ:**
```json
{
  "prediction": 1,
  "prediction_label": "DEFAULT",
  "probability_default": 0.6234,
  "probability_no_default": 0.3766,
  "risk_level": "MEDIUM",
  "model_version": "v1",
  "request_id": "20260504224512-3421",
  "timestamp": "2026-05-04T22:45:12.123456"
}
```

### Прямой вызов Model v1
```bash
curl -X POST http://localhost:5001/predict/v1 \\
  -H "Content-Type: application/json" \\
  -d '{"features": {...}}'
```

### Прямой вызов Model v2
```bash
curl -X POST http://localhost:5001/predict/v2 \\
  -H "Content-Type: application/json" \\
  -d '{"features": {...}}'
```

---

## 📊 Описание признаков

| Признак | Описание |
|---------|----------|
| LIMIT_BAL | Кредитный лимит (NT dollar) |
| SEX | Пол (1=male, 2=female) |
| EDUCATION | Образование (1=graduate, 2=university, 3=high school, 4=others) |
| MARRIAGE | Семейное положение (1=married, 2=single, 3=others) |
| AGE | Возраст (годы) |
| PAY_0 to PAY_6 | История платежей за последние 6 месяцев |
| BILL_AMT1 to BILL_AMT6 | Сумма счета за последние 6 месяцев |
| PAY_AMT1 to PAY_AMT6 | Сумма оплаты за последние 6 месяцев |

---


### Интерактивное тестирование
Используйте notebook: `notebooks/02_test_api.ipynb`

---

## 📈 A/B тестирование

Endpoint `/predict` автоматически распределяет 50% запросов на модель v1 и 50% на модель v2.

**Ключевые метрики:**
- F1-score
- Precision / Recall
- Expected Loss Reduction
- Latency (p50, p95, p99)

Подробности: [AB_TEST_PLAN.md](AB_TEST_PLAN.md)

---

## 📝 Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| MODEL_V1_PATH | Путь к модели v1 | models/model_v1.pkl |
| MODEL_V2_PATH | Путь к модели v2 | models/model_v2.pkl |
| AB_TEST_RATIO | Доля трафика на v2 | 0.5 |

---

## 🤝 Контакты

- GitHub: [@yarchegit](https://github.com/yarchegit)
- Email: yarche@example.com

---

## 📄 Лицензия

MIT License
