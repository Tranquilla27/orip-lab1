from time import perf_counter

import requests

API_URL = "http://127.0.0.1:8000/v1/predict"

payload = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127.0,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315_of_diluted_wines": 3.92,
    "proline": 1065.0,
}


def send_request(number):
    start = perf_counter()

    response = requests.post(
        API_URL,
        json=payload,
        timeout=10,
    )

    elapsed_ms = (perf_counter() - start) * 1000
    response.raise_for_status()
    result = response.json()

    print(
        f"Запрос {number}: HTTP {response.status_code} | "
        f"{elapsed_ms:.2f} мс | "
        f"Класс: {result['predicted_class']} | "
        f"Модель: {result['model_status']}"
    )

    return elapsed_ms


if __name__ == "__main__":
    print("Пять последовательных запросов к Wine API")

    try:
        times = [send_request(i) for i in range(1, 6)]
    except requests.RequestException as error:
        print(f"Ошибка обращения к API: {error}")
        raise SystemExit(1)

    print(f"Среднее время запроса: {sum(times) / len(times):.2f} мс")