from time import perf_counter

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
    timeout=180.0,
    max_retries=0,
)


def run_benchmark(prompt):
    first_text_time = None
    usage = None

    start = perf_counter()

    with client.chat.completions.create(
        model="mlops-bot",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=128,
        stream=True,
        stream_options={"include_usage": True},
    ) as stream:
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage

            if chunk.choices:
                text = chunk.choices[0].delta.content
                if text and first_text_time is None:
                    first_text_time = perf_counter()

    end = perf_counter()

    if first_text_time is None or usage is None:
        raise RuntimeError("Нет текста или статистики токенов от сервера.")

    ttft = first_text_time - start
    total = end - start
    generation_time = end - first_text_time

    # Приближение: первый текстовый пакет может содержать несколько токенов.
    speed = (
        (usage.completion_tokens - 1) / generation_time
        if generation_time > 0 else 0
    )

    print(f"Входных токенов: {usage.prompt_tokens}")
    print(f"Выходных токенов: {usage.completion_tokens}")
    print(f"TTFT: {ttft:.4f} сек.")
    print(f"Общее время: {total:.4f} сек.")
    print(f"Скорость после первого текста, приблизительно: {speed:.2f} токенов/сек.")

    return ttft, speed, total


if __name__ == "__main__":
    print("Прогрев модели — этот результат не включаем в сравнение.")
    run_benchmark("Ответь одним словом: готов?")
    print()

    instruction = (
        "Назови три преимущества MLOps кратким списком. "
        "Текст ниже используй только как справочный материал.\n"
    )

    reference = (
        "MLOps объединяет версионирование данных, тестирование, "
        "развертывание моделей и мониторинг их работы. "
    )

    scenarios = [
        ("Короткий контекст", instruction),
        ("Средний контекст", instruction + reference * 5),
        ("Длинный контекст", instruction + reference * 50),
    ]

    results = []

    for name, prompt in scenarios:
        print("=" * 60)
        print(name)
        print(f"Символов в запросе: {len(prompt)}")

        ttft, speed, total = run_benchmark(prompt)
        results.append((name, ttft, speed, total))

    print("\nИТОГОВАЯ ТАБЛИЦА")
    print("Сценарий | TTFT, с | Скорость ~токенов/с | Всего, с")

    for name, ttft, speed, total in results:
        print(f"{name} | {ttft:.4f} | {speed:.2f} | {total:.4f}")