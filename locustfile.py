from locust import HttpUser, between, task


class WineApiUser(HttpUser):
    # Пауза между запросами каждого пользователя.
    wait_time = between(0.5, 2.0)

    @task(1)
    def predict_wine(self):
        payload = {
            "alcohol": 14.23,
            "malic_acid": 1.71,
            "ash": 2.43,
            "alcalinity_of_ash": 15.6,
            "magnesium": 127,
            "total_phenols": 2.8,
            "flavanoids": 3.06,
            "nonflavanoid_phenols": 0.28,
            "proanthocyanins": 2.29,
            "color_intensity": 5.64,
            "hue": 1.04,
            "od280_od315_of_diluted_wines": 3.92,
            "proline": 1065,
        }

        with self.client.post(
            "/v1/predict",
            json=payload,
            catch_response=True,
            timeout=10,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            else:
                result = response.json()
                if result.get("predicted_class") != 0:
                    response.failure("Expected class 0")


    @task(9)
    def predict_shifted_wine(self):
        payload = {
            "alcohol": 12.8,
            "malic_acid": 5.0,
            "ash": 2.5,
            "alcalinity_of_ash": 25.0,
            "magnesium": 100,
            "total_phenols": 0.6,
            "flavanoids": 0.1,
            "nonflavanoid_phenols": 0.7,
            "proanthocyanins": 0.3,
            "color_intensity": 14.0,
            "hue": 0.35,
            "od280_od315_of_diluted_wines": 1.0,
            "proline": 500,
        }

        with self.client.post(
            "/v1/predict",
            json=payload,
            name="/v1/predict shifted",
            catch_response=True,
            timeout=10,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            else:
                result = response.json()
                if result.get("predicted_class") not in (0, 1, 2):
                    response.failure("Invalid predicted class")