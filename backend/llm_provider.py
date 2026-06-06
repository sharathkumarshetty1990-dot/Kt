import json
import time
import urllib.request


class NimChatProvider:
    def __init__(self, api_url, model, api_key, timeout_seconds, max_attempts, system_prompt):
        self.api_url = api_url
        self.model = model
        self.api_key = (api_key or "").strip()
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, int(max_attempts or 1))
        self.system_prompt = system_prompt

    def metadata(self):
        return {
            "provider": "nvidia_nim",
            "configured": bool(self.api_url and self.api_key),
            "api_url": self.api_url,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_attempts": self.max_attempts,
        }

    def chat_json(self, prompt):
        if not self.api_url:
            raise RuntimeError("NIM_API_URL is not configured")
        if not self.api_key:
            raise RuntimeError("NIM_API_KEY is not configured")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        for attempt in range(self.max_attempts):
            request_obj = urllib.request.Request(
                self.api_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(request_obj, timeout=self.timeout_seconds) as response:
                    data = json.loads(response.read().decode("utf-8"))

                if "choices" in data:
                    return data["choices"][0]["message"]["content"]
                return data
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts - 1:
                    time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"NIM planning failed after {self.max_attempts} attempts: {last_error}")
