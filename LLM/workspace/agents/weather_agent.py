import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "특정 도시의 현재 날씨를 조회합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"],
        }
    }
}

def create_weather_agent():
    return client.chat.completions
