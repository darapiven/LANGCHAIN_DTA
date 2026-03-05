import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

# ========== tools ===========================

# простий калькулятор
# @tool
# def calculator(expression: str) -> str:
#     """Evaluate a math expression."""
#     return str(eval(expression))

@tool
def calculator(expression: str) -> str:
    """Обчислює арифметичні вирази: + - * / та дужки."""
    allowed_chars = "0123456789+-*/(). "
    if not all(ch in allowed_chars for ch in expression):
        return "Помилка: дозволені лише цифри та + - * / ( ) . і пробіли."
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Помилка обчислення: {e}"
    
# _____________________

@tool
def weather_api (city: str) -> str:
    """Повертає погоду для заданого міста. Назва міста задано англійською"""
    data =  {
        "Kyiv": "Сонячно, 25°C",
        "Lviv": "Хмарно, 20°C",
        "Odesa": "Дощ, 22°C",
        "Kharkiv": "Вітер, 18°C"}
    return data.get(
        city,
        f"Немає данних для введеного міста: {city}. Спробуйте: {", ".join(list(data.keys()))}"
    )
tools = [calculator, weather_api]
# __ agent _________________________
agent = create_agent(
    model=llm,
    tools=tools,
    system_promt="Ти особистий помічник. Відповідай ввічливо та дружньо. Для відповідей користуйся інструментами tools, якщо потрібно",
    debug=True #False щоб отримати питання відповідь
)

# {
#     "messages": [
#         {
#             "role": "user",
#             "content": "Яка погода в Києві та скільки буде 2 + 2?"
#         },
#         {
#             "role": "assistant",
#             "content": "Погода в Києві: Сонячно, 25°C. Результат обчислення 2 + 2: 4.",
#             "tool_calls": []
#         },
#        {
#             "role": "user",
#             "content": [
#                 {
#                    "text": "Яка погода в Києві?",
#                    "tool_name": "weather_api",   
#                    "tool_args": {"city": "Kyiv"}
#                  },
#           ]
#     ]
# }
def get_output(result:dict) -> str:
    messages = result.get("messages", [])
    for msg in messages:
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", "")
        if content and not tool_calls:
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                return "".join(c.get("text", str(c)) if isinstance(c,dict) else str(c) for c in content)
    return ""