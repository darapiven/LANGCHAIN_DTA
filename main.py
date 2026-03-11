import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

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
with open("data/faq.txt", "r", encoding="utf-8") as f:
    faq_text = f.read()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# docs = splitter.split_text(faq_text)
docs = splitter.create_documents([faq_text])
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)
@tool
def search_faq(query: str) -> str:
    """Шукай відповід у внутрішньому файлі faq магазину. Питанні і відповіді мають бути українською"""
    try:
        # result = vector_store.search(query)
        result = vector_store.similarity_search(query, k=1)
        if not result:
            return "Нічого не знайдено у FAQ"
        return result[0].page_content
    except Exception as e:
        return f"Error search: {e}"
tools = [calculator, weather_api, search_faq]
# __ agent _________________________
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="Ти особистий помічник. Відповідай ввічливо та дружньо. Для відповідей користуйся інструментами tools, якщо потрібно",
    debug=False #False щоб отримати питання відповідь
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
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", "")
        if content and not tool_calls:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content)
    return ""

chat_messages = []
def chat(user_input: str) -> str:
    chat_messages.append({"role": "user", "content": user_input})
    result = agent.invoke({"messages":chat_messages})

    chat_messages.clear()                       # chat_messages = []
    chat_messages.extend(result["messages"])    # [] + [{"role": "user", "content": user_input}, {"role": "assistant", "content": ..., "tool_calls": ...}] = [{"role": "user", "content": user_input}, {"role": "assistant", "content": ..., "tool_calls": ...}]

    return get_output(result)
def run_interactive():
    while True:
        user = input("User: ")
        if user.lower() in ["q", "exit", "x", "^Z", "й"]:
            print("До побачення")
            break 
        print("Assistant: ", chat(user))

if __name__ == "__main__":
    run_interactive()