from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from .structure_model import Conversation


def base_chain(api_key: str):
    system_prompt = """content의 내용을 소재로 user_1과 user_2의 대화 생성"""
    user_prompt = "content:{content}\nuser_1:{user_1}\nuser_2:{user_2}\n"
    prompt_template = ChatPromptTemplate(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template(user_prompt),
        ]
    )
    model_name = "gemini-1.5-flash-002"

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        api_key=api_key,
        temperature=0,
        max_tokens=2000,
        streaming=True,
        async_client_running=True,
    ).with_structured_output(convert_to_openai_function(Conversation))
    chain = prompt_template | llm
    return chain
