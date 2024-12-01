from pydantic import SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function

from app.external_services import structured_output



def base_chain(model: str, api_key: SecretStr, custom_prompt: list[tuple[str, str]], llm_params: dict):
    prompt = ChatPromptTemplate.from_messages(custom_prompt)
    llm = ChatGoogleGenerativeAI(
        model=model,
        api_key=api_key,
        **llm_params,
        async_client_running=True,
    ).with_structured_output(convert_to_openai_function(structured_output.Conversation))
    chain = prompt | llm
    return chain
