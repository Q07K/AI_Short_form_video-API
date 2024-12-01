from pydantic import BaseModel, Field


class Segment(BaseModel):
    segment_number: int = Field(
        description="base number divided by segment length",
    )
    stable_diffusion_prompt: str = Field(
        description="Create a prompt that describes your conversation in a format similar to Stable Diffusion Prompt Example."
    )
    user_1: str = Field(description="user_1's conversation")
    user_2: str = Field(description="user_2's conversation")


class Conversation(BaseModel):
    segments: list[Segment] = Field(description="5회 이상 대화를 나눈 내용")
