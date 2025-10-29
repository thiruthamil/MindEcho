# app/models.py — Pydantic data schemas

from pydantic import BaseModel, Field
from typing import List


class EntryIn(BaseModel):
    text: str = Field(..., min_length=3, max_length=4000)


class Snippet(BaseModel):
    id: str
    date: str
    preview: str
    score: float


class EntryOut(BaseModel):
    reply: str
    similar: List[Snippet]


class WeeklySummary(BaseModel):
    bullets: List[str]


class SearchOut(BaseModel):
    results: List[Snippet]
