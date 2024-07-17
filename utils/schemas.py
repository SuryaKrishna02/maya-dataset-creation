from dataclasses import dataclass

@dataclass
class Prompt:
    preamble: str
    message: str