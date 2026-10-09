from dataclasses import dataclass


@dataclass
class Chunk:
    author: str
    source_file: str
    title: str
    url: str
    date: str
    text: str
    chunk_index: int
    file_hash: str = ""
