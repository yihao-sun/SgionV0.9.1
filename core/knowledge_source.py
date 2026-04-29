"""
知识源抽象基类
定义统一的知识获取接口
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional
from dataclasses import dataclass, field
import time


@dataclass
class KnowledgeChunk:
    """知识块"""
    content: str
    source_type: str  # 'file', 'url'
    source_identifier: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class KnowledgeSource(ABC):
    def __init__(self, identifier: str):
        self.identifier = identifier
    
    @abstractmethod
    def get_source_type(self) -> str:
        pass
    
    @abstractmethod
    def fetch_chunks(self, chunk_size: int = 512, overlap: int = 50) -> Iterator[KnowledgeChunk]:
        pass
    
    @abstractmethod
    def get_summary(self) -> str:
        pass

import os
from pathlib import Path


class FileSource(KnowledgeSource):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_path = Path(file_path)
    
    def get_source_type(self) -> str:
        return "file"
    
    def fetch_chunks(self, chunk_size=512, overlap=50) -> Iterator[KnowledgeChunk]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        step = chunk_size - overlap
        for i, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start:start + chunk_size]
            if not chunk_text.strip():
                continue
            yield KnowledgeChunk(
                content=chunk_text,
                source_type="file",
                source_identifier=str(self.file_path),
                chunk_index=i,
                metadata={'file_name': self.file_path.name}
            )
    
    def get_summary(self) -> str:
        return f"文件 {self.file_path.name}"

try:
    import requests
    from bs4 import BeautifulSoup
    WEB_SUPPORT = True
except ImportError:
    WEB_SUPPORT = False


class WebSource(KnowledgeSource):
    def __init__(self, url: str):
        super().__init__(url)
        self.url = url
        self._text_cache = None
    
    def get_source_type(self) -> str:
        return "url"
    
    def _fetch_text(self) -> str:
        if self._text_cache is not None:
            return self._text_cache
        if not WEB_SUPPORT:
            raise ImportError("需要安装 requests 和 beautifulsoup4")
        
        response = requests.get(self.url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        self._text_cache = '\n'.join(lines)
        return self._text_cache
    
    def fetch_chunks(self, chunk_size=512, overlap=50) -> Iterator[KnowledgeChunk]:
        text = self._fetch_text()
        step = chunk_size - overlap
        for i, start in enumerate(range(0, len(text), step)):
            chunk_text = text[start:start + chunk_size]
            if not chunk_text.strip():
                continue
            yield KnowledgeChunk(
                content=chunk_text,
                source_type="url",
                source_identifier=self.url,
                chunk_index=i,
                metadata={'url': self.url}
            )
    
    def get_summary(self) -> str:
        return f"网页 {self.url}"


class TextSource(KnowledgeSource):
    """纯文本知识源，用于直接学习内联文本"""
    def __init__(self, text: str, identifier: str = "inline_text"):
        super().__init__(identifier)
        self.text = text
    
    def get_source_type(self) -> str:
        return "inline"
    
    def fetch_chunks(self, chunk_size=512, overlap=50) -> Iterator[KnowledgeChunk]:
        step = chunk_size - overlap
        for i, start in enumerate(range(0, len(self.text), step)):
            chunk_text = self.text[start:start + chunk_size]
            if not chunk_text.strip():
                continue
            yield KnowledgeChunk(
                content=chunk_text,
                source_type="inline",
                source_identifier=self.identifier,
                chunk_index=i,
                metadata={'length': len(self.text)}
            )
    
    def get_summary(self) -> str:
        return f"内联文本 ({len(self.text)} 字符)"
