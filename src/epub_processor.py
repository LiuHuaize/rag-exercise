#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPUB电子书处理模块
专门用于处理中文小说的EPUB格式文件，提取文本和元数据
"""

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import re
import jieba
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EPUBProcessor:
    """EPUB文件处理器，专门优化中文小说解析"""
    
    def __init__(self):
        self.chapter_pattern = re.compile(r'第[一二三四五六七八九十\d]+[章回节]')
        
    def extract_text_and_metadata(self, epub_path: str) -> List[Dict]:
        """
        从EPUB文件中提取文本和元数据
        
        Args:
            epub_path: EPUB文件路径
            
        Returns:
            包含章节信息的字典列表
        """
        try:
            book = epub.read_epub(epub_path)
            chapters = []
            
            # 获取书籍基本信息
            book_title = book.get_metadata('DC', 'title')[0][0] if book.get_metadata('DC', 'title') else "未知书名"
            book_author = book.get_metadata('DC', 'creator')[0][0] if book.get_metadata('DC', 'creator') else "未知作者"
            
            logger.info(f"正在处理: {book_title} - {book_author}")
            
            chapter_num = 0
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    # 解析HTML内容
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    
                    # 移除脚本和样式
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # 提取文本
                    text = soup.get_text()
                    text = self._clean_text(text)
                    
                    if len(text.strip()) < 100:  # 跳过太短的内容
                        continue
                    
                    # 检测章节标题
                    chapter_title = self._extract_chapter_title(text)
                    if chapter_title:
                        chapter_num += 1
                    
                    chapter_info = {
                        'chapter_num': chapter_num,
                        'chapter_title': chapter_title or f"第{chapter_num}部分",
                        'content': text,
                        'word_count': len(text),
                        'file_name': item.get_name(),
                        'book_title': book_title,
                        'book_author': book_author
                    }
                    
                    chapters.append(chapter_info)
            
            logger.info(f"成功提取 {len(chapters)} 个章节")
            return chapters
            
        except Exception as e:
            logger.error(f"处理EPUB文件时出错: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        
        # 移除页眉页脚等无关内容
        text = re.sub(r'第\s*\d+\s*页', '', text)
        text = re.sub(r'www\.\w+\.\w+', '', text)
        
        # 标准化标点符号
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _extract_chapter_title(self, text: str) -> str:
        """提取章节标题"""
        lines = text.split('\n')[:5]  # 只检查前5行
        
        for line in lines:
            line = line.strip()
            if self.chapter_pattern.search(line):
                return line
        
        return ""
    
    def create_chunks_with_metadata(self, chapters: List[Dict], 
                                  chunk_size: int = 400, 
                                  overlap: int = 80) -> List[Dict]:
        """
        将章节内容分块并添加元数据
        
        Args:
            chapters: 章节列表
            chunk_size: 分块大小
            overlap: 重叠字符数
            
        Returns:
            包含分块和元数据的列表
        """
        chunks = []
        chunk_id = 0
        
        for chapter in chapters:
            content = chapter['content']
            
            # 按段落分割
            paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
            
            current_chunk = ""
            paragraph_start = 0
            
            for i, paragraph in enumerate(paragraphs):
                # 如果添加当前段落会超过chunk_size，则创建新chunk
                if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                    chunk_id += 1
                    
                    # 创建chunk元数据
                    chunk_metadata = {
                        'chunk_id': f"chunk_{chunk_id:04d}",
                        'chapter_num': chapter['chapter_num'],
                        'chapter_title': chapter['chapter_title'],
                        'book_title': chapter['book_title'],
                        'book_author': chapter['book_author'],
                        'paragraph_range': f"{paragraph_start}-{i-1}",
                        'word_count': len(current_chunk),
                        'characters': self._extract_characters(current_chunk),
                        'content': current_chunk
                    }
                    
                    chunks.append(chunk_metadata)
                    
                    # 处理重叠
                    if overlap > 0:
                        current_chunk = current_chunk[-overlap:] + paragraph
                    else:
                        current_chunk = paragraph
                    paragraph_start = i
                else:
                    current_chunk += paragraph + "\n"
            
            # 处理最后一个chunk
            if current_chunk.strip():
                chunk_id += 1
                chunk_metadata = {
                    'chunk_id': f"chunk_{chunk_id:04d}",
                    'chapter_num': chapter['chapter_num'],
                    'chapter_title': chapter['chapter_title'],
                    'book_title': chapter['book_title'],
                    'book_author': chapter['book_author'],
                    'paragraph_range': f"{paragraph_start}-{len(paragraphs)-1}",
                    'word_count': len(current_chunk),
                    'characters': self._extract_characters(current_chunk),
                    'content': current_chunk.strip()
                }
                chunks.append(chunk_metadata)
        
        logger.info(f"创建了 {len(chunks)} 个文本块")
        return chunks
    
    def _extract_characters(self, text: str) -> List[str]:
        """从文本中提取人物名称"""
        # 简单的人物名称提取（可以根据具体小说优化）
        common_names = ['祥子', '虎妞', '小福子', '刘四爷', '老马', '小马']
        found_characters = []
        
        for name in common_names:
            if name in text:
                found_characters.append(name)
        
        return found_characters

def main():
    """测试函数"""
    processor = EPUBProcessor()

    # 示例用法
    epub_file = "骆驼祥子（作家榜经典文库）.epub"  # 替换为实际文件路径
    
    # 提取章节
    chapters = processor.extract_text_and_metadata(epub_file)
    
    if chapters:
        print(f"成功提取 {len(chapters)} 个章节")
        print(f"第一章标题: {chapters[0]['chapter_title']}")
        print(f"第一章字数: {chapters[0]['word_count']}")
        
        # 创建分块
        chunks = processor.create_chunks_with_metadata(chapters)
        print(f"创建了 {len(chunks)} 个文本块")
        
        # 显示第一个chunk的信息
        if chunks:
            first_chunk = chunks[0]
            print(f"\n第一个文本块信息:")
            print(f"ID: {first_chunk['chunk_id']}")
            print(f"章节: {first_chunk['chapter_title']}")
            print(f"字数: {first_chunk['word_count']}")
            print(f"人物: {first_chunk['characters']}")
            print(f"内容预览: {first_chunk['content'][:100]}...")

if __name__ == "__main__":
    main()
