#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理完整小说JSON文件的脚本
将《骆驼祥子》的章节数据转换为文本块并生成向量
"""

import json
import os
import logging
from typing import List, Dict, Any
from vector_processor import VectorProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_novel_to_chunks(json_file_path: str, output_file_path: str, 
                           chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
    """
    将小说JSON文件转换为文本块
    
    Args:
        json_file_path: 输入的JSON文件路径
        output_file_path: 输出的chunks JSON文件路径
        chunk_size: 每个文本块的大小（字符数）
        overlap: 重叠字符数
        
    Returns:
        文本块列表
    """
    try:
        # 读取原始JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as f:
            novel_data = json.load(f)
        
        logger.info(f"读取小说数据: {novel_data['book_info']['title']}")
        logger.info(f"总章节数: {novel_data['book_info']['total_chapters']}")
        logger.info(f"总字数: {novel_data['book_info']['total_words']}")
        
        chunks = []
        chunk_id = 0
        
        # 处理每个章节
        for chapter in novel_data['chapters']:
            content = chapter['content']
            chapter_num = chapter['chapter_num']
            chapter_title = chapter['chapter_title']
            
            # 将章节内容分块
            start = 0
            while start < len(content):
                end = start + chunk_size
                
                # 如果不是最后一块，尝试在句号处断开
                if end < len(content):
                    # 寻找最近的句号
                    for i in range(end, max(start + chunk_size // 2, end - 100), -1):
                        if content[i] in '。！？':
                            end = i + 1
                            break
                
                chunk_content = content[start:end].strip()
                
                if len(chunk_content) > 50:  # 只保留有意义的文本块
                    chunk_id += 1
                    
                    # 提取人物名称（简单版本）
                    characters = extract_characters_simple(chunk_content)
                    
                    chunk_metadata = {
                        'chunk_id': f"chunk_{chunk_id:04d}",
                        'chapter_num': chapter_num,
                        'chapter_title': chapter_title[:100] + "..." if len(chapter_title) > 100 else chapter_title,
                        'book_title': novel_data['book_info']['title'],
                        'book_author': novel_data['book_info'].get('author', '老舍'),
                        'word_count': len(chunk_content),
                        'characters': characters,
                        'content': chunk_content,
                        'start_position': start,
                        'end_position': end
                    }
                    
                    chunks.append(chunk_metadata)
                
                # 移动到下一个位置，考虑重叠
                start = max(start + chunk_size - overlap, end)
                if start >= len(content):
                    break
        
        # 保存处理后的chunks
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"成功创建 {len(chunks)} 个文本块")
        logger.info(f"文本块已保存到: {output_file_path}")
        
        return chunks
        
    except Exception as e:
        logger.error(f"处理小说文件时出错: {e}")
        raise

def extract_characters_simple(text: str) -> List[str]:
    """
    简单的人物名称提取
    """
    # 《骆驼祥子》主要人物
    main_characters = [
        '祥子', '虎妞', '小福子', '刘四爷', '老马', '小马', 
        '曹先生', '高妈', '阮明', '夏太太', '杨太太', '张妈',
        '二强子', '小文', '老程', '丁四', '孙排长'
    ]
    
    found_characters = []
    for char in main_characters:
        if char in text:
            found_characters.append(char)
    
    return list(set(found_characters))  # 去重

def main():
    """主函数"""
    # 文件路径
    input_file = "processed_luotuoxiangzi.json"
    output_file = "data/processed/luotuoxiangzi_chunks.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    # 转换为文本块
    logger.info("开始处理小说文件...")
    chunks = convert_novel_to_chunks(input_file, output_file, chunk_size=400, overlap=80)
    
    # 初始化向量处理器
    logger.info("初始化向量处理器...")
    processor = VectorProcessor()
    
    # 创建新的集合（重置之前的数据）
    processor.create_collection(reset=True)
    
    # 处理文本块并生成向量
    logger.info("开始生成向量并存储到数据库...")
    result = processor.process_json_chunks(output_file)
    
    # 显示结果
    print("\n" + "="*50)
    print("处理完成！")
    print("="*50)
    print(f"总文本块数: {result['total_chunks']}")
    print(f"向量维度: {result['vector_dimension']}")
    print(f"数据库中的向量数: {result['collection_count']}")
    print(f"使用的模型: {result['model_name']}")
    
    # 获取统计信息
    stats = processor.get_collection_stats()
    print(f"集合名称: {stats['collection_name']}")
    print(f"数据库路径: {stats['chroma_persist_directory']}")
    
    # 测试搜索功能
    print("\n" + "="*50)
    print("测试搜索功能")
    print("="*50)
    
    test_queries = [
        "祥子做了什么",
        "祥子的车怎么了",
        "虎妞和祥子的关系",
        "祥子的梦想是什么",
        "骆驼的故事"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = processor.search_similar(query, n_results=3)
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            print(f"  结果 {i+1} (相似度: {1-distance:.3f}):")
            print(f"    章节: {metadata.get('chapter_title', 'N/A')[:50]}...")
            print(f"    人物: {metadata.get('characters', 'N/A')}")
            print(f"    内容: {doc[:100]}...")
            print()

if __name__ == "__main__":
    main()
