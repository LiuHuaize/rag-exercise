#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
向量处理模块
将处理后的JSON文档转换为向量并存储到Chroma向量数据库
使用BGE 1.5 small模型 (512维向量)
"""

import json
import os
import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorProcessor:
    """向量处理器，专门处理BGE模型和Chroma数据库的集成"""
    
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", 
                 chroma_persist_directory: str = "./chroma_db"):
        """
        初始化向量处理器
        
        Args:
            model_name: BGE模型名称
            chroma_persist_directory: Chroma数据库持久化目录
        """
        self.model_name = model_name
        self.chroma_persist_directory = chroma_persist_directory
        
        # 初始化BGE模型
        logger.info(f"正在加载BGE模型: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)
        
        # 获取模型向量维度
        self.vector_dimension = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"模型向量维度: {self.vector_dimension}")
        
        # 初始化Chroma客户端
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 创建或获取集合
        self.collection_name = "luotuo_xiangzi_collection"
        self.collection = None
        
    def create_collection(self, reset: bool = False):
        """
        创建或重置Chroma集合
        
        Args:
            reset: 是否重置现有集合
        """
        try:
            if reset:
                try:
                    self.chroma_client.delete_collection(self.collection_name)
                    logger.info(f"已删除现有集合: {self.collection_name}")
                except:
                    pass
            
            # 创建集合，指定embedding函数
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "骆驼祥子小说文本向量集合"},
                embedding_function=None  # 我们手动提供embeddings
            )
            
            logger.info(f"成功创建/获取集合: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"创建集合时出错: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        生成文本向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            向量数组 (n_texts, vector_dimension)
        """
        logger.info(f"正在为 {len(texts)} 个文本生成向量...")
        
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                normalize_embeddings=True,  # 归一化向量
                show_progress_bar=True
            )
            embeddings.append(batch_embeddings)
            
            logger.info(f"已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文本")
        
        all_embeddings = np.vstack(embeddings)
        logger.info(f"生成向量形状: {all_embeddings.shape}")
        
        return all_embeddings
    
    def process_json_chunks(self, json_file_path: str) -> Dict[str, Any]:
        """
        处理JSON文件中的文本块
        
        Args:
            json_file_path: JSON文件路径
            
        Returns:
            处理结果统计
        """
        try:
            # 读取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            logger.info(f"从 {json_file_path} 读取了 {len(chunks_data)} 个文本块")
            
            # 提取文本内容
            texts = [chunk['content'] for chunk in chunks_data]
            
            # 生成向量
            embeddings = self.generate_embeddings(texts)
            
            # 准备存储到Chroma的数据
            ids = []
            metadatas = []
            documents = []
            embeddings_list = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks_data, embeddings)):
                # 生成唯一ID
                chunk_id = chunk.get('chunk_id', f"chunk_{i:04d}")
                ids.append(chunk_id)
                
                # 准备元数据（移除content字段，因为它会作为document存储）
                metadata = {k: v for k, v in chunk.items() if k != 'content'}
                metadata['created_at'] = datetime.now().isoformat()
                metadata['vector_model'] = self.model_name
                metadata['vector_dimension'] = self.vector_dimension
                
                # 处理列表类型的元数据（Chroma不支持复杂类型）
                if 'characters' in metadata and isinstance(metadata['characters'], list):
                    metadata['characters'] = ','.join(metadata['characters'])
                
                metadatas.append(metadata)
                documents.append(chunk['content'])
                embeddings_list.append(embedding.tolist())
            
            # 存储到Chroma
            logger.info("正在存储向量到Chroma数据库...")
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents
            )
            
            # 验证存储
            collection_count = self.collection.count()
            logger.info(f"成功存储 {collection_count} 个向量到数据库")
            
            return {
                'total_chunks': len(chunks_data),
                'vector_dimension': self.vector_dimension,
                'collection_count': collection_count,
                'model_name': self.model_name
            }
            
        except Exception as e:
            logger.error(f"处理JSON文件时出错: {e}")
            raise
    
    def search_similar(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        搜索相似文本
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            搜索结果
        """
        try:
            # 生成查询向量
            query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
            
            # 搜索
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            return results
            
        except Exception as e:
            logger.error(f"搜索时出错: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            count = self.collection.count()
            
            # 获取一个样本来检查数据结构
            sample = self.collection.peek(limit=1)
            
            stats = {
                'collection_name': self.collection_name,
                'total_documents': count,
                'vector_dimension': self.vector_dimension,
                'model_name': self.model_name,
                'chroma_persist_directory': self.chroma_persist_directory
            }
            
            if sample['metadatas']:
                stats['sample_metadata_keys'] = list(sample['metadatas'][0].keys())
            
            return stats
            
        except Exception as e:
            logger.error(f"获取统计信息时出错: {e}")
            return {}

# 注意：此模块作为库使用，不需要直接运行
# 请使用 process_full_novel.py 来处理完整的小说文本
