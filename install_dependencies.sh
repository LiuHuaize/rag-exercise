#!/bin/bash

# RAG小说分析项目 - 依赖安装脚本

echo "=================================================="
echo "RAG小说分析项目 - 依赖安装"
echo "=================================================="

# 检查Python版本
echo "检查Python版本..."
python3 --version

# 升级pip
echo "升级pip..."
python3 -m pip install --upgrade pip

# 安装基础依赖
echo "安装基础依赖包..."
python3 -m pip install numpy pandas matplotlib seaborn jupyter

# 安装中文处理
echo "安装中文文本处理包..."
python3 -m pip install jieba

# 安装轻量级版本的依赖（适合CPU环境）
echo "安装RAG相关包..."
python3 -m pip install sentence-transformers
python3 -m pip install chromadb
python3 -m pip install langchain

# 验证安装
echo "验证安装..."
python3 -c "
try:
    import sentence_transformers
    import chromadb
    import langchain
    import jieba
    import pandas
    import numpy
    print('✓ 所有依赖包安装成功！')
except ImportError as e:
    print(f'✗ 安装失败: {e}')
"

echo "=================================================="
echo "安装完成！现在可以运行 python3 src/process_full_novel.py"
echo "=================================================="
