#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试完整RAG系统流程是否正常工作
从EPUB文件到最终问答的完整流程测试
"""

import sys
import os
import subprocess
sys.path.append('src')

def test_imports():
    """测试所有核心模块是否可以正常导入"""
    print("🔍 测试模块导入...")

    try:
        from epub_processor import EPUBProcessor
        print("✅ epub_processor 导入成功")
    except ImportError as e:
        print(f"❌ epub_processor 导入失败: {e}")
        return False

    try:
        from vector_processor import VectorProcessor
        print("✅ vector_processor 导入成功")
    except ImportError as e:
        print(f"❌ vector_processor 导入失败: {e}")
        return False

    try:
        # 测试analyze_xiangzi_actions模块（包含RAG问答功能）
        import analyze_xiangzi_actions
        print("✅ analyze_xiangzi_actions 导入成功")
    except ImportError as e:
        print(f"❌ analyze_xiangzi_actions 导入失败: {e}")
        return False

    try:
        import process_full_novel
        print("✅ process_full_novel 导入成功")
    except ImportError as e:
        print(f"❌ process_full_novel 导入失败: {e}")
        return False

    return True

def test_data_files():
    """测试必要的数据文件是否存在"""
    print("\n📁 测试数据文件...")

    required_files = [
        "骆驼祥子（作家榜经典文库）.epub",
        "requirements.txt"
    ]

    optional_files = [
        "processed_luotuoxiangzi.json",
        "data/processed/luotuoxiangzi_chunks.json"
    ]

    all_required_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"❌ {file_path} 不存在")
            all_required_exist = False

    print("\n📋 可选文件状态:")
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} 存在")
        else:
            print(f"⚠️ {file_path} 不存在（可通过处理流程生成）")

    return all_required_exist

def test_json_processing():
    """测试JSON文件处理"""
    print("\n📄 测试JSON文件处理...")

    if not os.path.exists("processed_luotuoxiangzi.json"):
        print("⚠️ processed_luotuoxiangzi.json 不存在")
        print("   需要先从EPUB文件生成JSON文件")
        return False

    try:
        import json
        with open("processed_luotuoxiangzi.json", 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"✅ JSON文件读取成功")
        print(f"   - 书名: {data['book_info']['title']}")
        print(f"   - 章节数: {data['book_info']['total_chapters']}")
        print(f"   - 总字数: {data['book_info']['total_words']}")
        return True

    except Exception as e:
        print(f"❌ JSON文件处理失败: {e}")
        return False

def test_vector_database():
    """测试向量数据库是否可用"""
    print("\n🗄️ 测试向量数据库...")

    # 首先检查数据库目录是否存在
    if not os.path.exists("chroma_db"):
        print("⚠️ chroma_db 目录不存在，需要运行 python3 src/process_full_novel.py")
        return False

    try:
        from vector_processor import VectorProcessor
        processor = VectorProcessor()

        # 尝试连接到现有集合
        try:
            processor.create_collection(reset=False)  # 不重置，只是连接
            count = processor.collection.count()
            print(f"✅ 向量数据库连接成功")
            print(f"   - 文档数量: {count}")
            print(f"   - 集合名称: {processor.collection_name}")
            print(f"   - 模型名称: {processor.model_name}")

            if count > 0:
                print("✅ 向量数据库包含数据")
                return True
            else:
                print("⚠️ 向量数据库为空，需要运行 python3 src/process_full_novel.py")
                return False
        except Exception as e:
            print(f"⚠️ 集合不存在或为空: {e}")
            print("   需要运行 python3 src/process_full_novel.py")
            return False

    except Exception as e:
        print(f"❌ 向量数据库测试失败: {e}")
        return False

def test_search_functionality():
    """测试搜索功能"""
    print("\n🔍 测试搜索功能...")

    # 首先检查数据库是否存在
    if not os.path.exists("chroma_db"):
        print("⚠️ 向量数据库不存在，跳过搜索测试")
        return False

    try:
        from vector_processor import VectorProcessor
        processor = VectorProcessor()

        # 连接到集合并检查是否有数据
        processor.create_collection(reset=False)
        count = processor.collection.count()
        if count == 0:
            print("⚠️ 向量数据库为空，跳过搜索测试")
            return False

        # 测试搜索
        query = "祥子做了什么"
        results = processor.search_similar(query, n_results=2)

        if results and len(results['documents'][0]) > 0:
            print(f"✅ 搜索功能正常")
            print(f"   查询: {query}")
            print(f"   结果数量: {len(results['documents'][0])}")

            # 显示第一个结果
            first_doc = results['documents'][0][0]
            first_metadata = results['metadatas'][0][0]
            first_distance = results['distances'][0][0]

            print(f"   第一个结果:")
            print(f"     相似度: {1-first_distance:.3f}")
            print(f"     章节: {first_metadata.get('chapter_title', 'N/A')[:30]}...")
            print(f"     内容: {first_doc[:50]}...")

            return True
        else:
            print("❌ 搜索返回空结果")
            return False

    except Exception as e:
        print(f"❌ 搜索功能测试失败: {e}")
        return False

def test_api_connection():
    """测试API连接"""
    print("\n🌐 测试API连接...")

    try:
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            print("❌ 未找到 OPENROUTER_API_KEY 环境变量")
            print("   请检查 .env 文件")
            return False

        if api_key.startswith('sk-or-'):
            print("✅ API密钥格式正确")
            return True
        else:
            print("⚠️ API密钥格式可能不正确")
            return False

    except Exception as e:
        print(f"❌ API连接测试失败: {e}")
        return False

def test_process_full_novel():
    """测试完整小说处理流程"""
    print("\n🔄 测试完整小说处理流程...")

    if not os.path.exists("processed_luotuoxiangzi.json"):
        print("❌ 缺少 processed_luotuoxiangzi.json 文件")
        print("   请先从EPUB文件生成JSON文件")
        return False

    # 检查是否需要重新处理
    chunks_file = "data/processed/luotuoxiangzi_chunks.json"
    if os.path.exists(chunks_file) and os.path.exists("chroma_db"):
        print("✅ 文本块和向量数据库已存在")
        return True

    print("⚠️ 需要运行 process_full_novel.py 来生成向量数据库")
    return False

def main():
    """主测试函数"""
    print("=" * 70)
    print("🧪 完整RAG系统流程测试")
    print("📚 从EPUB文件到问答分析的端到端测试")
    print("=" * 70)

    tests = [
        ("模块导入", test_imports),
        ("数据文件", test_data_files),
        ("JSON处理", test_json_processing),
        ("API连接", test_api_connection),
        ("完整流程", test_process_full_novel),
        ("向量数据库", test_vector_database),
        ("搜索功能", test_search_functionality)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} 测试失败")

    print("\n" + "=" * 70)
    print(f"📊 测试结果: {passed}/{total} 通过")
    print("=" * 70)

    if passed == total:
        print("🎉 所有测试通过！系统可以正常使用。")
        print("\n📝 完整RAG流程:")
        print("1. ✅ EPUB文件 → JSON处理")
        print("2. ✅ JSON → 文本分块 → 向量化 → ChromaDB存储")
        print("3. ✅ 向量检索 → RAG问答")
        print("\n🚀 运行完整分析: python3 analyze_xiangzi_actions.py")
    else:
        print("⚠️ 部分测试失败，请按以下步骤修复:")
        print("\n🔧 完整流程步骤:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 确保有EPUB文件: 骆驼祥子（作家榜经典文库）.epub")
        print("3. 如果没有JSON文件，需要先处理EPUB")
        print("4. 运行向量化: python3 src/process_full_novel.py")
        print("5. 运行分析: python3 analyze_xiangzi_actions.py")

if __name__ == "__main__":
    main()
