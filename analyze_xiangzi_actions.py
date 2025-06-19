#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析骆驼祥子的行为和经历
找出所有有骆驼祥子的章节，RAG这些章节的内容，询问骆驼祥子干了什么，
然后把所有信息给到Gemini大模型进行中文输出
"""

import sys
import os
import json
sys.path.append('src')

from vector_processor import VectorProcessor
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def find_xiangzi_chapters():
    """找出所有包含'骆驼祥子'或'祥子'的章节"""
    
    print("🔍 正在查找包含祥子的章节...")
    
    # 读取处理过的小说数据
    with open('processed_luotuoxiangzi.json', 'r', encoding='utf-8') as f:
        novel_data = json.load(f)
    
    xiangzi_chapters = []
    
    for chapter in novel_data['chapters']:
        chapter_content = chapter['content']
        chapter_title = chapter['chapter_title']
        chapter_num = chapter['chapter_num']
        
        # 检查章节内容是否包含祥子
        if '祥子' in chapter_content or '骆驼祥子' in chapter_content:
            xiangzi_chapters.append({
                'chapter_num': chapter_num,
                'chapter_title': chapter_title[:100] + "..." if len(chapter_title) > 100 else chapter_title,
                'content_preview': chapter_content[:200] + "..." if len(chapter_content) > 200 else chapter_content,
                'word_count': len(chapter_content)
            })
    
    print(f"📚 找到 {len(xiangzi_chapters)} 个包含祥子的章节:")
    for chapter in xiangzi_chapters:
        print(f"  第{chapter['chapter_num']}章: {chapter['chapter_title']}")
    
    return xiangzi_chapters

def analyze_xiangzi_actions():
    """分析祥子在各章节中的行为"""
    
    print("\n" + "="*80)
    print("🎭 《骆驼祥子》主角行为分析")
    print("="*80)
    
    # 找出包含祥子的章节
    xiangzi_chapters = find_xiangzi_chapters()
    
    # 初始化RAG系统
    print("\n🚀 正在初始化RAG系统...")
    processor = VectorProcessor()
    processor.create_collection(reset=False)
    
    # 初始化OpenAI客户端
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
    
    # 针对每个章节询问祥子的行为
    all_actions = []
    
    print(f"\n📖 开始分析 {len(xiangzi_chapters)} 个章节中祥子的行为...")
    
    for i, chapter in enumerate(xiangzi_chapters, 1):
        chapter_num = chapter['chapter_num']
        chapter_title = chapter['chapter_title']
        
        print(f"\n【第{chapter_num}章分析】: {chapter_title}")
        print("-" * 60)
        
        # 构建针对该章节的查询
        queries = [
            f"第{chapter_num}章祥子做了什么",
            f"第{chapter_num}章祥子的行为",
            f"第{chapter_num}章祥子的经历",
            f"祥子在第{chapter_num}章的活动"
        ]
        
        chapter_context = ""
        
        for query in queries:
            print(f"  🔍 查询: {query}")
            
            # 搜索相关内容
            results = processor.search_similar(query, n_results=3)
            
            for j, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                # 只选择相关度高的结果
                similarity = 1 - distance
                if similarity > 0.3:  # 只要相似度高于0.3的结果
                    chapter_context += f"相关内容 (相似度: {similarity:.3f}): {doc}\n\n"
                    print(f"    ✓ 找到相关内容 (相似度: {similarity:.3f})")
        
        if chapter_context:
            all_actions.append({
                'chapter_num': chapter_num,
                'chapter_title': chapter_title,
                'context': chapter_context
            })
            print(f"  ✅ 第{chapter_num}章分析完成")
        else:
            print(f"  ⚠️  第{chapter_num}章未找到足够相关的内容")
    
    return all_actions

def generate_comprehensive_analysis(all_actions):
    """使用Gemini生成综合分析"""
    
    print(f"\n🤖 正在使用Gemini生成综合分析...")
    
    # 初始化OpenAI客户端
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv('OPENROUTER_API_KEY'),
    )
    
    # 构建完整的上下文
    full_context = ""
    for action in all_actions:
        full_context += f"\n=== 第{action['chapter_num']}章: {action['chapter_title']} ===\n"
        full_context += action['context']
        full_context += "\n" + "-"*50 + "\n"
    
    system_prompt = """你是一个专门分析老舍小说《骆驼祥子》的文学专家。

请根据提供的各章节文本片段，全面分析主角祥子在整个故事中的行为和经历。

请按照以下结构进行分析：

1. **祥子的主要行为总结**
   - 按时间顺序梳理祥子的重要行为
   - 分析每个行为的动机和背景

2. **祥子的人生轨迹**
   - 祥子的起点和目标
   - 关键的转折点
   - 最终的结局

3. **祥子的性格变化**
   - 初期的性格特点
   - 中期的变化过程
   - 后期的堕落过程

4. **祥子行为的深层含义**
   - 反映的社会问题
   - 体现的人性特点
   - 作者想要表达的主题

请用清晰的中文进行详细分析，语言要准确、生动。"""

    user_message = f"""请基于以下从《骆驼祥子》各章节中提取的关于祥子行为的文本片段，进行全面分析：

{full_context}

请详细分析祥子在整个故事中做了什么，他的行为如何体现了他的性格变化和命运轨迹。"""

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "RAG QA System",
            },
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=3000
        )
        
        answer = completion.choices[0].message.content
        
        print("\n" + "="*80)
        print("📋 《骆驼祥子》主角行为综合分析")
        print("="*80)
        print(answer)
        print("="*80)
        
        return answer
        
    except Exception as e:
        print(f"❌ 生成综合分析时出错: {e}")
        
        # 如果API调用失败，提供基础分析
        print("\n📋 基于检索内容的基础分析：")
        print("="*80)
        print("从检索到的内容可以看出，祥子的主要行为包括：")
        
        for action in all_actions:
            print(f"\n第{action['chapter_num']}章:")
            # 简单提取一些关键信息
            context = action['context'][:300] + "..." if len(action['context']) > 300 else action['context']
            print(f"  {context}")
        
        return None

def main():
    """主函数"""
    
    print("🎬 开始分析《骆驼祥子》主角行为...")
    
    # 分析祥子在各章节中的行为
    all_actions = analyze_xiangzi_actions()
    
    if not all_actions:
        print("❌ 未找到足够的相关内容进行分析")
        return
    
    print(f"\n✅ 成功分析了 {len(all_actions)} 个章节的内容")
    
    # 生成综合分析
    analysis = generate_comprehensive_analysis(all_actions)
    
    if analysis:
        # 保存分析结果
        output_file = "xiangzi_behavior_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("《骆驼祥子》主角行为综合分析\n")
            f.write("="*80 + "\n\n")
            f.write(analysis)
        
        print(f"\n💾 分析结果已保存到: {output_file}")
    
    print("\n🎉 分析完成！")

if __name__ == "__main__":
    main()
