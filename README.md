# rag-exercise
# 《骆驼祥子》RAG分析系统

这是一个基于RAG（检索增强生成）技术的中文文学分析系统，专门用于分析老舍小说《骆驼祥子》中主角祥子的行为轨迹和人物发展。

## 🎯 项目特色

- **完整的RAG工作流程**：从EPUB文件处理到最终的AI分析
- **中文优化**：专门针对中文文本的处理和分析
- **学术导向**：适合作为RAG技术学习和文学分析的案例
- **端到端自动化**：一键完成从原始文本到深度分析的全过程

## 📋 环境要求

- Python 3.8+
- 8GB+ RAM（推荐）
- 网络连接（用于下载模型和API调用）

## 🚀 完整流程：从EPUB到RAG问答

### 第一步：环境准备

#### 1.1 安装依赖
```bash
# 方式1：直接安装
pip install -r requirements.txt

# 方式2：使用安装脚本
bash install_dependencies.sh
```

#### 1.2 配置API密钥
创建 `.env` 文件：
```bash
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 第二步：EPUB到JSON转换

#### 2.1 确认文件结构
```
rag-exercise/
├── 骆驼祥子（作家榜经典文库）.epub  # ✅ 原始EPUB文件
├── src/
│   ├── epub_processor.py             # 🔥 EPUB处理模块
│   ├── vector_processor.py           # 向量处理模块（库）
│   ├── process_full_novel.py         # 主要处理脚本
│   └── rag_qa_system.py              # RAG系统核心
├── analyze_xiangzi_actions.py        # 分析脚本
└── requirements.txt
```

#### 2.2 转换EPUB为JSON格式

如果你还没有 `processed_luotuoxiangzi.json` 文件，需要先从EPUB文件生成：

```bash
python3 src/epub_processor.py
```

**这个步骤会：**
- 📖 读取 `骆驼祥子（作家榜经典文库）.epub` 文件
- 🔍 提取章节结构和元数据（书名、作者、章节标题）
- 🧹 清理文本内容（移除HTML标签、标准化标点符号）
- � 识别主要人物（祥子、虎妞、小福子、刘四爷等）
- 📝 生成结构化的JSON文件

**预期输出：**
```
INFO:__main__:正在处理: 骆驼祥子 - 老舍
INFO:__main__:成功提取 26 个章节
成功提取 26 个章节
第一章标题: 第一章
第一章字数: 5234
创建了 374 个文本块

第一个文本块信息:
ID: chunk_0001
章节: 第一章
字数: 400
人物: ['祥子']
内容预览: 祥子拉着洋车在北京的胡同里穿行...
```

**生成的文件：**
- `processed_luotuoxiangzi.json` - 包含26个章节的结构化数据

### 第三步：文本处理和向量化

#### 3.1 运行文本处理和向量化
```bash
python3 src/process_full_novel.py
```

**这个步骤会：**
- 📖 读取 `processed_luotuoxiangzi.json`（26章，139,384字）
- ✂️ 智能文本分块（374个块，每块400字符，重叠80字符）
- 🧠 使用BGE-small-zh-v1.5模型生成512维向量
- 💾 存储到ChromaDB向量数据库
- 🔍 自动测试搜索功能

**预期输出：**
```
INFO:__main__:读取小说数据: 骆驼祥子
INFO:__main__:总章节数: 26
INFO:__main__:总字数: 139384
INFO:__main__:成功创建 374 个文本块
...
==================================================
处理完成！
==================================================
总文本块数: 374
向量维度: 512
数据库中的向量数: 374
使用的模型: BAAI/bge-small-zh-v1.5
```

### 第四步：RAG问答分析

#### 4.1 运行祥子行为分析
```bash
python3 analyze_xiangzi_actions.py
```

**这个步骤会：**
- 🔍 自动识别26个包含祥子的章节
- 📝 对每章节进行4个维度的查询：
  - "第X章祥子做了什么"
  - "第X章祥子的行为"
  - "第X章祥子的经历"
  - "祥子在第X章的活动"
- 🎯 向量检索相关文本片段
- 🤖 使用Gemini-2.5-pro生成综合分析
- 📄 输出完整分析报告

**预期输出：**
```
🎭 《骆驼祥子》主角行为分析
📚 找到 26 个包含祥子的章节
🚀 正在初始化RAG系统...
📖 开始分析 26 个章节中祥子的行为...
...
✅ 成功分析了 26 个章节的内容
🤖 正在使用Gemini生成综合分析...
💾 分析结果已保存到: xiangzi_behavior_analysis.txt
🎉 分析完成！
```

## 📊 核心技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| **文本处理** | Python + JSON | 章节分块，元数据提取 |
| **向量化** | BGE-small-zh-v1.5 | 512维中文语义向量 |
| **向量数据库** | ChromaDB | 本地向量存储和检索 |
| **RAG检索** | 语义相似度搜索 | 余弦相似度匹配 |
| **AI分析** | Gemini-2.5-pro | 通过OpenRouter API |

## 📁 生成的文件

| 文件 | 描述 |
|------|------|
| `data/processed/luotuoxiangzi_chunks.json` | 374个文本块数据 |
| `chroma_db/` | ChromaDB向量数据库目录 |
| `xiangzi_behavior_analysis.txt` | 最终的文学分析报告 |

## 🔧 故障排除

### 常见问题

#### 1. 向量维度不匹配
```bash
# 解决方案：删除数据库重新生成
rm -rf chroma_db/
python3 src/process_full_novel.py
```

#### 2. API调用失败
- 检查 `.env` 文件中的API密钥
- 确认网络连接正常
- 验证OpenRouter账户余额

#### 3. 内存不足
- 减少批处理大小（修改 `process_full_novel.py` 中的参数）
- 调整文本块大小

### 完全重新开始
```bash
# 清理所有生成的文件
rm -rf chroma_db/
rm -rf data/processed/
rm -f xiangzi_behavior_analysis.txt

# 重新运行完整流程
python3 src/process_full_novel.py
python3 analyze_xiangzi_actions.py
```

## 🧪 系统测试

运行完整性测试：
```bash
python3 test_system.py
```

## 🎓 学习价值

这个项目展示了：
- **RAG技术的完整实现**：从文本处理到最终问答
- **中文NLP处理**：分词、向量化、语义检索
- **向量数据库应用**：ChromaDB的实际使用
- **AI集成**：大语言模型在文学分析中的应用

## 📚 扩展应用

这个系统可以扩展用于：
- 其他中文小说的分析
- 不同角色的行为分析
- 主题和情感分析
- 比较文学研究

---

**开始你的RAG文学分析之旅！** 🚀
