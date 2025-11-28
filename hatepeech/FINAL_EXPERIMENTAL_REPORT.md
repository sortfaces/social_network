# 仇恨言论检测复现与扩展实验报告

**日期**：2025年11月28日  
**实验人员**：gao  
**复现对象**：Waseem, Z., & Hovy, D. (2016). *Are You a Racist or Am I Seeing Things? Annotator Influence on Hate Speech Detection on Twitter*.

---

## 1. 实验概述 (Abstract)

本实验旨在复现 Waseem 和 Hovy (2016) 关于仇恨言论检测中“标注者影响”的研究成果，并在此基础上构建可运行的仇恨言论分类模型。由于原数据集（Waseem 2016）受限于 Twitter 隐私政策仅提供了推文 ID，本实验采取了**“分步复现 + 替代数据扩展”**的策略：
1.  **统计复现**：利用原数据集的标注文件，复现了专家与业余标注者之间的一致性分析（Cohen's Kappa）。
2.  **模型构建**：引入开源的 Davidson et al. (2017) 数据集作为替代文本源，构建并训练了基于 TF-IDF 和逻辑回归的基准分类模型。

实验结果证实了仇恨言论标注具有高度主观性（Kappa = 0.52），同时在替代数据集上实现了 85.35% 的分类准确率。

---

## 2. 实验方法与数据 (Methodology & Data)

### 2.1 数据集一：Waseem & Hovy (2016)
*   **内容**：包含推文 ID 和标注信息，无推文文本。
*   **文件结构**：
    *   `NAACL_SRW_2016.csv`: 包含推文 ID 和专家（Zeerak Waseem）的标注。
    *   `NLP+CSS_2016.csv`: 包含专家标注与 CrowdFlower 平台上的业余标注者的详细对比矩阵。
*   **用途**：用于分析标注者之间的一致性（Inter-annotator Agreement）。

### 2.2 数据集二：Davidson et al. (2017)
*   **内容**：包含约 2.5 万条推文的完整文本及其分类标签（Hate Speech, Offensive Language, Neither）。
*   **用途**：作为原数据集文本缺失的替代方案，用于训练和评估监督学习模型。

---

## 3. 实验第一部分：标注者影响分析 (Annotator Influence Analysis)

### 3.1 实验过程
我们编写了 `scripts/analyze_annotators.py` 脚本，读取 `NLP+CSS_2016.csv`。该文件每一行代表一条推文，列包含了专家的标注结果以及多个业余标注者的投票结果。
我们计算了专家标注（Expert）与业余标注者多数投票（Amateur Majority）之间的 **Cohen's Kappa** 系数，并生成了混淆矩阵。

### 3.2 代码分析 (`scripts/analyze_annotators.py`)

本脚本的核心目的是量化专家与业余标注者之间的分歧。

#### 3.2.1 数据加载与预处理
```python
def load_data(filepath):
    # ...
    df = pd.read_csv(filepath, sep='\t')
    # ...
```
*   **分析**：使用 `pandas` 读取制表符分隔（TSV）的 `NLP+CSS_2016.csv` 文件。这是数据处理的第一步，确保数据被正确加载到 DataFrame 中。

#### 3.2.2 多数投票计算 (Majority Vote Calculation)
```python
    # Extract Amateur columns (assuming they start with 'Amateur_')
    amateur_cols = [c for c in df.columns if c.startswith('Amateur_')]
    
    # Function to get majority vote
    def get_majority(row):
        valid_votes = [v for v in row if pd.notna(v) and str(v).strip() != '']
        if not valid_votes:
            return "No Vote"
        return max(set(valid_votes), key=valid_votes.count)

    amateur_majority = amateur_votes.apply(get_majority, axis=1)
```
*   **分析**：
    *   代码首先识别所有以 `Amateur_` 开头的列，这些列代表不同众包工人的标注。
    *   `get_majority` 函数实现了多数投票逻辑：它过滤掉空值，然后使用 `max(set(...), key=count)` 找出出现频率最高的标签。这是处理众包数据的标准方法，用于将多个不确定的标注聚合为一个“共识”标签。

#### 3.2.3 一致性指标计算 (Agreement Metrics)
```python
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa: {kappa:.4f}")
```
*   **分析**：
    *   使用 `sklearn.metrics.cohen_kappa_score` 计算 Kappa 系数。
    *   `y_true` 设为专家标注（Expert），`y_pred` 设为业余者的多数投票（Amateur_Majority）。
    *   **意义**：Kappa 系数考虑了随机一致性的可能性，比单纯的准确率更能反映真实的标注一致性。结果 **0.5205** 表明一致性仅为中等，验证了仇恨言论的主观性。

#### 3.2.4 混淆矩阵生成 (Confusion Matrix)
```python
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    print(cm_df)
```
*   **分析**：
    *   生成混淆矩阵以可视化具体的错误类型。
    *   这部分代码直接输出了我们在报告中引用的表格，揭示了业余标注者倾向于忽略“性别歧视”（Sexism）而过度标记“种族主义”（Racism）的现象。

### 3.3 实验结果分析
*   **一致性得分 (Cohen's Kappa)**: **0.5205**
    *   **分析**：根据统计学标准，0.4-0.6 属于“中等一致性”（Moderate Agreement）。这验证了原论文的核心观点：仇恨言论的定义模糊，普通众包标注者与领域专家之间存在显著的认知差异。

*   **混淆矩阵深度解读**：
    *   **性别歧视 (Sexism) 的漏报**：
        *   专家标记为 `Sexism` 的样本中，有 **378** 条被业余标注者标记为 `Neither`（无仇恨）。
        *   *结论*：业余标注者对性别歧视的敏感度远低于专家，可能忽略了隐晦的歧视言论。
    *   **种族主义 (Racism) 的误报**：
        *   专家标记为 `Neither` 的样本中，有 **258** 条被业余标注者标记为 `Racism`。
        *   *结论*：业余标注者倾向于将某些非仇恨言论误判为种族主义，显示出判定标准的不稳定性。

---

## 4. 实验第二部分：模型训练与评估 (Model Training & Evaluation)

### 4.1 实验过程
由于无法直接获取 Waseem 数据集的文本，我们使用 Davidson 数据集进行模型实验。我们编写了 `scripts/train_model_davidson.py`，构建了一个经典的 NLP 分类流水线。

### 4.2 代码分析 (`scripts/train_model_davidson.py`)

本脚本实现了一个完整的监督学习流程：数据加载 -> 特征提取 -> 模型训练 -> 评估。

#### 4.2.1 数据映射与分割
```python
    class_names = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    df['label_name'] = df['class'].map(class_names)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```
*   **分析**：
    *   将数字标签映射为可读的文本标签，便于理解。
    *   使用 `train_test_split` 进行 8:2 的数据集划分。
    *   **关键点**：使用了 `stratify=y` 参数。由于仇恨言论数据通常极度不平衡（Hate Speech 样本很少），分层抽样（Stratified Sampling）确保了训练集和测试集中各类别的比例一致，避免测试集全是负样本的情况。

#### 4.2.2 机器学习流水线 (Pipeline)
```python
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
```
*   **分析**：
    *   **TfidfVectorizer**：将文本转换为 TF-IDF 特征矩阵。
        *   `stop_words='english'`：去除 "the", "is" 等无意义的常用词。
        *   `max_features=5000`：仅保留 TF-IDF 值最高的 5000 个词，减少噪音并提高训练速度。
    *   **LogisticRegression**：使用逻辑回归作为分类器。
        *   `class_weight='balanced'`：**这是最关键的参数**。它根据类别的频率自动调整权重，频率低的类别（如 Hate Speech）权重高，频率高的类别权重低。这直接解决了数据不平衡导致模型倾向于预测多数类的问题。

#### 4.2.3 模型评估
```python
    print(classification_report(y_test, y_pred, target_names=['Hate Speech', 'Offensive Language', 'Neither']))
```
*   **分析**：
    *   输出精确率（Precision）、召回率（Recall）和 F1 值。对于不平衡数据集，单纯的准确率（Accuracy）具有误导性，F1 值是更重要的参考指标。

### 4.3 实验结果分析
*   **总体准确率 (Accuracy)**: **85.35%**
    *   模型在绝大多数样本上能做出正确判断。

*   **分类详细报告**：
    | 类别 | Precision | Recall | F1-Score | 样本数 |
    | :--- | :--- | :--- | :--- | :--- |
    | **Hate Speech** | 0.31 | 0.60 | 0.41 | 286 |
    | **Offensive Language** | 0.97 | 0.85 | 0.91 | 3838 |
    | **Neither** | 0.78 | 0.95 | 0.85 | 833 |

    *   **分析**：
        1.  **Offensive Language (冒犯性语言)**：模型表现极佳 (F1=0.91)，说明模型很容易识别脏话或攻击性词汇。
        2.  **Hate Speech (仇恨言论)**：表现较差 (Precision=0.31)。这意味着模型虽然召回了 60% 的仇恨言论，但也把大量非仇恨言论误判为仇恨言论。
        3.  **原因推测**：仇恨言论往往依赖于上下文（Context），而不仅仅是特定的词汇（TF-IDF 的局限性）。此外，"Hate Speech" 与 "Offensive Language" 之间的界限非常模糊，模型很难区分“单纯的脏话”和“带有仇恨意图的攻击”。

---

## 5. 结论 (Conclusion)

本次复现实验取得了圆满成功，主要结论如下：

1.  **验证了标注的主观性**：通过对 Waseem 数据集的统计分析，确认了仇恨言论检测任务中，专家与普通大众之间存在显著的认知鸿沟（Kappa=0.52）。这提示我们在构建数据集时，标注者的背景筛选至关重要。
2.  **建立了有效的基准模型**：在 Davidson 数据集上，基于 TF-IDF 的逻辑回归模型达到了 85% 的准确率。
3.  **揭示了任务的难点**：模型在区分“冒犯性语言”和“仇恨言论”时表现挣扎，说明简单的词袋模型（Bag-of-Words）不足以捕捉复杂的语义语境，未来工作应考虑引入 BERT 等深度学习模型来提升对语境的理解能力。

