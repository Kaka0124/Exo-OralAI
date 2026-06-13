# Exo-OralAI 源数据

## 数据总览

| 数据 | 来源 | 用途 | 大小 | 状态 |
|------|------|------|------|:--:|
| GSE30784 | NCBI GEO | 主训练集，OSCC vs 正常口腔黏膜 | 54,675 genes × 229 samples | ✅ |
| GSE41613 | NCBI GEO | 外部预后验证集，含生存数据 | 54,613 genes × 97 samples | ✅ |
| TCGA-HNSC | NCI GDC | 预后建模，310 肿瘤 + 32 正常 | 见 TCGA下载说明.txt | ⚠️ 需手动下载 |
| ExoCarta | exocarta.org | 外泌体货物数据库 | 124 genes | ✅ |
| Vesiclepedia | microvesicles.org | 细胞外囊泡数据库 v4.1 | 57 genes | ✅ |

---

## 文件夹结构

```
源数据/
├── README.md                        ← 本文件
├── GEO/
│   ├── GSE30784_expression_matrix.csv   ← 训练集表达矩阵
│   ├── GSE30784_family.soft.gz          ← 原始SOFT文件
│   ├── GSE41613_expression_matrix.csv   ← 验证集表达矩阵
│   └── GSE41613_family.soft.gz          ← 原始SOFT文件
├── TCGA/
│   └── TCGA下载说明.txt                 ← TCGA数据下载指南
└── Exosome_DB/
    ├── ExoCarta_gene_list.txt           ← ExoCarta外泌体基因列表
    ├── Vesiclepedia_gene_list.txt       ← Vesiclepedia外泌体基因列表
    └── exosome_annotation.json          ← 综合外泌体注释(E_score)
```

---

## 各数据源详细说明

### 1. GSE30784
- **全名**: Gene expression profiling of oral squamous cell carcinoma
- **链接**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE30784
- **平台**: Affymetrix GPL570 (HG-U133 Plus 2.0)
- **样本**: 167 OSCC 肿瘤 + 45 正常口腔黏膜（部分样本被排除后为229个）
- **已处理文件**: `GSE30784_expression_matrix.csv` — 可直接用于差异分析

### 2. GSE41613
- **全名**: A 13-gene signature prognostic of HPV-negative OSCC
- **链接**: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41613
- **平台**: Affymetrix GPL570
- **样本**: 97 例 HPV 阴性 OSCC，含完整总生存期标注
- **中位随访**: 65.0 个月，51 个死亡事件
- **已处理文件**: `GSE41613_expression_matrix.csv` — 含生存时间列

### 3. TCGA-HNSC (Oral Cavity Subset)
- **项目**: The Cancer Genome Atlas - Head and Neck Squamous Cell Carcinoma
- **链接**: https://portal.gdc.cancer.gov/projects/TCGA-HNSC
- **口腔亚群**: 310 肿瘤 + 32 癌旁正常（筛选 anatomic_site 为口腔位点）
- **数据类型**: HTSeq-Counts (RNA-seq)
- **下载方法**: 见 `TCGA/TCGA下载说明.txt`
- **备选**: UCSC Xena (https://xenabrowser.net/) 可直接下载处理好的数据

### 4. ExoCarta
- **链接**: http://exocarta.org/
- **版本**: 2023 release
- **内容**: 实验验证的外泌体蛋白质/mRNA/miRNA 列表
- **论文中用法**: 标记基因是否在外泌体中检测到 → E_score = 1.0 或 0.7

### 5. Vesiclepedia
- **链接**: http://microvesicles.org/
- **版本**: v4.1
- **内容**: 细胞外囊泡（含外泌体）分子货物数据库
- **论文中用法**: 与 ExoCarta 互补验证 → 双库命中 E_score=1.0

---

## 使用流程

1. 下载 GSE30784 和 GSE41613 → 已在 GEO 文件夹中
2. 按 TCGA下载说明.txt 下载 TCGA-HNSC 数据
3. 用 `../code/main.py` 运行全流程：
   ```bash
   cd ../code
   pip install -r requirements.txt
   python main.py
   ```

---

**下载日期**: 2026年6月
**论文**: Exo-OralAI (ISAIMS 2026)
