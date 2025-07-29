import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from typing import List
from config import config

DATA_PATH = config["csv_path"]
PKL_PATH = config["pkl_path"]
EMBED_PATH = config["embed_path"]
MAX_TEXT_LENGTH = 500
MODEL_NAME = "BAAI/bge-base-zh"

def generate_embeddings_from_csv():
    print("🚀 载入模型中...")
    # embed_model = SentenceTransformer("BAAI/bge-base-zh", device="cuda")
    embed_model = SentenceTransformer("BAAI/bge-base-zh-v1.5", device="cuda")

    print("📄 读取 CSV 数据...")
    df = pd.read_csv(DATA_PATH)
    df = df.fillna('')

    print("🧩 构建文本字段...")
    def card_to_text(row) -> str:
        weighted_text = (
            f"角色名：{row['name']}\n"
            f"简介：{row['profile_text']}\n" * 2 +
            f"核心特质：{row['traits']}\n" * 3 +
            f"典型需求：{row['default_needs']}\n" * 2 +
            f"情感模式：{row['emotional_patterns']}\n"
            f"写作风格：{row['writing_style']}\n"
            f"依恋类型：{row['attachment_style']}\n"
            f"推荐语： {row['one_sentence_summary']}"
        )
        return re.sub(r'[\n\t\u3000]+', ' ', weighted_text)[:MAX_TEXT_LENGTH]

    texts = df.apply(card_to_text, axis=1).tolist()

    print("🧠 编码辅助函数初始化...")
    def encode_field(texts: List[str]) -> List[np.ndarray]:
        arr = embed_model.encode(
            texts,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return [arr[i] for i in range(arr.shape[0])]

    print("🔢 开始字段编码...")
    df["traits_vec"]    = encode_field(df["traits"].tolist())
    df["needs_vec"]     = encode_field(df["default_needs"].tolist())
    df["profile_vec"]   = encode_field(df["profile_text"].tolist())
    df["emotions_vec"]  = encode_field(df["emotional_patterns"].tolist())
    df["style_vec"]     = encode_field(df["writing_style"].tolist())
    df["one_sentence_summary_vec"] = encode_field(df["one_sentence_summary"].tolist())

    print("🔎 编码全文嵌入（用于 faiss 检索）...")
    full_embeddings = embed_model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print("💾 正在保存向量...")
    df.to_pickle(PKL_PATH)
    np.save(EMBED_PATH, full_embeddings)

    print("✅ 保存完成：")
    print(f" - 字段向量 → {PKL_PATH}")
    print(f" - 检索嵌入 → {EMBED_PATH}")