import os
import json
import requests
from io import BytesIO
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from glob import glob
import openai
from tqdm import tqdm  # 新增
import time
from config import config

MODEL_DIR = config.get("model_dir", "/root/autodl-fs")
# import ZhiPUAI
# # OpenAI 初始化
# client = openai.OpenAI(
#     api_key="sk-MnAWpDIEPZid_w-oeZVjZLLw-XehhhPCSwumOVKu2WT3BlbkFJsgTLbZWkYi3akrNnamWh96rOclJofmj8oXi9k3MagA"
# )

blip_path = os.path.join(MODEL_DIR, "Salesforce", "blip2-opt-2.7b")
processor = Blip2Processor.from_pretrained(blip_path)
blip_model = Blip2ForConditionalGeneration.from_pretrained(blip_path).to("cuda")
CSV_PATH = config["csv_path"]

def init_csv():
    if not os.path.exists(CSV_PATH):
        df = pd.DataFrame(columns=[
            "name", "nickName", "traits", "writing_style",
            "emotional_patterns", "default_needs", "attachment_style",
            "profile_text", "one_sentence_summary"
        ])
        df.to_csv(CSV_PATH, index=False)

def extract_text(user_data):
    # user = user_data['data']['user']
    user = user_data['user']
    field_names = [
        "genderType", "mbtiType", "zodiacType", "birthday",
        "bio", "job", "edu", "region", "pet", "thirdPartyAuth", "lastSyncTime"
    ]
    user_texts = []
    for f in field_names:
        v = user.get(f)
        if v and v != "Unknown" and v is not None:
            user_texts.append(f"{f}: {v}")
    return "\n".join(user_texts)

def extract_image_urls(user_data, max_imgs=100):
    # fragments = user_data['data']['fragments']
    fragments = user_data['fragments']
    image_urls = [f['sourceUrl'] for f in fragments if f['type'] == 'Picture']
    return image_urls[:100]

def generate_caption_from_url(url):
    try:
        print(f"  下载图片: {url}")
        response = requests.get(url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"  图片加载完毕，正在生成描述...")
        inputs = processor(images=img, return_tensors="pt").to(blip_model.device)
        out = blip_model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        print(f"  图片描述: {caption}")
        return caption
    except Exception as e:
        print(f"  图片处理失败: {url}, 错误：{e}")
        return "无图像内容"

def call_glm4(user_id, texts, captions):
    print("  向GLM-4发送分析请求...")

    full_prompt = f"""
你是一个心理学专家，正在分析一位用户的公开内容（图像描述和文字）来构建一个结构化用户卡片。
请根据以下内容，输出一个严格的 JSON 对象，包含以下字段：
traits, writing_style, emotional_patterns, default_needs, attachment_style, profile_text, one_sentence_summary。

【文本内容】：
{texts}

【图像内容】：
{"；".join(captions)}

请用中文回答：
- 必须返回标准 JSON 对象，不能包含任何多余内容（如说明文字、前后缀、markdown代码块、```json 等）；
- 不允许添加注释（如 // 或 #）；
- 字段值如果信息不完整，请根据上下文合理推测填写，避免留空；
- `profile_text` 是一段简洁自然语言简介；
- `one_sentence_summary` 是一句极简总结，格式为“一个xxx的xxx”，不需要标点；

请严格按上述要求返回有效 JSON 对象。
"""

    # 智谱GLM-4的API参数
    glm_api_key = config["glm_api_key"]
    glm_api_url = config["glm_api_url"]

    headers = {
        "Authorization": f"Bearer {glm_api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "glm-4",
        "messages": [
            {"role": "user", "content": full_prompt}
        ],
        "temperature": 0
    }
    # res_json = response.json()
    # print("[DEBUG] GLM4返回内容：", res_json)
    response = requests.post(glm_api_url, headers=headers, json=payload, timeout=60)
    # print(f"GLM-4 API Response Text: {response.text}")
    import json as pyjson
    content = response.json()["choices"][0]["message"]["content"].strip()
    # content = response.choices[0].message.content.strip()
    try:
        # 只取大模型输出的第一段json对象（更保险）
        # print("原始内容：", content)
        start = content.find('{')
        end = content.rfind('}')+1
        json_str = content[start:end]
        result = pyjson.loads(json_str)
        print("  GLM4分析完成。")
        return result
    except Exception as e:
        print("  大模型输出解析失败:", e)
        print("  原始内容:", content)
        return {}

def multimodal_analyze_and_save(user_data):
    print("user_data", user_data)
    init_csv()
    # user = user_data['data']['user']
    user = user_data['user']
    user_id = user['userId']
    nick_name = user.get('nickName', "")
    print(f"\n正在处理用户: {user_id} (昵称: {nick_name})")
    texts = extract_text(user_data)
    # print("  文本内容已提取。")
    image_urls = extract_image_urls(user_data, max_imgs=100)
    captions = []
    for url in image_urls:
        captions.append(generate_caption_from_url(url))
    while len(captions) < 3:
        captions.append("无图像内容")
    print("  所有图片描述已生成，开始AI分析...")
    card = call_glm4(user_id, texts, captions)
    # print("card keys:", list(card.keys()))
    result_row = {
        "name": user_id,
        "nickName": nick_name,
        "traits": card.get("traits", ""),
        "writing_style": card.get("writing_style", ""),
        "emotional_patterns": card.get("emotional_patterns", ""),
        "default_needs": card.get("default_needs", ""),
        "attachment_style": card.get("attachment_style", ""),
        "profile_text": card.get("profile_text", ""),
        "one_sentence_summary": card.get("one_sentence_summary", "")
    }
    # print("写入行内容：", result_row)


    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
        print(existing_df.head())
    else:
        existing_df = pd.DataFrame(columns=[
            "name", "nickName", "traits", "writing_style",
            "emotional_patterns", "default_needs", "attachment_style",
            "profile_text", "one_sentence_summary"
        ])

    existing_df = existing_df[existing_df["name"] != user_id]
    updated_df = pd.concat([existing_df, pd.DataFrame([result_row])], ignore_index=True)
    updated_df.to_csv(CSV_PATH, index=False)

    # 追加保存
    # df = pd.DataFrame([result_row])
    # df.to_csv(CSV_PATH, mode='a', index=False, header=False)
    print(f"用户 {user_id} 处理完毕，已追加至 {CSV_PATH}")
