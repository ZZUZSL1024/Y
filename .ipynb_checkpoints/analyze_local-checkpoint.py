# -*- coding: utf-8 -*-
import base64, mimetypes, os, sys, json, time
from multimodal_preprocessing_service import MultimodalPreprocessingService

def file_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"

def main():
    if len(sys.argv) < 2:
        print("用法: python analyze_local.py /path/to/image.jpg")
        sys.exit(1)

    img_path = sys.argv[1]
    assert os.path.exists(img_path), f"文件不存在: {img_path}"

    svc = MultimodalPreprocessingService()  # 复用你已有的 GLM 封装 & prompt
    data_url = file_to_data_url(img_path)

    t0 = time.time()
    out = svc._call_glm("frag-local", data_url, "picture")
    dt = (time.time() - t0) * 1000
    if not out:
        print("分析失败")
        sys.exit(2)
    print(f"[OK] cost={dt:.1f}ms")
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
