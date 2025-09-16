#!/bin/bash
# 一键推送服务器代码到 GitHub dev 分支
# 用法：bash push_to_github_dev.sh

# === 配置 ===
REPO_URL="https://github.com/ZZUZSL1024/Y.git"  # 改成你的仓库地址
BRANCH_NAME="new_feature"

echo "=== 初始化 Git（如果还没初始化） ==="
if [ ! -d .git ]; then
    git init
    git remote add origin "$REPO_URL"
else
    git remote set-url origin "$REPO_URL"
fi

echo "=== 切换到 $BRANCH_NAME 分支 ==="
git checkout -B "$BRANCH_NAME"

echo "=== 添加并提交所有更改 ==="
git add .
git commit -m "服务器代码更新: $(date '+%Y-%m-%d %H:%M:%S')" || echo "⚠️ 没有文件变化，跳过提交"

echo "=== 强制推送到远程 $BRANCH_NAME 分支（覆盖旧代码） ==="
git push origin "$BRANCH_NAME" --force

echo "✅ 推送完成，GitHub dev 分支已更新为服务器版本"
