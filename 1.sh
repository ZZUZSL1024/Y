# ========= 变量区域 =========
GIT_USER="ZZUZSL1024"
GIT_EMAIL="zhangshilin.lijin@gmail.com"
GIT_REPO="https://github.com/BIXING-CODE/Bixing_MQ_GPU.git"  # 仓库地址
MAIN_BRANCH="main"
DEV_BRANCH="dev"
FEATURE_BRANCH="feature/first-sync"
COMMIT_MSG="feat: 服务器端最新代码同步到GitHub主干"
# ========= 变量区域 =========

# 1. 配置git信息
git config --global user.name "$GIT_USER"
git config --global user.email "$GIT_EMAIL"

# 2. 初始化git仓库（如果还没有）
if [ ! -d .git ]; then
    git init
fi

# 3. 设置远程origin
git remote remove origin 2>/dev/null
git remote add origin $GIT_REPO

# 4. 确保本地main分支
git fetch origin
git checkout $MAIN_BRANCH 2>/dev/null || git checkout -b $MAIN_BRANCH

# 5. 将所有代码提交到main
git add .
git commit -m "$COMMIT_MSG"

# 6. 强制推送到远程main分支（以服务器为准！）
git push origin $MAIN_BRANCH --force

# 7. 创建/切换dev分支并推送
git checkout -b $DEV_BRANCH 2>/dev/null || git checkout $DEV_BRANCH
git push origin $DEV_BRANCH --force

echo "🚀 代码已全部推送到 main 和 dev 分支！"
echo "===> GitHub仓库: $GIT_REPO"
