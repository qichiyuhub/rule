#!/bin/bash
REPO_OWNER="galendu"
REPO_NAME="private"
WORKFLOW_ID="sing-box.yml"
BRANCH="main"
# https://github.com/settings/personal-access-tokens
GITHUB_TOKEN="xx" # 请替换为你的GitHub PAT,github token,可以在Settings->Developer settings->Personal access tokens->Generate new token中生成
SUB_ADDR="xx" # 请替换为你的订阅地址
CONFIG="https://raw.githubusercontent.com/galendu/rule/refs/heads/master/config/singbox/config_tun.json" # 请替换为你的配置文件,也可以使用这个

# 触发GitHub Actions
curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/actions/workflows/$WORKFLOW_ID/dispatches \
  -d "{
	\"ref\": \"$BRANCH\",
	\"inputs\": {
		\"subscriptionAddress\": \"$SUB_ADDR\",
		\"stencilFile\": \"$CONFIG\"
	}
}"
# 等待流水线运行完成（需要根据实际需求添加延迟或轮询逻辑）
sleep 40
# 获取流水线制品
ARTIFACTS_URL=$(curl -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/repos/$REPO_OWNER/$REPO_NAME/actions/runs | jq -r '.workflow_runs[0].artifacts_url')

# 下载制品
curl -L -H "Authorization: token $GITHUB_TOKEN" $ARTIFACTS_URL | jq -r '.artifacts[].archive_download_url' | while read url; do
  curl -L -O -H "Authorization: token $GITHUB_TOKEN" "$url"
done

## 解压制品得到config.json
unzip zip
#./sing-box check -c config.json
#
### 启动sing-box
#./sing-box run -c config.json