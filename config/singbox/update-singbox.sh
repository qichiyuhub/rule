#!/bin/sh
curl -L "http://192.168.10.12:5000/config/订阅地址&file=https://raw.githubusercontent.com/qichiyuhub/rule/refs/heads/master/config/singbox/config.json" -o /etc/sing-box/config.json
killall sing-box || echo "No sing-box process found."
sleep 3
sing-box run -c /etc/sing-box/config.json &