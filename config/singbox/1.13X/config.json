{
  "dns": {
    "servers": [
      {"tag": "local", "type": "local", "prefer_go": true},
      {"tag": "ali", "type": "https", "server": "223.5.5.5"},
      {"tag": "google", "type": "https", "server": "8.8.8.8", "detour": "默认代理"},
      {"tag": "fakeip", "type": "fakeip", "inet4_range": "28.0.0.1/8", "inet6_range": "2001:db8::/32"}
    ],
    "rules": [
      {"query_type": ["HTTPS", "SVCB"], "action": "reject"},
      {"clash_mode": "Direct", "server": "ali"},
      {"clash_mode": "Global", "server": "fakeip"},
      {"rule_set": "geosite-fakeipfilter", "server": "ali"},
      {"query_type": ["A", "AAAA"], "server": "fakeip", "rewrite_ttl": 1},
      {"rule_set": "geosite-cn","server": "ali"}
    ],
    "final": "google",
    "client_subnet": "223.5.5.0/24",
    "strategy": "prefer_ipv4",
    "independent_cache": true,
    "cache_capacity": 8192,
    "reverse_mapping": true
  },
  "outbounds": [
    {"tag": "默认代理", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "AI", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "YouTube", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "Google", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "Github", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "Telegram", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "TikTok", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "Netflix", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "PayPal", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "Steam", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "Microsoft", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "OneDrive", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择"]},
    {"tag": "Apple", "type": "selector", "outbounds": ["日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择", "直连"]},
    {"tag": "漏网之鱼", "type": "selector", "outbounds": ["默认代理", "日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择", "直连"]},
    {"tag": "日本手动", "type": "selector", "outbounds": []},
    {"tag": "狮城手动", "type": "selector", "outbounds": []},
    {"tag": "香港手动", "type": "selector", "outbounds": []},
    {"tag": "美国手动", "type": "selector", "outbounds": []},
    {"tag": "手动选择", "type": "selector", "outbounds": []},
    {"tag": "自动选择", "type": "urltest", "outbounds": [], "url": "https://clients1.google.com/generate_204", "interval": "30m", "tolerance": 100},
    {"tag": "延迟辅助", "type": "urltest", "outbounds": ["直连"], "url": "http://connect.rom.miui.com/generate_204", "interval": "30m"},
    {"tag": "GLOBAL", "type": "selector", "outbounds": ["默认代理", "AI", "YouTube", "Google", "Github", "Telegram", "TikTok", "Netflix", "PayPal", "Steam", "Microsoft", "OneDrive", "Apple", "漏网之鱼", "日本手动", "狮城手动", "香港手动", "美国手动", "手动选择", "自动选择", "延迟辅助", "直连"]},
    {"tag": "直连", "type": "direct", "domain_resolver": "ali"}
  ],
  "route": {
    "rules": [
      {"type": "logical", "mode": "or", "rules": [{"domain": "Mijia Cloud"}, {"domain_suffix": "push.apple.com"}, {"rule_set": "geoip-telegram"}], "invert": true, "action": "sniff", "sniffer": ["http", "tls", "quic", "dns"], "timeout": "500ms"},
      {"type": "logical", "mode": "or", "rules": [{"port": 53}, {"protocol": "dns"}], "action": "hijack-dns"},
      {"type": "logical", "mode": "or", "rules": [{"port": 853}, {"protocol": "stun"}], "action": "reject"},
      {"type": "logical", "mode": "and", "rules": [{"protocol": "quic"}, {"rule_set": "geosite-!cn"}], "action": "reject"},
      {"network": "icmp", "outbound": "直连"},
      {"ip_is_private": true, "outbound": "直连"},
      {"clash_mode": "Direct", "outbound": "直连"},
      {"clash_mode": "Global", "outbound": "GLOBAL"},
      {"rule_set": "geosite-ai", "outbound": "AI"},
      {"rule_set": "geosite-youtube", "outbound": "YouTube"},
      {"rule_set": "geosite-google", "outbound": "Google"},
      {"rule_set": "geosite-github", "outbound": "Github"},
      {"rule_set": "geosite-onedrive", "outbound": "OneDrive"},
      {"rule_set": "geosite-microsoft", "outbound": "Microsoft"},
      {"rule_set": "geosite-apple", "outbound": "Apple"},
      {"rule_set": "geosite-telegram", "outbound": "Telegram"},
      {"rule_set": "geosite-tiktok", "outbound": "TikTok"},
      {"rule_set": "geosite-netflix", "outbound": "Netflix"},
      {"rule_set": "geosite-paypal", "outbound": "PayPal"},
      {"rule_set": "geosite-usd", "outbound": "PayPal"},
      {"rule_set": "geosite-steamcn", "outbound": "直连"},
      {"rule_set": "geosite-steam", "outbound": "Steam"},
      {"rule_set": "geosite-!cn", "outbound": "默认代理"},
      {"action": "resolve"},
      {"rule_set": "geoip-google", "outbound": "Google"},
      {"rule_set": "geoip-apple", "outbound": "Apple"},
      {"rule_set": "geoip-telegram", "outbound": "Telegram"},
      {"rule_set": "geoip-netflix", "outbound": "Netflix"},
      {"rule_set": "geoip-cn", "outbound": "直连"}
    ],
    "rule_set": [
      {"tag": "geosite-fakeipfilter", "type": "remote", "format": "source", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/qichiyuhub/rule/refs/heads/main/rules/fakeipfilter.json", "download_detour": "直连"},
      {"tag": "geosite-ai", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/category-ai-!cn.srs", "download_detour": "直连"},
      {"tag": "geosite-youtube", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/youtube.srs", "download_detour": "直连"},
      {"tag": "geosite-google", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/google.srs", "download_detour": "直连"},
      {"tag": "geosite-github", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/github.srs", "download_detour": "直连"},
      {"tag": "geosite-onedrive", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/onedrive.srs", "download_detour": "直连"},
      {"tag": "geosite-microsoft", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/microsoft.srs", "download_detour": "直连"},
      {"tag": "geosite-apple", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/apple.srs", "download_detour": "直连"},
      {"tag": "geosite-telegram", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/telegram.srs", "download_detour": "直连"},
      {"tag": "geosite-tiktok", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/tiktok.srs", "download_detour": "直连"},
      {"tag": "geosite-netflix", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/netflix.srs", "download_detour": "直连"},
      {"tag": "geosite-paypal", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/paypal.srs", "download_detour": "直连"},
      {"tag": "geosite-usd", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/category-cryptocurrency.srs", "download_detour": "直连"},
      {"tag": "geosite-steamcn", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/steam@cn.srs", "download_detour": "直连"},
      {"tag": "geosite-steam", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/steam.srs", "download_detour": "直连"},
      {"tag": "geosite-!cn", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/geolocation-!cn.srs", "download_detour": "直连"},
      {"tag": "geosite-cn", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geosite/cn.srs", "download_detour": "直连"},
      {"tag": "geoip-google", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geoip/google.srs", "download_detour": "直连"},
      {"tag": "geoip-apple", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo-lite/geoip/apple.srs", "download_detour": "直连"},
      {"tag": "geoip-telegram", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geoip/telegram.srs", "download_detour": "直连"},
      {"tag": "geoip-netflix", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geoip/netflix.srs", "download_detour": "直连"},
      {"tag": "geoip-cn", "type": "remote", "format": "binary", "url": "https://gh-proxy.com/https://raw.githubusercontent.com/OneOhCloud/one-geoip/rule-set/one-china.srs", "download_detour": "直连"}
    ],
    "final": "漏网之鱼",
    "auto_detect_interface": true,
    "default_domain_resolver": {"server": "ali"}
  },
  "inbounds": [
    {
      "tag": "tun-in",
      "type": "tun",
      "address": [
        "172.19.0.1/30",
        "fdfe:dcba:9876::1/126"
      ],
      "stack": "mixed",
      "auto_route": true,
      "auto_redirect": true
    },
    {
      "tag": "mixed-in",
      "type": "mixed",
      "listen": "0.0.0.0",
      "listen_port": 7890
    }
  ],
  "experimental": {
    "cache_file": {
      "enabled": true,
      "path": "/etc/sing-box/cache.db",
      "store_fakeip": true
    },
    "clash_api": {
      "external_controller": "0.0.0.0:9095",
      "external_ui": "/etc/sing-box/ui",
      "external_ui_download_url": "https://gh-proxy.com/https://github.com/Zephyruso/zashboard/archive/refs/heads/gh-pages.zip",
      "external_ui_download_detour": "直连",
      "secret": "",
      "default_mode": "Rule"
    }
  },
  "certificate": {
    "store": "chrome"
  },
  "ntp": {
    "enabled": true,
    "server": "time.apple.com"
  },
  "log": {
    "disabled": false,
    "level": "debug",
    "timestamp": true
  }
}
