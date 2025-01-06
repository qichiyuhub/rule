
## 

```bash
{
  "log": {
    "disabled": false, 
    "level": "info", #日志级别,支持 trace debug info warn error fatal panic
    "output": "box.log",
    "timestamp": true
  },
  "dns": {
    "servers": [
        "tag": "", # DNS 服务器的标签
        "address": "", 
        "address_resolver": "", # 用于解析本 DNS 服务器的域名的另一个 DNS 服务器的标签
        "address_strategy": "", # 用于解析本 DNS 服务器的域名的策略
        "strategy": "", #默认解析策略
        "detour": "", #用于连接到 DNS 服务器的出站的标签
        "client_subnet": ""
    ],
    "rules": [],
    "final": "", #默认dns服务器的标签,默认使用第一个服务器。
    "strategy": "", #可选值: prefer_ipv4 prefer_ipv6 ipv4_only ipv6_only
    "disable_cache": false,
    "disable_expire": false,
    "independent_cache": false,
    "cache_capacity": 0,
    "reverse_mapping": false,
    "client_subnet": "",
    "fakeip": { #虚假ip地址段
        "enabled": true,
        "inet4_range": "198.18.0.0/15",
        "inet6_range": "fc00::/18"
    }
  },
  "endpoints": [],
  "inbounds": [],
  "outbounds": [],
  "route": {},
  "experimental": {}
}
```

**level**
日志等级，可选值：trace debug info warn error fatal panic。  

