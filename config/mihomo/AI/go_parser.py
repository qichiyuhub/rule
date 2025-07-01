# go_parser.py
import re

class GoTransformParser:
    def __init__(self, go_file_path):
        try:
            with open(go_file_path, 'r', encoding='utf-8') as f:
                self.content = f.read()
            print(f"成功读取 Go 文件: {go_file_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"错误: Go 文件 '{go_file_path}' 未找到！请确保它和脚本在同一目录。")
        
        self.feature_order = self._parse_feature_order()

    def _parse_feature_order(self):
        print("正在解析 getDefaultFeatureOrder...")
        match = re.search(r'func getDefaultFeatureOrder\(\) map\[int\]string \{\s*return map\[int\]string\{(.*?)\}\s*\}', self.content, re.DOTALL)
        if not match:
            print("警告: 未能找到 getDefaultFeatureOrder 函数，将使用硬编码的备用顺序。")
            return self._get_fallback_feature_order()
        body = match.group(1)
        features = re.findall(r'(\d+):\s*"([^"]+)"', body)
        if not features:
            print("警告: 在 getDefaultFeatureOrder 函数中未找到特征，将使用备用顺序。")
            return self._get_fallback_feature_order()
        feature_dict = {int(idx): name for idx, name in features}
        sorted_features = [feature_dict[i] for i in sorted(feature_dict.keys())]
        print(f"成功解析出 {len(sorted_features)} 个特征顺序。")
        return sorted_features

    def get_feature_order(self):
        return self.feature_order

    def _get_fallback_feature_order(self):
        return [
            'success', 'failure', 'connect_time', 'latency', 'upload_mb', 'download_mb', 
            'duration_minutes', 'last_used_seconds', 'is_udp', 'is_tcp', 'asn_feature', 
            'country_feature', 'address_feature', 'port_feature', 'traffic_ratio', 
            'traffic_density', 'connection_type_feature', 'asn_hash', 'host_hash', 
            'ip_hash', 'geoip_hash'
        ]