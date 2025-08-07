import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from go_parser import GoTransformParser
from typing import Tuple, List, Optional

# ==============================================================================
# 1. 配置中心 (CONFIGURATIONS)
# ==============================================================================

# 文件路径配置
DATA_FILE = 'smart_weight_data.csv' 
GO_FILE = 'transform.go'
MODEL_FILE = 'Model.bin'

# 特征变换配置
STD_SCALER_FEATURES = [
    'connect_time', 'latency', 'upload_mb', 'download_mb', 'duration_minutes', 
    'last_used_seconds', 'traffic_density'
]
ROBUST_SCALER_FEATURES = ['success', 'failure']

# LightGBM 模型参数
LGBM_PARAMS = {
    'objective': 'regression', 'metric': 'rmse', 'n_estimators': 1000,
    'learning_rate': 0.03, 'random_state': 42, 'n_jobs': -1, 'device': 'gpu'
}
EARLY_STOPPING_ROUNDS = 100


# ==============================================================================
# 2. 功能函数 (FUNCTIONS)
# ==============================================================================

def load_and_clean_data(file_path: str) -> Optional[pd.DataFrame]:
    """从CSV文件加载数据并进行清洗。如果失败则返回None。"""
    print(f"--> 正在加载数据: {file_path}")
    try:
        data = pd.read_csv(file_path)
        print(f"    原始数据加载成功，共 {len(data)} 条。")
    except FileNotFoundError:
        print(f"    错误: 数据文件 '{file_path}' 未找到!")
        return None

    data.dropna(subset=['weight'], inplace=True)
    data = data[data['weight'] > 0].copy()
    print(f"    清洗后剩余 {len(data)} 条有效记录。")
    return data

# 这是基于你的预处理数据文件的新版提取函数
def extract_features_from_preprocessed(data: pd.DataFrame, feature_order: List[str]) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """从已经是特征格式的CSV中提取X和y。"""
    print("--> 正在从预处理数据中提取特征 (X) 和目标 (y)...")
    try:
        X = data[feature_order]
        y = data['weight']
        print("    成功提取特征和目标。")
        return X, y
    except KeyError as e:
        print(f"    错误: 数据文件中缺少必要的特征列: {e}")
        return None, None

def apply_feature_transforms(X: pd.DataFrame, feature_order: List[str]) -> Tuple[pd.DataFrame, StandardScaler, RobustScaler]:
    """对特征矩阵应用 StandardScaler 和 RobustScaler。"""
    print("--> 正在进行特征变换...")
    X_scaled = X.copy()
    
    std_scaler = StandardScaler()
    X_scaled[STD_SCALER_FEATURES] = std_scaler.fit_transform(X_scaled[STD_SCALER_FEATURES])
    print(f"    已应用 StandardScaler 到 {len(STD_SCALER_FEATURES)} 个特征。")

    robust_scaler = RobustScaler()
    X_scaled[ROBUST_SCALER_FEATURES] = robust_scaler.fit_transform(X_scaled[ROBUST_SCALER_FEATURES])
    print(f"    已应用 RobustScaler 到 {len(ROBUST_SCALER_FEATURES)} 个特征。")
    
    return X_scaled, std_scaler, robust_scaler

def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> lgb.Booster:
    """训练 LightGBM 模型。"""
    print("--> 正在训练 LightGBM 模型...")
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True)]
    )
    return model

def save_model_and_config(model: lgb.Booster, std_scaler: StandardScaler, robust_scaler: RobustScaler, feature_order: List[str]):
    """保存模型为文本格式，并附加变换配置。"""
    print("--> 正在保存模型及配置...")
    
    model.save_model(MODEL_FILE, num_iteration=model.best_iteration)
    print(f"    模型主体已保存为文本格式到: {MODEL_FILE}")

    order_block = "[order]\n" + "".join([f"{i}={name}\n" for i, name in enumerate(feature_order)]) + "[/order]\n"
    
    std_indices = [feature_order.index(f) for f in STD_SCALER_FEATURES]
    robust_indices = [feature_order.index(f) for f in ROBUST_SCALER_FEATURES]
    
    definitions_block = "[definitions]\n"
    definitions_block += f"std_type=StandardScaler\nstd_features={','.join(map(str, std_indices))}\nstd_mean={','.join(map(str, std_scaler.mean_))}\nstd_scale={','.join(map(str, std_scaler.scale_))}\n\n"
    definitions_block += f"robust_type=RobustScaler\nrobust_features={','.join(map(str, robust_indices))}\nrobust_center={','.join(map(str, robust_scaler.center_))}\nrobust_scale={','.join(map(str, robust_scaler.scale_))}\n"
    definitions_block += "[/definitions]\n"
    
    transformed_indices = set(std_indices + robust_indices)
    untransformed_list = [f"{i}:{name}" for i, name in enumerate(feature_order) if i not in transformed_indices]

    final_transforms_block = (
        "\n\nend of trees\n\n"
        f"[transforms]\n{order_block}{definitions_block}untransformed_features={','.join(untransformed_list)}\ntransform=true\n[/transforms]\n"
    )
    
    with open(MODEL_FILE, 'a', encoding='utf-8') as f:
        f.write(final_transforms_block)
    print("    变换配置已成功附加到模型文件末尾。")

# ==============================================================================
# 3. 主执行流程 (MAIN EXECUTION)
# ==============================================================================

def main():
    """主函数，按顺序执行所有步骤。"""
    print("--- Mihomo 模型训练开始 ---")
    
    try:
        parser = GoTransformParser(GO_FILE)
        feature_order = parser.get_feature_order()
    except Exception as e:
        print(f"初始化失败: {e}")
        return
        
    full_data = load_and_clean_data(DATA_FILE)
    if full_data is None:
        return

    result = extract_features_from_preprocessed(full_data, feature_order)
    if result is None:
        return
    X, y = result

    X_scaled, std_scaler, robust_scaler = apply_feature_transforms(X, feature_order)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train, X_test, y_test)
    
    save_model_and_config(model, std_scaler, robust_scaler, feature_order)
    
    print("\n🎉 --- 训练全部完成 --- 🎉")
    print(f"最终模型 '{MODEL_FILE}' 已生成，随时可以部署！")

if __name__ == "__main__":
    main()