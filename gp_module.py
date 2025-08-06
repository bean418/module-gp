# imports
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import gpflow as gp
# import gpsig
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

# Set up environment
sys.path.append('..')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.logging.set_verbosity(tf.logging.ERROR)

# Classes
class GPSigTrainer:
    def __init__(self, len_ex: int, n_feat: int, num_levels: int = 4, num_inducing: int = 200):
        self.len_ex = len_ex
        self.n_feat = n_feat
        self.num_levels = num_levels
        self.num_inducing = num_inducing

class Preprocessor:
    @staticmethod
    def load_data(directory_path: str) -> list:
        df_list = []
        print(f"'{directory_path}' 디렉토리에서 CSV 파일을 탐색합니다.")

        if not os.path.isdir(directory_path):
            print(f"Error: '{directory_path}'is not a valid directory.")
            return df_list

        for file_name in os.listdir(directory_path):
            if file_name.endswith('.csv'):
                full_path = os.path.join(directory_path, file_name)
                
                try:
                    df = pd.read_csv(full_path)
                    df_list.append(df)
                    print(f" - '{file_name}' 파일을 DataFrame으로 변환하여 리스트에 추가했습니다.")
                except Exception as e:
                    print(f" - 오류: '{file_name}' 파일을 읽는 중 문제가 발생했습니다: {e}")

        if not df_list:
            print("디렉토리에서 CSV 파일을 찾지 못했습니다.")
            
        return df_list

    @staticmethod
    def drop_col(df_list : list) -> list:
        """
        현재 drug 데이터에 맞추어져 있기 때문에
        하드코딩 되어있음.
        -> TVOC와 SHT 40 제거
        """
        for df in df_list:
            print(f"현재 데이터프레임의 컬럼: {df.columns.tolist()}")
            if 'TVOC' in df.columns:
                print("TVOC을 제거.")
                df.drop(labels=["Date", "Time", "TVOC"], axis=1, inplace=True)
            else:
                print("TVOC이 없습니다. SHT 40을 제거.")
                df.drop(labels=["Date", "Time", "SHT 40"], axis=1, inplace=True)
        for df in df_list:
            print(f"최종 데이터프레임의 컬럼: {df.columns.tolist()}")

        return df_list
    
    @staticmethod
    def detection_with_CUSUM(df_list: list, phase1: int = 150, threshold: float = 0.5, k: float = 2) -> list:
        pass
    
    @staticmethod
    def add_noise(df_list: list, noise_level: float = 0.05) -> list:
        dfs_noised = []
        print(f"노이즈 레벨: {noise_level}")
        for df in df_list:
            print(f"현재 데이터의 타입: {type(df)}")
            
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            df_noised = df.copy()

            col_stds = df_noised[numeric_cols].std(ddof=0)  # 모집단 표준편차 기준 :contentReference[oaicite:1]{index=1}
            print(f"Numeric 컬럼: {numeric_cols}")
            print(f"표준편차:\n{col_stds}")

            for col in numeric_cols:
                sd = col_stds[col]
                scale = sd * noise_level
                if scale == 0 or pd.isna(scale):
                    print(f"  - {col}: sd 0 또는 NaN → 노이즈 없음")
                    continue

                noise = np.random.normal(loc=0.0, scale=scale, size=df_noised.shape[0])
                df_noised[col] = df_noised[col] + noise
                print(f"  - {col}: sd={sd:.4g}, noise_scale={scale:.4g}")

            print(f"Noise 적용후 head:\n{df_noised.head()}")
            dfs_noised.append(df_noised)

        return dfs_noised

# main execution
dfs = Preprocessor.load_data('./data_drug')
dfs = Preprocessor.drop_col(dfs)
dfs_noise = Preprocessor.add_noise(dfs, noise_level=0.01)