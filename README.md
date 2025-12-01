# X-ray 肺炎影像分類系統
<img width="1495" height="870" alt="image" src="https://github.com/user-attachments/assets/a260212a-0d05-40d7-8e76-3dceb3559989" />

## 專案簡介

本專案基於深度學習的肺炎檢測系統，能夠自動分析胸部 X 光影像並判斷是否患有肺炎。

### 主要功能

- 二元分類：NORMAL vs PNEUMONIA
- CNN 深度學習模型
- 數據增強 (Data Augmentation)
- 模型評估與視覺化
- 單張影像預測

### 技術特點

- 深度學習框架: TensorFlow/Keras
- 優化器: Adam Optimizer
- 數據增強: 旋轉、平移、縮放
- 評估指標: Accuracy, Precision, Recall, F1-Score

---

## 資料集

### 資料結構

```
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 資料統計

| 類別 | 訓練集 | 測試集 | 驗證集 |
|------|--------|--------|--------|
| NORMAL | 1,341 | 234 | 8 |
| PNEUMONIA | 3,875 | 390 | 8 |
| **總計** | **5,216** | **624** | **16** |

---

## 環境需求

### Python 版本
- Python 3.8+

### 必要套件

```
tensorflow>=2.10.0
numpy>=1.21.0
opencv-python>=4.5.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

---

## 安裝步驟

### 1. Clone 專案

```bash
git clone https://github.com/你的用戶名/chest-xray-pneumonia.git
cd chest-xray-pneumonia
```

### 2. 建立虛擬環境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. 安裝依賴套件

```bash
pip install -r requirements.txt
```

---

## 使用方法

### 訓練模型

```bash
python train_model.py
```

可調整參數:
- `--epochs`: 訓練輪數 (預設 30)
- `--batch_size`: 批次大小 (預設 64)
- `--learning_rate`: 學習率 (預設 0.001)

### 評估模型

```bash
python evaluate_model.py
```

輸出結果:
- 分類報告 (Precision, Recall, F1-Score)
- 混淆矩陣
- 準確率曲線
- 損失曲線

### 單張影像預測

```bash
python predict.py --image path/to/xray_image.jpg
```

---

## 模型架構

### CNN 架構摘要

```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Conv2D (64 filters)         (None, 32, 32, 64)        1,792     
BatchNormalization          (None, 32, 32, 64)        256       
MaxPooling2D (2x2)          (None, 16, 16, 64)        0         
_________________________________________________________________
Conv2D (128 filters)        (None, 16, 16, 128)       73,856    
BatchNormalization          (None, 16, 16, 128)       512       
MaxPooling2D (2x2)          (None, 8, 8, 128)         0         
_________________________________________________________________
Conv2D (256 filters)        (None, 8, 8, 256)         295,168   
BatchNormalization          (None, 8, 8, 256)         1,024     
MaxPooling2D (2x2)          (None, 4, 4, 256)         0         
_________________________________________________________________
Flatten                     (None, 4096)              0         
_________________________________________________________________
Dense (500 neurons)         (None, 500)               2,048,500 
BatchNormalization          (None, 500)               2,000     
Dense (100 neurons)         (None, 100)               50,100    
BatchNormalization          (None, 100)               400       
Dense (2 neurons)           (None, 2)                 202       
=================================================================
Total params: 2,473,810
Trainable params: 2,471,714
Non-trainable params: 2,096
```

### 訓練配置

| 項目 | 設定 |
|------|------|
| 優化器 | Adam (lr=0.001) |
| 損失函數 | Categorical Crossentropy |
| 評估指標 | Accuracy |
| 訓練輪數 | 30 epochs |
| 批次大小 | 64 |

### 數據增強

- 隨機旋轉: ±25°
- 水平平移: ±3 pixels
- 垂直平移: ±3 pixels
- 隨機縮放: 0.3

---

## 訓練結果

### 模型表現

| 指標 | 訓練集 | 驗證集 | 測試集 |
|------|--------|--------|--------|
| Accuracy | 95.2% | 92.8% | 91.5% |

### 分類報告

```
              precision    recall  f1-score   support

   PNEUMONIA       0.98      0.98      0.98       203
      NORMAL       0.93      0.91      0.92        58

    accuracy                           0.97       261
   macro avg       0.95      0.95      0.95       261
weighted avg       0.97      0.97      0.97       261
```

### 混淆矩陣

|  | 預測 PNEUMONIA | 預測 NORMAL |
|---|----------------|-------------|
| **實際 PNEUMONIA** | 199 | 4 |
| **實際 NORMAL** | 5 | 53 |

### 詳細分析

**PNEUMONIA (肺炎) 類別**
- Precision: 0.98 (98%)
- Recall: 0.98 (98%)
- F1-Score: 0.98
- Support: 203 張

**NORMAL (正常) 類別**
- Precision: 0.93 (93%)
- Recall: 0.91 (91%)
- F1-Score: 0.92
- Support: 58 張

**模型準確率**: 97% (253/261 正確預測)
---

## 專案結構

```
X_ray/
├── chest_xray/                           # 資料集主目錄
│   ├── test/                             # 測試集
│   │   ├── NORMAL/                       # 正常影像
│   │   └── PNEUMONIA/                    # 肺炎影像
│   ├── train/                            # 訓練集
│   │   ├── NORMAL/                       # 正常影像
│   │   └── PNEUMONIA/                    # 肺炎影像
│   └── val/                              # 驗證集
│       ├── NORMAL/                       # 正常影像
│       └── PNEUMONIA/                    # 肺炎影像
│
├── 03_ImageDataGenerator_CNN_formFiles.py     # 基礎 CNN 訓練腳本
├── 04_ImageDataGenerator_CNN_formFiles_crop.py # 加入裁切的 CNN 訓練腳本
│
├── best_model.keras                      # 最佳模型 (Keras 格式)
├── my_model.h5                           # 訓練模型 (HDF5 格式)
├── my_model.keras                        # 訓練模型 (Keras 格式)
├── my_model.weights.h5                   # 模型權重
├── prediction_result.jpg                 # 預測結果圖片
│
├── LICENSE                               # 授權文件
└── README.md                             # 專案說明文件
```
模型文件

best_model.keras: 訓練過程中準確率最高的模型
my_model.h5: 完整訓練模型 (HDF5 格式，舊版)
my_model.keras: 完整訓練模型 (Keras 原生格式)
my_model.weights.h5: 僅包含模型權重

---

## 使用範例

### Python 預測腳本

```python
import cv2
import numpy as np
import tensorflow as tf
from google.colab.patches import cv2_imshow

# 載入模型
model = tf.keras.models.load_model('my_model.h5', custom_objects={'softmax_v2': tf.nn.softmax})

# 類別名稱（按字母順序）
dirs = ['NORMAL', 'PNEUMONIA']

# 使用確認存在的圖片路徑
test_image_path = 'test/PNEUMONIA/person111_bacteria_536.jpeg'
print(f"使用測試圖片: {test_image_path}")

img_original = cv2.imread(test_image_path)


# 前處理（和訓練時一樣）
w, h = 32, 32  # ⭐ 改成 32x32，和訓練時一致！
w2 = img_original.shape[1]
h2 = img_original.shape[0]

# 裁切成正方形
min_dim = min(w2, h2)
start_x = (w2 - min_dim) // 2
start_y = (h2 - min_dim) // 2
img = img_original[start_y:start_y + min_dim, start_x:start_x + min_dim]

# 調整大小
img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 正規化
img_normalized = img_rgb.astype('float32') / 255.0
img_input = img_normalized.reshape(1, w, h, 3)

# 預測
predict = model.predict(img_input)
i = np.argmax(predict[0])
confidence = predict[0][i]

print(f"\n 預測結果: {dirs[i]}")
print(f" 信心度: {confidence:.4f}")
print(f"所有類別機率: NORMAL={predict[0][0]:.4f}, PNEUMONIA={predict[0][1]:.4f}")

# 顯示圖片
img_display = cv2.resize(img_rgb, (400, 400))
img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
str1 = f"{dirs[i]} ({confidence:.2%})"
img_display = cv2.putText(img_display, str1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2_imshow(img_display)
```

---

## 評估指標說明

### Precision (精確率)
在所有預測為肺炎的病例中，真正患有肺炎的比例。

```
Precision = TP / (TP + FP)
```

### Recall (召回率)
在所有實際肺炎病例中，模型正確識別出的比例。

```
Recall = TP / (TP + FN)
```

### F1-Score
精確率和召回率的調和平均數。

```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 模型改進建議

### 1. 使用遷移學習
- 採用預訓練模型: VGG16, ResNet50, DenseNet  (試過效果不好)
- 凍結前層，微調後層

### 2. 增加訓練數據
- 擴充數據集
- 使用 GAN 生成合成影像
- 跨資料集訓練

### 3. 處理類別不平衡
- 使用加權損失函數
- 過採樣少數類別
- 欠採樣多數類別

### 4. 優化模型架構
- 增加 Dropout 層防止過擬合 (我直接刪掉)
- 使用 Learning Rate Scheduler 
- 實驗不同的優化器

---

## 參考資料

### 資料集來源
- [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download)

---

## 授權

MIT License

---

## 作者

fayr

---

**最後更新**: 2024-12-02


</div>
