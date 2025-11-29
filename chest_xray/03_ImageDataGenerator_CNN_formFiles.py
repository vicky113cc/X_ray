#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "柯博文老師 Powen Ko, www.powenko.com"


import tensorflow as tf
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import os.path as path
import os
import cv2
from tensorflow.keras.activations import softmax

IMAGEPATH = 'chest_xray/train/'                 #  圖片資料夾路徑
dirs = os.listdir(IMAGEPATH)         #  找所有的檔案
X=[]
Y=[]
print(dirs)
w=32 # 224                            # 要訓練時的圖片大小
h=32 # 224
c=3                                   # 顏色數 RGB 3  灰階 1
i=0                                   # 類別編號
for name in dirs:                     #  再往下讀取每個資料夾
    file_paths = glob.glob(path.join(IMAGEPATH+"/"+name, '*.*'))   # 取得該文件內的所有檔案名稱

    ## 判斷是否是圖片檔案         
    for path3 in file_paths:
        if not path3.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # 支援多種圖片格式 
            continue
        img = cv2.imread(path3)                                # 讀取檔案
        if img is None:
            print(f"[WARN] Failed to load image: {path3}")     # 錯誤處理
            continue
        w2=img.shape[1]  # 取得圖片寬度
        h2=img.shape[0]  # 取得圖片高度
        # 依照原讀等比縮小 然後把多的切掉
        if w2>h2:
            scale=w/w2
        else:
            scale=h/h2
        new_w=int(w2*scale)
        new_h=int(h2*scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)  # 調整影像大小

        # 以中間為單位進行裁切
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        img = img[start_y:start_y + h, start_x:start_x + w]
        # im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # 轉圖檔 BGR  轉 RGB
        if c==1:                                               # 如果是灰階
            #im_rgb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)  # 轉灰階
            img = img.reshape(w,h,1)                     # 調整形狀
        print(f"path: {name} Loaded image: {path3} {img.shape}")  # 調試訊息    
        X.append(img)            # 放入X資料集
        Y.append(i)                 # 放入Y資料集
    i=i+1                           # 下一個文件夾 類別編號遞增

X = np.asarray(X)                   # 轉為 numpy array　　
Y = np.asarray(Y)

#  資料前處理 均一化 與 形狀轉換
X = X.astype('float32')
X=X/255
X=X.reshape(X.shape[0],w,h,3)    # 張 高 寬 顏色數


category=len(dirs)               # 有個分類
dim=X.shape[1]                   # 圖片大小

# 分割訓練與測試資料
x_train , x_test , y_train , y_test = train_test_split(X,Y,test_size=0.05)  # 把資料打亂
# 將數字轉為 One-hot 向量
y_train2 = tf.keras.utils.to_categorical(y_train, category)
y_test2  = tf.keras.utils.to_categorical(y_test, category)


# 載入資料（將資料打散，放入 train 與 test 資料集）
# 圖片產生器
print(x_train.shape)
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                            rotation_range=25 ,            # 隨機旋轉角度
                            width_shift_range=[-3,3],      # 水平平移
                            height_shift_range=[-3,3] ,    # 垂直平移
                            zoom_range=0.3 ,               # 隨機縮放範圍
							data_format='channels_last')   # 圖片格式為 (張,高,寬,顏色數)

# 建立模型
model = tf.keras.models.Sequential()
# 加入 2D 的 Convolution Layer，接著一層 ReLU 的 Activation 函數
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                 padding="same",
                 activation='relu',
                 input_shape=(w,h,c)))

model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(500, activation='relu'))
model.add(tf.keras.layers.Dense(250, activation='relu'))
model.add(tf.keras.layers.Dense(100))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))

learning_rate = 0.001   # 學習率
opt1 = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 優化器
model.compile(
    optimizer=opt1,
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])

model.summary()        # 顯示模型摘要



# 產生訓練圖片
trainData=datagen.flow(x_train,y_train2,batch_size=64)  # 批次大小 64 原本的一張圖片變成64張

history = model.fit(trainData,
                   epochs=20 #,
                   # callbacks=[checkpoint]
                   )


# 保存模型
#  tf.keras.models.save('model.h5')
model.save('my_model.h5')
model.save('my_model.keras')
##### 載入模型
# model = tf.keras.models.load_model('model.h5')  # <---
model = tf.keras.models.load_model('my_model.h5', custom_objects={'softmax_v2': softmax})

#保存模型權重
model.save_weights("my_model.weights.h5")
# 讀取模型權重
model.load_weights("my_model.weights.h5")


#測試
score = model.evaluate(x_test, y_test2, batch_size=128)
# 輸出結果
print("score:",score)

predict = model.predict(x_test)
print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))

predict2 = np.argmax(predict, axis=1)
print("predict_classes:",predict2)
print("y_test",y_test[:])
for t1 in predict2:
    print(dirs[t1])

img=x_test[0]
img=img.reshape(w,h,3)
img=img*255
img = img.astype('uint8')
img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
im_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
i = np.argmax(predict[0])
str1 = dirs[i] + "   " + str(predict[0][i])
print(str1)
im_bgr = cv2.putText(im_bgr, str1, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 1, cv2.LINE_AA)
cv2.imshow('image', im_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()