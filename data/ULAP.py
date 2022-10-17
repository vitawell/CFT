import os
import cv2
import numpy as np
#train and test 先将txr文件复制到文件夹中
path = "/data1/multispectral-object-detection/data/seacuD/images/val" 
path2 = "/data1/multispectral-object-detection/data/seacuD/depth/val"

#files= os.listdir(path) #得到文件夹下的所有文件名称
#s = []
for i in range(1,1172):
    n=str(i)
    #n=n.zfill(6) #前面用0填充到6位
    
    img_path = os.path.join(path, n) + '.jpg'
    txt_path = os.path.join(path, n) + '.txt'
    img_path2 = os.path.join(path2, n) + '.jpg'
    txt_path2 = os.path.join(path2, n) + '.txt'
    if not os.path.exists(img_path):  # 如果不能打开图片
        print(img_path)
        if os.path.exists(txt_path):  # 如果txt存在，则删除
            os.remove(txt_path)
        if os.path.exists(img_path2):  # 如果图片2存在，则删除
            os.remove(img_path2)
        if os.path.exists(txt_path2):  # 如果txt2存在，则删除
            os.remove(txt_path2)
    else :
        # "a"表示以不覆盖的形式写入到文件中,当前文件夹如果没有".txt"会自动创建
        #with open("/data1/multispectral-object-detection/data/DUO/depth/train/train.txt", "a") as file:
        #with open("/data1/multispectral-object-detection/data/DUO/depth/test/val.txt", "a") as file:
           # file.write(img_path + "\n")
        #file.close()
        img = cv2.imread(img_path) #openCV读取图片
        B,G,R = cv2.split(img) #得到的数组是按照 B，G，R 的顺序返回
        width = img.shape[0]
        length = img.shape[1]
        D = np.empty(shape=(width,length))
        for i in range(width):
            for j in range(length):
                b = B[i,j]
                g = G[i,j]
                r = R[i,j]
                m = max(b,g)
                v = r
                d = 0.53214829 + 0.51309827*m - 0.91066194*v
                D[i,j] = d

        cv2.imwrite(img_path2, D)#保存图片

