import os
path = "/data1/multispectral-object-detection/data/DUO/depth/train"
#train and test 先将txr文件复制到文件夹中
path2 = "/data1/multispectral-object-detection/data/DUO/images/train" 

#files= os.listdir(path) #得到文件夹下的所有文件名称
#s = []
for i in range(1,6672):
    n=str(i)
    n=n.zfill(6) #前面用0填充到6位
    
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
        with open("/data1/multispectral-object-detection/data/DUO/depth/train/train.txt", "a") as file:
        #with open("/data1/multispectral-object-detection/data/DUO/depth/test/val.txt", "a") as file:
            file.write(img_path + "\n")
        file.close()




