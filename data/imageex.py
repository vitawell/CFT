import os
from PIL import Image
path = "/data1/multispectral-object-detection/data/tctest/depth"
path2 = "/data1/multispectral-object-detection/data/tctest/images" 

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

#files= os.listdir(path) #得到文件夹下的所有图片
image_filenames = [os.path.join(path, x) for x in os.listdir(path) if is_image_file(x)]

for img_filename in image_filenames:
    filename = os.path.basename(img_filename)
    img_filename2 = os.path.join(path2,filename)

    img = Image.open(img_filename)
    #imgSize = img.size  #大小/尺寸
    w = img.width       #图片的宽
    h = img.height      #图片的高
    
    img2 = Image.open(img_filename2)
    #imgSize2 = img2.size  #大小/尺寸
    w2 = img2.width       #图片的宽
    h2 = img2.height      #图片的高
    
    if w < w2 or h <h2:  # 如果深度图小于原图，则改变深度图大小
        img = img.resize((w2, h2))
        img.save(img_filename)
        print(img_filename)


