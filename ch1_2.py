import PIL.Image as Image
import os

path = 'D:\\01 A  Dream\\09 机器学习\\ch1\\Pictures\\'  # 图片集地址
Format = ['.jpg', '.JPG']  # 图片格式
size = 256  # 每张小图片的大小
row = 3  # 图片间隔，也就是合并成一张图后，一共有几行
column = 3  # 图片间隔，也就是合并成一张图后，一共有几列
SavePath = 'D:\\01 A  Dream\\09 机器学习\\ch1\\Pictures\\final.jpg'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(path) for item in Format if
               os.path.splitext(name)[1] == item]

# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != row * column:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")


# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB', (column * size, row * size))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, row + 1):
        for x in range(1, column + 1):
            from_image = Image.open(path + image_names[column * (y - 1) + x - 1]).resize(
                (size, size), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * size, (y - 1) * size))
    return to_image.save(SavePath)  # 保存新图


image_compose()  # 调用函数
