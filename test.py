import os

file_path = "./sample"
file_list = os.listdir(file_path)
imgs_path = []

print(file_list)

for f in file_list:
    img = f'{file_path}/{f}'
    imgs_path.append(img)

for i in imgs_path:
    print(i[9:-4])


print(imgs_path)