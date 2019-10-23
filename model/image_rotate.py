import os
from PIL import Image, ImageOps

def folder_list(file_path):

    folder = os.listdir(file_path)
    folder.sort()
    label_list=[]

    for data_folder in folder:
        print("Working on {} folder..".format(data_folder))
        data_folder_path = os.path.join(file_path,'{}'.format(data_folder))
        data_folder_list = os.listdir(data_folder_path)
        data_folder_list.sort()
        label_list.append(data_folder)

        #build fix folder
        target_path = '/home/mll/Capstone/fix_image_set'
        print("{}/{}".format(target_path, data_folder))
        os.mkdir(target_path + "/" + data_folder + "/train/")
        os.mkdir(target_path + "/" + data_folder + "/vaild/")
        os.mkdir(target_path + "/" + data_folder + "/test/")

        #case is 3 train, vaild, test
        #image total 6000
        #train 5000 vaild 500 test 500

        count=0
        for imagefile in data_folder_list:
            image = Image.open(data_folder_path+"/"+imagefile)
            image = image.convert('RGB')
            save_image = image.resize((255,255))
            save_image = save_image.rotate(180)


            if(count<5000):
                save_image.save(target_path + "/" + data_folder + "/train/" + imagefile, quality=100)
            elif (count >= 5000 & count<5500):
                save_image.save(target_path + "/" + data_folder + "/vaild/" + imagefile, quality=100)
            else:
                save_image.save(target_path + "/" + data_folder + "/test/" + imagefile, quality=100)

        print("done!")


'''
attractionList = ['angel', 'statue of liberty', 'niagara falls', 'colosseum', 'pyramid']
    attractionFolderList = ['eiffel', 'liberty', 'niagara', 'colosseum', 'pyramid']
    for attractionFolder in attractionFolderList:
        image_dir = "./data_orign/" + attractionFolder + "/"
        target_resize_dir = "./data/" + attractionFolder + "/"
        target_rotate_dir = "./data_rotate/" + attractionFolder + "/"
        if not os.path.isdir(target_resize_dir):
            os.makedirs(target_resize_dir)
        if not os.path.isdir(target_rotate_dir):
            os.makedirs(target_rotate_dir)
        files = glob.glob(image_dir + "*.*")
        print(len(files))
        count = 1;
        size = (224, 224)
        for file in files:
            im = Image.open(file)
            im = im.convert('RGB')
            print("i: ", count, im.format, im.size, im.mode, file.split("/")[-1])
            count += 1
            im = ImageOps.fit(im, size, Image.ANTIALIAS, 0, (0.5, 0.5))
            im.save(target_resize_dir + file.split("/")[-1].split(".")[0] + ".jpg", quality=100)
            im.rotate(90).save(target_rotate_dir + "resize_" + file.split("/")[-1].split(".")[0] + ".jpg", quality=100)
'''


#def image_separate():




def main():

    print("Image rorate start...")
    folder_list("/home/mll/Capstone/image_set")
    #image_separate("/home/mll/Capstone/image_set")


if __name__ == "__main__":
    main()