import fiftyone as fo
import fiftyone.zoo as foz
import yaml
import glob
import shutil
import os


def getImagesFromCoco(newFolderName, imageAmount, classPresent):
    dataset, samples = getSamples(imageAmount, classPresent)
    classOfData = getClasses(dataset)
    exportSamples(newFolderName, samples, classOfData)
    maketxtFiles(newFolderName)
    copyImages(newFolderName)
    deleteFiles(newFolderName)



def getSamples(imageAmount, classPresent):   
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        dataset_name="coco-2017-"+str(imageAmount),
        max_samples=imageAmount,
        classes = [classPresent],
        shuffle=True,
    )
    samples = dataset.take(imageAmount)
    return dataset, samples

def getClasses(dataset):
    classOfData = dataset.distinct("ground_truth.detections.label")
    return classOfData

def exportSamples(newFolderName, samples, classOfData):
    samples.export(
        export_dir="./"+newFolderName+"/ground_truth",
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=classOfData,
    )

def maketxtFiles(newFolderName):
    directory = './'+newFolderName+'/'

    data_loaded = ""
    with open(directory+'ground_truth/dataset.yaml', "r") as stream:
            data_loaded = yaml.safe_load(stream)

    for filename in os.listdir(directory + 'ground_truth' + '/labels/val'):
        if filename.endswith('.txt'):
            with open(os.path.join(directory + 'ground_truth' + '/labels/val', filename)) as f:
                lines = f.readlines()
                arrWrite = []
                for l in lines:
                    data = l.split()
                    x_center = float(data[1])
                    y_center = float(data[2])
                    width = float(data[3])
                    height = float(data[4])
                    xmin = x_center - width / 2
                    ymin = y_center - height / 2
                    xmax = x_center + width / 2
                    ymax = y_center + height / 2
                    arrWrite.append(data_loaded['names'][int(data[0])] + ' ' + str(xmin) + ' ' + str(ymin) + ' ' + str(xmax) + ' ' + str(ymax) + '\n')

                newpath = directory+'/ground_truth/' 
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                with open(directory+'/ground_truth/'+ filename, 'w') as f:
                    f.writelines(arrWrite)


def copyImages(newFolderName):
    src = './'+newFolderName+'/'

    src_dir = src + 'ground_truth/images/val/'

    dst_dir = src +'images/'
    os.makedirs(dst_dir)

    for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):

        shutil.copy(jpgfile, dst_dir)

def deleteFiles(newFolderName):


    root_dir ='./'+newFolderName+'/ground_truth/'

    for dir_name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, dir_name)
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
    file_path = os.path.join(root_dir, 'dataset.yaml')
    if os.path.exists(file_path):
        os.remove(file_path)

getImagesFromCoco('twentyFive', 25, 'person')