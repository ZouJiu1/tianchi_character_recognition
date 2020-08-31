import os
import cv2
import json
import shutil
import numpy as np
seq = os.sep
pwd = os.path.abspath(r'.%s'%seq)

'''
将数据下载到data目录中，并将数据解压到单独的文件夹中mchar_train、mchar_val和mchar_test_a
理解数据和清洗的过程
'''
def read_json(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as obj:
            jsonf = json.load(obj)
    except:
        with open(filename, 'r', encoding='gbk') as obj:
            jsonf = json.load(obj)
    return jsonf

def length(jsonf, name): #查看字符长度
    jsonf = read_json(jsonf)
    length = {}
    classes = set()
    for key, value in jsonf.items():
        if len(jsonf[key]['label']) not in length.keys():
            length[len(jsonf[key]['label'])] = 1
        else:
            length[len(jsonf[key]['label'])] += 1
        classes= classes|set(jsonf[key]['label'])
        # if 0 in jsonf[key]['label']:
        #     print(key, jsonf[key])
    classes = list(classes)
    classes.sort()
    length = sorted(length.items(), key=lambda x:x[0])
    print('%s字符长度以及个数集合：'%name, length)  # 可见其中的字符长度不会超过6个
    print('%s字符类别集合和类别数：'%name, classes, '， ', len(classes), '个类别')
    with open(os.path.join(pwd, r'data%scustom%sclasses.names'%(seq, seq)), 'w') as obj:
        for i in classes:
            obj.write(str(i))
            obj.write('\n')

trainjson = os.path.join(pwd, r'data%smchar_train.json'%seq)
valjson = os.path.join(pwd, r'data%smchar_val.json'%seq)
length(trainjson, '训练集') #查看其中字符的长度
length(valjson, '验证集')

#据上可以知道有10个类，字符长度不会多于6个
#由于数据label中没有错误内容，所以不需要进行数据清洗，这里可以直接使用这些数据来训练模型
###################################################################################################################

def process(dict1, shape):
    l = ''
    for i in range(len(dict1['left'])):
        l += str(dict1['label'][i]) + ' ' + \
             str((dict1['left'][i] + dict1['width'][i] / 2) / shape[1]) + ' ' + \
             str((dict1['top'][i] + dict1['height'][i] / 2) / shape[0]) + ' ' + \
             str(dict1['width'][i] / shape[1]) + ' ' + \
             str(dict1['height'][i] / shape[0]) \
             + '\n'
    return l
def genmove(jsonfile, way):
    f = open(jsonfile, encoding='utf-8')
    data = json.load(f)
    for i in data.keys():
        img = cv2.imread(os.path.join(pwd, r'data%smchar_%s'%(seq, way)) + seq + i)
        shape = img.shape
        if way=='val':
            _i='_'+i
        else:
            _i=i
        shutil.copyfile(os.path.join(pwd, r'data%smchar_%s'%(seq,way)) + seq + i, \
                        r'data%scustom%simages'%(seq, seq) + seq + _i)
        f = open(os.path.join(pwd, r'data%scustom%slabels%s%s.txt' %(seq, seq, seq, _i.replace('.jpg', '').replace('.png', ''))),\
                 'w')
        f.write(process(data[i], shape))
        f.close()

def write_train_val(file, name):
    with open(os.path.join(pwd, r'data%scustom%s%s.txt'%(seq, seq, name)), 'w') as obj:
        for i in file:
            obj.write(i)
            obj.write('\n')

#处理数据集成YOLOV3需要的VOC数据标注格式
col = {'train':trainjson, 'val':valjson}
try:
    shutil.rmtree(os.path.join(pwd, r'data%scustom%simages'%(seq, seq)))
    shutil.rmtree(os.path.join(pwd, r'data%scustom%slabels'%(seq, seq)))
except:
    pass
os.mkdir(os.path.join(pwd, r'data%scustom%simages'%(seq, seq)))
os.mkdir(os.path.join(pwd, r'data%scustom%slabels'%(seq, seq)))
for i in col:
    genmove(col[i], i)

#自动划分训练集和验证集
imagepath = os.path.join(pwd, r'data%scustom%simages'%(seq, seq))
namelist = [os.path.join(imagepath, i) for i in os.listdir(imagepath)]
trainchoose = list(np.random.choice(namelist, len(namelist)*8//10, replace=False))
valchoose = list(set(namelist) - set(trainchoose))
print('训练集图片个数：{}，验证集图片个数：{}'.format(len(trainchoose), len(valchoose)))
write_train_val(trainchoose, 'train')
write_train_val(valchoose, 'valid')

with open(os.path.join(pwd, r'config%scustom_.data'%seq), 'w') as obj:
    obj.write('classes={}\ntrain=data{}custom{}train.txt\nvalid=data{}custom{}valid.txt\nnames=data{}custom{}classes.names\n'.\
              format(10,seq,seq,seq,seq,seq,seq))

'''
以下的内容代码里面都修改好了，可以不看的
手动修改配置文件config/yolov3.cfg，合理设置batch和subdivisions，width和height都设置为512，有三处classes设置为10
classes前的第一个filters：3*(5+len(classes))=45,共要修改三处filters；

修改train.py文件中的第31行，"--data_config", type=str, default="config/custom_.data"
修改train.py文件中的第34行，"--img_size", type=int, default=512
修改train.py文件中的第28行，"--batch_size", type=int, default=2
'''