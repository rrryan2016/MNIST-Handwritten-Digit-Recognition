import numpy as np 
import struct 
import matplotlib.pyplot as plt # 训练集文件 

train_images_idx3_ubyte_file = 'MNIST_data/train-images.idx3-ubyte' # 训练集标签文件 
train_labels_idx1_ubyte_file = 'MNIST_data/train-labels.idx1-ubyte' # 测试集文件 
test_images_idx3_ubyte_file = 'MNIST_data/t10k-images.idx3-ubyte' # 测试集标签文件 
test_labels_idx1_ubyte_file = 'MNIST_data/t10k-labels.idx1-ubyte' 

# Test Image
def decode_idx3_ubyte(idx3_ubyte_file): 

    """
    general function to parse idx3 file
    :param idx3_ubyte_file: file path of idx3
    :return: dataset 
    """ 

    # 读取二进制数据  Read Binary Data 
    bin_data = open(idx3_ubyte_file, 'rb').read() 
    # Parse header info: respectively, magic number, number of pictures, height and width of per picture  
    offset = 0 
    fmt_header = '>iiii' 
    #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset) 
    # print('magic number:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols)) 
    
    #解析数据集  Parsing dataset
    image_size = num_rows * num_cols 
    offset += struct.calcsize(fmt_header) 
    #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，读取了前4行之后，指针位置（即偏移位置offset）指向0016。 
    # print(offset) 
    fmt_image = '>' + str(image_size) + 'B' 
    #图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值） 
    # print(fmt_image,offset,struct.calcsize(fmt_image)) 
    images = np.empty((num_images, num_rows, num_cols)) 
    #plt.figure() 
    for i in range(num_images): 
        # if (i + 1) % 10000 == 0: 
            # print('已解析 %d' % (i + 1) + '张') 
            # print(offset) 
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols)) 
        #print(images[i]) 
        offset += struct.calcsize(fmt_image) 
    #plt.imshow(images[i],'gray') 
    #plt.pause(0.00001) 
    #plt.show() 
    #plt.show() 
    return images 

# Test Label
def decode_idx1_ubyte(idx1_ubyte_file): 
    """
    General function of parsing idx1 file
    :param idx1_ubyte_file: file path of idx1
    :return: dataset
    """ 
    # Binary data Reading 
    bin_data = open(idx1_ubyte_file, 'rb').read() 
    # Parsing header info: respectively, magic number and label number  
    offset = 0 
    fmt_header = '>ii' 
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset) 
    # print('magic number:%d, number of pictures: %d' % (magic_number, num_images)) 

    # Parsing dataset
    offset += struct.calcsize(fmt_header) 
    fmt_image = '>B' 
    labels = np.empty(num_images) 
    for i in range(num_images): 
        # if (i + 1) % 10000 == 0: 
        #     print ('%d' % (i + 1) + 'pictures done') 
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0] 
        offset += struct.calcsize(fmt_image) 
    return labels 


def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file): 
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file) 

def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file): 
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """ 
    return decode_idx1_ubyte(idx_ubyte_file) 

def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file): 
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """ 
    return decode_idx3_ubyte(idx_ubyte_file) 

def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file): 
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """ 
    return decode_idx1_ubyte(idx_ubyte_file) 

def get_MNIST_data(num_training=50000,num_validation=10000,num_test=10000):
    train_images = load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file) 
    train_labels = load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file) 
    test_images = load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file) 
    test_labels = load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file) 
    
    # Subsample the data 
    mask = range(num_training,num_training + num_validation)
    X_val = train_images[mask]
    y_val = train_labels[mask]
    mask = range(num_training)
    X_train = train_images[mask]
    y_train = train_labels[mask]
    mask = range(num_test)
    X_test = test_images[mask]
    y_test = test_labels[mask]

    # Normalize the data: substract the mean image 
    mean_image = np.mean(X_train,axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    X_train = X_train[:,np.newaxis]
    X_val = X_val[:,np.newaxis]
    X_test = X_test[:,np.newaxis]

    # # 查看前十个数据及其标签以读取是否正确
    # # Show first 10 data graph and their labels to check are they correct 
    # for i in range(10):
    #     print(train_labels[i])  # its data format is numpy.float64
    #     print(train_images[i])
    #     print(train_images[i].shape) # 28 * 28 No Channel
    #     # plt.imshow(train_images[i], cmap='gray')
    #     # plt.pause(0.000001)
    #     # plt.show()


    # Package data into a dictionary    
    return {'X_train':X_train,'y_train':y_train,
            'X_val':X_val,'y_val':y_val,
            'X_test':X_test,'y_test':y_test,
            }
