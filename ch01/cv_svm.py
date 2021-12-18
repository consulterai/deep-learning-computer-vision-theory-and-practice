import os

import cv2
import numpy as np

if __name__ == '__main__':

    root_path_train = "mnist_playground_ims/svm_train"
    root_path_test = "mnist_playground_ims/svm_test"
    data = np.empty([10, 28 * 28], dtype=np.float32)
    for i in range(10):
        data[i] = cv2.imread(
            os.path.join(root_path_train, "{}.png".format(
                i)), cv2.IMREAD_GRAYSCALE).reshape(1, 28 * 28)

    label = np.arange(0, 10).reshape(10, 1)

    ###############################
    # 训练
    svm = cv2.ml.SVM_create()
    # 属性设置
    svm.setType(cv2.ml.SVM_C_SVC)  # 设置svm类型
    svm.setKernel(cv2.ml.SVM_LINEAR)  # 设置svm内核，线性内核
    svm.setC(0.01)  # 和svm内核有关的参数

    result = svm.train(data, cv2.ml.ROW_SAMPLE,
                       label)  # 进行训练，第一个数据，第二个类型，第三个标签

    ##################################
    # 预测
    test_batch = len(os.listdir(root_path_test))
    pt_data = np.empty([test_batch, 28 * 28], dtype=np.float32)
    for i, file in enumerate(os.listdir(root_path_test)):
        pt_data[i] = cv2.imread(
            os.path.join(root_path_test, file),
            cv2.IMREAD_GRAYSCALE).reshape(1, 28 * 28)

    ret = svm.predict(pt_data)
    print(ret)
