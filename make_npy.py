'''
알고리즘 투입 전 전처리 및 npy 파일 생성
'''

import numpy as np
from sklearn.model_selection import train_test_split
from citrus_tea.additonal_function import raise_image


# pre_processing
# 1) train directory path 설정
train_dir = 'C:\\Users\\Owner\\Desktop\\확장data\\Train'

# 이미지 불러오기
X, y = raise_image(train_dir)

# train / test data 분리 후 npy 파일 저장
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# X_train, X_test shape
print(X_train.shape)   # X_train: (15916, 64, 64, 3) | y_train : (15916, 5)
print(X_test.shape)     # X_test :(3980, 64, 64, 3)  | y_test : (3980, 5)

# npy :: 1 개의 배열을 Numpy format의 바이너리 파일로 저장
# => dtype 및 shape 정보 포함 컴퓨터의 배열을 재구성하는데 필요한 모든 정보 저장
xy = (X_train, X_test, y_train, y_test)  # np.load 시 데이터를 편하게 불러오기 위해 tuple로 저장
np.save("./test_data_after.npy", xy)
