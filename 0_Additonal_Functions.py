'''
Additional Function
:: 이미지 전처리와 이후 작업을 위한 함수들 정의
참조 : https://deepestdocs.readthedocs.io/en/latest/003_image_processing/0030/
but 평균을 0으로 맞추는 작업 / 데이터 크기(scale) 조정
1. zero centering : 훈련 데이터 전체에 대해 각 픽셀에 들어갈 값이 평균 0이 되도록
=> 전체 이미지를 더하고 갯수로 나눠 '평균 이미지'를 구하고 이 이미지를 모든 이미지에서 뺌
2. contrast normalization : 다양한 이미지의 경우 이미지마다 밝기, 찍은 환경이 다름, 어느정도 통일해주는 함수
3. resize =>
'''

# module import
import glob
import os
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


# 1. 해당 디렉토리 파일이름 & 확장자 일괄적 변경 함수
def rename_files_and_change_file_extension(dir, new_name, new_ext):
    files = os.listdir(dir)
    for i, file in enumerate(files):
        os.rename(os.path.join(dir, file), os.path.join(dir, new_name + str(i) + '.' + new_ext))


# 2. 폴더에서 이미지 불러와서 정규화된 사진과 정답 레이블을 X, y로 반환
def raise_image(img_path):
    this_dir = img_path
    categories = os.listdir(this_dir)
    num_cat = len(categories)

    image_w, image_h = 64, 64
    X = []
    y = []
    for idx, cat in enumerate(categories):
        # one-hot label
        label = [0 for i in range(num_cat)]
        label[idx] = 1
        cat_dir = this_dir + "/" + cat

        # 4) 현재 보유중인 이미지 현황 inform
        # glob :: 파일의 목록을 뽑을 때 사용, 파일의 경로명을 이용해 마음대로 쓸 수 있음
        files = glob.glob(cat_dir + "/*.jpg")
        print(cat, "이미지 파일 수:", len(files))  # 각 카테고리별 이미지
        for i, f in enumerate(files):
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize((image_w, image_h), Image.ANTIALIAS)
            # asarry : 입력 데이터를 ndarray로 변환, 하지만 이미 ndarray인 경우 새로 메모리에 ndarray가 생성되진 않음
            data = np.asarray(img)
            X.append(data)
            y.append(label)
    X = np.array(X).astype(float) / 255
    y = np.array(y)

    return X, y

# 3. Image Augmentation :: 경로를 입력하면 해당 경로의 이미지 부스터해주는 함수
def image_booster(image_path, save_path, n=50):
    '''
    :param image_path: 부스터할 이미지가 있는 경로
    :param save_path: 부스터할 이미지를 저장할 경로
    :param n: 얼마나 증가시킬건지?
    :return:
    '''
    dir_path = image_path
    sv_path = save_path
    data_aug_gen = ImageDataGenerator(rescale=1. / 255,
                                      rotation_range=15,  # 0-15도 사이 각도 회전
                                      width_shift_range=0.1,  # 수평방향 범위 내 이동
                                      height_shift_range=0.1,  # 수직방향 범위 내 이동
                                      shear_range=0.5,  # 해당 범위 내 이미지 변형
                                      zoom_range=[0.8, 2.0],  # 지정된 범위 내 임의로 이미지 확대/ 축소
                                      horizontal_flip=True,  # 수평 방향 뒤집기
                                      vertical_flip=True,  # 수직 방향 뒤집기
                                      fill_mode='nearest')

    filename_in_dir = []
    for root, dirs, files in os.walk(dir_path):
        for fname in files:
            full_fname = os.path.join(root, fname)
            filename_in_dir.append(full_fname)

    for file_image in filename_in_dir:
        print(file_image)
        img = load_img(file_image)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in data_aug_gen.flow(x, batch_size=1, save_to_dir=sv_path,
                                       save_prefix=os.path.basename(fname), save_format='jpg'):
            i += 1
            if i > n:
                break


if __name__ == '__main__':

    # 각 파일 이름들을 변경 후 확장자 jpg로 바꿈
    es_dir = './test/b'
    rename_files_and_change_file_extension(es_dir, 'scene', 'jpg')
    path = './test'
    X, y = raise_image(path)
    print(X)
    print(y)
    sv = './test/save'
    image_booster(es_dir, sv)