from glob import glob
from os import environ
from pathlib import Path
from shutil import move, rmtree
from zipfile import ZipFile

import gdown
import nibabel as nib
import numpy as np
import requests
import tensorflow as tf
import tensorflowjs as tfjs
from keras.utils.np_utils import to_categorical
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from tensorflow.keras.models import Model


def data_generator_1(index_range: range) -> tuple: # type: ignore[type-arg]
    urls = ['https://drive.google.com/uc?id=1SJoMelgRqb0EuqlTuq6dxBWf2j9Kno8S', 'https://drive.google.com/uc?id=1MEqpbpwXjrLrH42DqDygWeSkDq0bi92f', 'https://drive.google.com/uc?id=1zj4N_KV0LBko1VSQ7FPZ38eaEGNU0K6-']
    file_names = ['tr_im.nii.gz', 'tr_mask.nii.gz', 'tr_lungmasks_updated.nii.gz']
    for url, file_name in zip(urls, file_names): # noqa: B905
        file_name_path = Path(f'bin/{file_name}')
        if not file_name_path.is_file():
            gdown.download(url, file_name_path.as_posix(), quiet=False)
    images_file_path = 'bin/tr_im.nii.gz'
    images = nib.load(images_file_path)
    images = images.get_fdata()[..., index_range]
    images = np.moveaxis(images, -1, 0)
    mask_lesions_file_path = 'bin/tr_mask.nii.gz'
    mask_lesions = nib.load(mask_lesions_file_path)
    mask_lesions = mask_lesions.get_fdata()[..., index_range]
    mask_lungs_file_path = 'bin/tr_lungmasks_updated.nii.gz'
    mask_lungs = nib.load(mask_lungs_file_path)
    mask_lungs = mask_lungs.get_fdata()[..., index_range]
    mask_lungs[mask_lungs == 2] = 1 # noqa: PLR2004
    masks = mask_lungs
    masks[mask_lesions == 1] = 2
    masks = np.moveaxis(masks, -1, 0)
    return (images, masks)


def data_generator_2(index_volume: int) -> tuple: # type: ignore[type-arg]
    index_volume = 0
    urls = ['https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m', 'https://drive.google.com/uc?id=1gVuDwFeAGa6jIVX9MeJV5ByIHFpOo5Bp', 'https://drive.google.com/uc?id=1MIp89YhuAKh4as2v_5DUoExgt6-y3AnH']
    file_names = ['rp_im.zip', 'rp_msk.zip', 'rp_lung_msk.zip']
    for url, file_name in zip(urls, file_names): # noqa: B905
        zip_file_path = Path(f'bin/{file_name}')
        if not zip_file_path.is_file():
            gdown.download(url, zip_file_path.as_posix(), quiet=False)
            with ZipFile(zip_file_path, 'r') as zip_file:
                zip_file.extractall('bin')
    image_file_paths = sorted(glob('bin/rp_im/*.nii.gz'))
    images = nib.load(image_file_paths[index_volume])
    images = images.get_fdata()
    images = np.moveaxis(images, -1, 0)
    mask_lesions_file_paths = sorted(glob('bin/rp_msk/*.nii.gz'))
    mask_lesions = nib.load(mask_lesions_file_paths[index_volume])
    mask_lesions = mask_lesions.get_fdata()
    mask_lungs_file_paths = sorted(glob('bin/rp_lung_msk/*.nii.gz'))
    mask_lungs = nib.load(mask_lungs_file_paths[index_volume])
    mask_lungs = mask_lungs.get_fdata()
    mask_lungs[mask_lungs == 2] = 1 # noqa: PLR2004
    masks = mask_lungs
    masks[mask_lesions == 1] = 2
    masks = np.moveaxis(masks, -1, 0)
    return (images, masks)


def data_generator_3(index_range: range) -> tuple: # type: ignore[type-arg]
    urls = ['https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1', 'https://zenodo.org/record/3757476/files/Infection_Mask.zip?download=1', 'https://zenodo.org/record/3757476/files/Lung_Mask.zip?download=1']
    file_names = ['COVID-19-CT-Seg_20cases', 'Infection_Mask', 'Lung_Mask']
    for url, file_name in zip(urls, file_names): # noqa: B905
        zip_file_path = Path(f'bin/{file_name}.zip')
        if not zip_file_path.is_file():
            response = requests.get(url, timeout=60)
            with zip_file_path.open('wb') as file:
                file.write(response.content)
            with ZipFile(zip_file_path, 'r') as zip_file:
                zip_file.extractall(f'bin/{file_name}')
    images = np.array([]).reshape(512, 512, 0)
    for file_path in glob('bin/COVID-19-CT-Seg_20cases/*.nii.gz'):
        images_ = nib.load(file_path)
        images_ = np.resize(images_.get_fdata(), (512, 512, images_.shape[-1]))
        images = np.concatenate((images, images_), 2)
    images = images[..., index_range]
    mask_lesions = np.array([]).reshape(512, 512, 0)
    for file_path in glob('bin/Infection_Mask/*.nii.gz'):
        mask_lesions_ = nib.load(file_path)
        mask_lesions_ = np.resize(mask_lesions_.get_fdata(), (512, 512, mask_lesions_.shape[-1]))
        mask_lesions = np.concatenate((mask_lesions, mask_lesions_), 2)
    mask_lesions = mask_lesions[..., index_range]
    mask_lungs = np.array([]).reshape(512, 512, 0)
    for file_path in glob('bin/Lung_Mask/*.nii.gz'):
        mask_lungs_ = nib.load(file_path)
        mask_lungs_ = np.resize(mask_lungs_.get_fdata(), (512, 512, mask_lungs.shape[-1]))
        mask_lungs = np.concatenate((mask_lungs, mask_lungs_), 2)
    mask_lungs = mask_lungs[..., index_range]
    mask_lungs[mask_lungs == 2] = 1 # noqa: PLR2004
    masks = mask_lungs
    masks[mask_lesions == 1] = 2
    return (images, masks)


def get_model(classes_num: int, img_size: tuple) -> Model: # type: ignore[type-arg] # noqa: PLR0915
    dropout = 0.4
    activation = 'relu'
    initializer = 'he_normal'
    base_filters = 32
    input_layer = Input(img_size)
    c1 = Conv2D(base_filters, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(input_layer)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)
    c1 = Conv2D(base_filters, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(base_filters * 2, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(dropout)(c2)
    c2 = Conv2D(base_filters * 2, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(base_filters * 4, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(dropout)(c3)
    c3 = Conv2D(base_filters * 4, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(base_filters * 8, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(dropout)(c4)
    c4 = Conv2D(base_filters * 8, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(base_filters * 16, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(dropout)(c5)
    c5 = Conv2D(base_filters * 16, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c5)
    c5 = BatchNormalization()(c5)
    u6 = Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(base_filters * 8, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv2D(base_filters * 8, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c6)
    c6 = BatchNormalization()(c6)
    u7 = Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(base_filters * 4, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(dropout)(c7)
    c7 = Conv2D(base_filters * 4, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c7)
    c7 = BatchNormalization()(c7)
    u8 = Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(base_filters * 2, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(dropout)(c8)
    c8 = Conv2D(base_filters * 2, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c8)
    c8 = BatchNormalization()(c8)
    u9 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(base_filters, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv2D(base_filters, (3, 3), activation=activation, kernel_initializer=initializer, padding='same')(c9)
    output_layer = Conv2D(classes_num, (1, 1), activation='softmax')(c9)
    return Model(inputs=[input_layer], outputs=[output_layer])


def main() -> None:
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    epochs_num = 100
    index_range = range(100)
    if environ['DEBUG'] == '1':
        epochs_num = 10
        index_range = range(10)
    [images, masks] = data_generator_1(index_range)
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(preprocess)
    training_size = int(0.7 * len(index_range))
    validation_size = int(0.15 * len(index_range))
    test_size = int(0.15 * len(index_range))
    train_dataset = dataset.take(training_size).batch(1)
    test_dataset = dataset.skip(training_size)
    validation_dataset = test_dataset.skip(validation_size).batch(1)
    test_dataset = test_dataset.take(test_size)
    img_size = (256, 256, 1)
    classes_num = 3
    model = get_model(classes_num, img_size)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics='accuracy')
    model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs_num)
    tfjs_path = Path('bin/tfjs')
    if tfjs_path.exists():
        rmtree(tfjs_path)
    tfjs_path.mkdir(exist_ok=True)
    tfjs.converters.save_keras_model(model, tfjs_path)
    if environ['DEBUG'] != '1':
        dist_path = Path('dist')
        if dist_path.exists():
            rmtree(dist_path)
        move('bin/tfjs', 'dist')


def preprocess(images: tf.float32, masks: tf.float32) -> tuple: # type: ignore[type-arg]
    images, masks = tf.numpy_function(process_image_mask, [images, masks], [tf.float32, tf.float32])
    images.set_shape([256, 256, 1])
    masks.set_shape([256, 256, 3])
    return (images, masks)


def process_image_mask(image: tf.float32, mask: tf.float32) -> tuple: # type: ignore[type-arg]
    image = tf.image.resize(image[..., tf.newaxis], (256, 256))
    image = image / 4095
    mask = tf.image.resize(mask[..., tf.newaxis], (256, 256), method='nearest')
    mask = tf.cast(mask, tf.int32)
    mask = to_categorical(mask, num_classes=3)
    return (image, mask)


if __name__ == '__main__':
    main()
