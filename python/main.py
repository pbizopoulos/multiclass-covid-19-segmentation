from os import getenv
from pathlib import Path
from shutil import move, rmtree
from zipfile import ZipFile

import gdown
import nibabel as nib
import numpy as np
import requests
import tensorflow as tf
import tensorflowjs as tfjs
from keras import optimizers
from keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from keras.models import Model
from keras.utils import to_categorical


def data_generator_1(index_range: range) -> tuple:  # type: ignore[type-arg]
    urls = [
        "https://drive.google.com/uc?id=1SJoMelgRqb0EuqlTuq6dxBWf2j9Kno8S",
        "https://drive.google.com/uc?id=1MEqpbpwXjrLrH42DqDygWeSkDq0bi92f",
        "https://drive.google.com/uc?id=1zj4N_KV0LBko1VSQ7FPZ38eaEGNU0K6-",
    ]
    file_names = ["tr_im.nii.gz", "tr_mask.nii.gz", "tr_lungmasks_updated.nii.gz"]
    for url, file_name in zip(urls, file_names):
        file_name_path = Path(f"tmp/{file_name}")
        if not file_name_path.is_file():
            gdown.download(url, file_name_path.as_posix(), quiet=False)
    images_file_path = "tmp/tr_im.nii.gz"
    images = nib.load(images_file_path)  # type: ignore[attr-defined]
    images = images.get_fdata()[..., index_range]  # type: ignore[attr-defined]
    images = np.moveaxis(images, -1, 0)  # type: ignore[arg-type,assignment]
    mask_lesions_file_path = "tmp/tr_mask.nii.gz"
    mask_lesions = nib.load(mask_lesions_file_path)  # type: ignore[attr-defined]
    mask_lesions = mask_lesions.get_fdata()[..., index_range]  # type: ignore[attr-defined]
    mask_lungs_file_path = "tmp/tr_lungmasks_updated.nii.gz"
    mask_lungs = nib.load(mask_lungs_file_path)  # type: ignore[attr-defined]
    mask_lungs = mask_lungs.get_fdata()[..., index_range]  # type: ignore[attr-defined]
    mask_lungs[mask_lungs == 2] = 1  # type: ignore[comparison-overlap,index] # noqa: PLR2004
    masks = mask_lungs
    masks[mask_lesions == 1] = 2  # type: ignore[comparison-overlap,index]
    masks = np.moveaxis(masks, -1, 0)  # type: ignore[arg-type,assignment]
    return (images, masks)


def data_generator_2(index_volume: int) -> tuple:  # type: ignore[type-arg]
    index_volume = 0
    urls = [
        "https://drive.google.com/uc?id=1ruTiKdmqhqdbE9xOEmjQGing76nrTK2m",
        "https://drive.google.com/uc?id=1gVuDwFeAGa6jIVX9MeJV5ByIHFpOo5Bp",
        "https://drive.google.com/uc?id=1MIp89YhuAKh4as2v_5DUoExgt6-y3AnH",
    ]
    file_names = ["rp_im.zip", "rp_msk.zip", "rp_lung_msk.zip"]
    for url, file_name in zip(urls, file_names):
        zip_file_path = Path(f"tmp/{file_name}")
        if not zip_file_path.is_file():
            gdown.download(url, zip_file_path.as_posix(), quiet=False)
            with ZipFile(zip_file_path, "r") as zip_file:
                zip_file.extractall("tmp")
    image_file_paths = sorted(Path("tmp/rp_im/").glob("*.nii.gz"))
    images = nib.load(image_file_paths[index_volume])  # type: ignore[attr-defined]
    images = images.get_fdata()  # type: ignore[attr-defined]
    images = np.moveaxis(images, -1, 0)  # type: ignore[arg-type,assignment]
    mask_lesions_file_paths = sorted(Path("tmp/rp_msk/").glob("*.nii.gz"))
    mask_lesions = nib.load(mask_lesions_file_paths[index_volume])  # type: ignore[attr-defined]
    mask_lesions = mask_lesions.get_fdata()  # type: ignore[attr-defined]
    mask_lungs_file_paths = sorted(Path("tmp/rp_lung_msk/").glob("*.nii.gz"))
    mask_lungs = nib.load(mask_lungs_file_paths[index_volume])  # type: ignore[attr-defined]
    mask_lungs = mask_lungs.get_fdata()  # type: ignore[attr-defined]
    mask_lungs[mask_lungs == 2] = 1  # type: ignore[comparison-overlap,index] # noqa: PLR2004
    masks = mask_lungs
    masks[mask_lesions == 1] = 2  # type: ignore[comparison-overlap,index]
    masks = np.moveaxis(masks, -1, 0)  # type: ignore[arg-type,assignment]
    return (images, masks)


def data_generator_3(index_range: range) -> tuple:  # type: ignore[type-arg]
    urls = [
        "https://zenodo.org/record/3757476/files/COVID-19-CT-Seg_20cases.zip?download=1",
        "https://zenodo.org/record/3757476/files/Infection_Mask.zip?download=1",
        "https://zenodo.org/record/3757476/files/Lung_Mask.zip?download=1",
    ]
    file_names = ["COVID-19-CT-Seg_20cases", "Infection_Mask", "Lung_Mask"]
    for url, file_name in zip(urls, file_names):
        zip_file_path = Path(f"tmp/{file_name}.zip")
        if not zip_file_path.is_file():
            response = requests.get(url, timeout=60)
            with zip_file_path.open("wb") as file:
                file.write(response.content)
            with ZipFile(zip_file_path, "r") as zip_file:
                zip_file.extractall(f"tmp/{file_name}")
    images = np.array([]).reshape(512, 512, 0)
    for file_path in Path("tmp/COVID-19-CT-Seg_20cases/").glob("*.nii.gz"):
        images_ = nib.load(file_path)  # type: ignore[attr-defined]
        images_ = np.resize(images_.get_fdata(), (512, 512, images_.shape[-1]))  # type: ignore[assignment,attr-defined]
        images = np.concatenate((images, images_), 2)  # type: ignore[arg-type]
    images = images[..., index_range]
    mask_lesions = np.array([]).reshape(512, 512, 0)
    for file_path in Path("tmp/Infection_Mask/").glob("*.nii.gz"):
        mask_lesions_ = nib.load(file_path)  # type: ignore[attr-defined]
        mask_lesions_ = np.resize(  # type: ignore[assignment]
            mask_lesions_.get_fdata(),  # type: ignore[attr-defined]
            (512, 512, mask_lesions_.shape[-1]),  # type: ignore[attr-defined]
        )
        mask_lesions = np.concatenate((mask_lesions, mask_lesions_), 2)  # type: ignore[arg-type]
    mask_lesions = mask_lesions[..., index_range]
    mask_lungs = np.array([]).reshape(512, 512, 0)
    for file_path in Path("tmp/Lung_Mask/").glob("*.nii.gz"):
        mask_lungs_ = nib.load(file_path)  # type: ignore[attr-defined]
        mask_lungs_ = np.resize(  # type: ignore[assignment]
            mask_lungs_.get_fdata(),  # type: ignore[attr-defined]
            (512, 512, mask_lungs.shape[-1]),
        )
        mask_lungs = np.concatenate((mask_lungs, mask_lungs_), 2)  # type: ignore[arg-type]
    mask_lungs = mask_lungs[..., index_range]
    mask_lungs[mask_lungs == 2] = 1  # noqa: PLR2004
    masks = mask_lungs
    masks[mask_lesions == 1] = 2
    return (images, masks)


def get_model(  # noqa: PLR0915
    classes_num: int,
    img_size: tuple,  # type: ignore[type-arg]
) -> Model:
    dropout = 0.4
    activation = "relu"
    initializer = "he_normal"
    base_filters = 32
    input_layer = Input(img_size)
    c1 = Conv2D(
        base_filters,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(input_layer)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(dropout)(c1)
    c1 = Conv2D(
        base_filters,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(
        base_filters * 2,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(dropout)(c2)
    c2 = Conv2D(
        base_filters * 2,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(
        base_filters * 4,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(dropout)(c3)
    c3 = Conv2D(
        base_filters * 4,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(
        base_filters * 8,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(dropout)(c4)
    c4 = Conv2D(
        base_filters * 8,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = Conv2D(
        base_filters * 16,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(dropout)(c5)
    c5 = Conv2D(
        base_filters * 16,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c5)
    c5 = BatchNormalization()(c5)
    u6 = Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(
        base_filters * 8,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv2D(
        base_filters * 8,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c6)
    c6 = BatchNormalization()(c6)
    u7 = Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(
        base_filters * 4,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(dropout)(c7)
    c7 = Conv2D(
        base_filters * 4,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c7)
    c7 = BatchNormalization()(c7)
    u8 = Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(
        base_filters * 2,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(dropout)(c8)
    c8 = Conv2D(
        base_filters * 2,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c8)
    c8 = BatchNormalization()(c8)
    u9 = Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(
        base_filters,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv2D(
        base_filters,
        (3, 3),
        activation=activation,
        kernel_initializer=initializer,
        padding="same",
    )(c9)
    output_layer = Conv2D(classes_num, (1, 1), activation="softmax")(c9)
    return Model(inputs=[input_layer], outputs=[output_layer])


def process_image_mask(
    image: tf.float32,
    mask: tf.float32,
) -> tuple:  # type: ignore[type-arg]
    image = tf.image.resize(image[..., tf.newaxis], (256, 256))
    image = image / 4095
    mask = tf.image.resize(mask[..., tf.newaxis], (256, 256), method="nearest")
    mask = tf.cast(mask, tf.int32)
    mask = to_categorical(mask, num_classes=3)
    return (image, mask)


def preprocess(
    images: tf.float32,
    masks: tf.float32,
) -> tuple:  # type: ignore[type-arg]
    images, masks = tf.numpy_function(
        process_image_mask,
        [images, masks],
        [tf.float32, tf.float32],
    )
    images.set_shape([256, 256, 1])
    masks.set_shape([256, 256, 3])
    return (images, masks)


def main() -> None:
    tf.keras.utils.set_random_seed(1)
    tf.config.experimental.enable_op_determinism()
    if getenv("STAGING"):
        epochs_num = 100
        index_range = range(100)
    else:
        epochs_num = 10
        index_range = range(10)
    [images, masks] = data_generator_1(index_range)
    dataset = tf.data.Dataset.from_tensor_slices((images, masks))
    dataset = dataset.map(preprocess)
    train_size = int(0.7 * len(index_range))
    validation_size = int(0.15 * len(index_range))
    test_size = int(0.15 * len(index_range))
    train_dataset = dataset.take(train_size).batch(1)
    test_dataset = dataset.skip(train_size)
    validation_dataset = test_dataset.skip(validation_size).batch(1)
    test_dataset = test_dataset.take(test_size)
    img_size = (256, 256, 1)
    classes_num = 3
    model = get_model(classes_num, img_size)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics="accuracy",
    )
    model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs_num)
    tfjs_path = Path("tmp/tfjs")
    if tfjs_path.exists():
        rmtree(tfjs_path)
    tfjs_path.mkdir(exist_ok=True)
    tfjs.converters.save_keras_model(model, tfjs_path)
    if getenv("STAGING"):
        output_path = Path("prm/model/")
        if output_path.exists():
            rmtree(output_path)
        move("tmp/tfjs", "prm/model/")


if __name__ == "__main__":
    main()
