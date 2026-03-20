from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_path, img_size=(160, 160), batch_size=16):

    train_dir = f"{base_path}/train"
    val_dir = f"{base_path}/val"
    test_dir = f"{base_path}/test"

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.05,
         height_shift_range=0.05

    )

    test_val_datagen = ImageDataGenerator(rescale=1./255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_data = test_val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_data = test_val_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_data, val_data, test_data
