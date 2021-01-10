from tensorflow.keras import layers
from tensorflow.keras import models


class BASIC:

    def __init__(self, num_classes, img_size, channels, name="basic"):
        self.name = name
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels

    def build(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.input_width_height,
                                                                            self.input_width_height,
                                                                            self.channels)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))  # Dropout for regularization
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model
