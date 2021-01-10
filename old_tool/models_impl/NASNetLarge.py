from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import NASNetLarge
from old_tool.utils_backup.training_utils import training_EXP, test_EXP


class NASNet:
    def __init__(self, num_classes, img_size, channels, weights='imagenet', name="InceptionResNet152V2",
                 include_top=False):
        self.name = name
        self.weights = weights
        self.include_top = include_top
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels
        self.model = None
        self.trained = False

    def build(self):

        base_model = None
        output = None

        if self.include_top:
            if self.input_width_height != 224 or self.channels != 3:
                print("IF include_top=True, input_shape MUST be (224,224,3), exiting...")
                exit()
            else:
                base_model = NASNetLarge(weights=self.weights, include_top=True, classes=self.num_classes)
                output = base_model.output
        else:
            inputs = Input(shape=(self.input_width_height, self.input_width_height, self.channels))
            base_model = NASNetLarge(weights=self.weights, include_top=False, input_tensor=inputs)
            flatten = Flatten(name='my_flatten')
            output_layer = Dense(self.num_classes, activation='softmax', name='my_predictions')
            output = output_layer(flatten(base_model.output))

        input_layer = base_model.input

        model = Model(input_layer, output)
        # model.summary(line_length=50)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        self.model = model

    '''
        training and test function provided to copy basic_exp functions and make use of main_expert_DB
    '''
    def training(self, train_ds=None, epochs=None,  val_ds=None, tb_logs=None, tr_vl_steps=(None, None),
                 fw_cm=(None, None)):
        if train_ds is None:
            print("train_ds not provided, exiting...")
            exit()

        res, _ = training_EXP(self, train_ds, epochs, val_ds=val_ds, tf_callback=tb_logs)

        self.trained = True

        return res

    def test(self, test_ds=None, train_ds=None, epochs=None, tb_logs=None, tr_te_steps=(None, None),
             fw_cm=(None, None)):

        if test_ds is None:
            print("test_ds not provided, exiting...")
            exit()

        if train_ds is not None or not self.trained or self.model is None:
            print("Model reinitialized, start training over the entire training set before testing")
            _ = self.training(train_ds=train_ds, epochs=epochs, tb_logs=tb_logs)

        _ = test_EXP(self, test_ds=test_ds, fw_cm=fw_cm)
