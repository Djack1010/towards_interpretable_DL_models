import kerastuner as kt
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Dropout, Input
from tensorflow.keras import Model, losses, optimizers, metrics
from old_tool.utils_backup.analyzing_data import log_confusion_matrix
import tensorflow as tf
import numpy as np
from old_tool.utils_backup.generic_utils import *
from datetime import datetime, timedelta


class MyModel:

    def __init__(self, num_classes, img_size, channels, name="nedo_exp"):
        # My variables
        self.mod_name = name
        self.num_classes = num_classes
        self.input_width_height = img_size
        self.channels = channels

        # Model Architecture
        self.model = None
        self.trained = False

        # Choose Loss and Optimizer
        # CHOOSEN IN build()
        self.loss_object = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.Adam()

        # Define our metrics
        self.train_loss = metrics.Mean('train_loss')  # dtype=tf.float32
        self.train_accuracy = metrics.CategoricalAccuracy('train_accuracy')
        self.val_loss = metrics.Mean('val_loss')
        self.val_accuracy = metrics.CategoricalAccuracy('val_accuracy')
        self.test_loss = metrics.Mean('test_loss')
        self.test_accuracy = metrics.CategoricalAccuracy('test_accuracy')

    def build(self):

        inputs = Input(shape=(self.input_width_height, self.input_width_height, self.channels))
        # x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        # outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
        # model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x = Conv2D(64, (3, 3), activation='relu', input_shape=(self.input_width_height,
                                                               self.input_width_height,
                                                               self.channels))(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(96, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.45)(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.35)(x)
        x = Dense(512, activation='relu')(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=output)
        self.model.summary()


    def build_tuning(self, hp):

        inputs = Input(shape=(self.input_width_height, self.input_width_height, self.channels))
        # x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
        # outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
        # model = tf.keras.Model(inputs=inputs, outputs=outputs)
        x = Conv2D(hp.Int('filters_1', 16, 128, step=16), (3, 3), activation='relu',
                   input_shape=(self.input_width_height, self.input_width_height, self.channels))(inputs)
        x = MaxPooling2D((2, 2))(x)

        for i in range(hp.Int('conv_blocks', 2, 5, default=3)):
            x = Conv2D(hp.Int('filters_' + str(i), 32, 256, step=32), kernel_size=(3, 3), activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            #if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
            #    x = tf.keras.layers.MaxPool2D()(x)
            #else:
            #    x = tf.keras.layers.AvgPool2D()(x)

        x = Flatten()(x)
        x = Dropout(hp.Float('dropout', 0, 0.7, step=0.1, default=0.5))(x)
        x = Dense(hp.Int('hidden_size', 256, 1024, step=128, default=512),
                  activation=hp.Choice('act_1', ['relu', 'tanh']))(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        # Choose Loss and Optimizer
        self.loss_object = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.Adam(hp.Float('learning_rate', 1e-5, 1e-1))

        self.model = Model(inputs=inputs, outputs=output)
        self.model.compile(optimizer=self.optimizer, loss=self.loss_object, metrics=['accuracy'])

        return self.model

    def reinitialize_model(self):
        self.build()
        self._reset_metrics()
        self.trained = False


    '''
    Methods to train the model.
    It reinitialize the model (the layers) each time and set the variable self.trained = True at the end
    IF val_ds is provided, it run a validation loop for each epoch, otherwise just train the model.
    tot_steps helps in printing progress bar (total steps is length train/batch size)
    '''
    def training(self, train_ds=None, epochs=None, tb_logs=(None, None), val_ds=None, tr_vl_steps=(None, None),
                 fw_cm=(None, None)):

        if train_ds is None or epochs is None:
            print("train_ds or epochs not provided, exiting...")
            exit()

        # reset model
        self.reinitialize_model()

        # Initialize struct for keeping track of acc/loss per epoch
        res = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

        start_training = time.perf_counter()
        print_log("START TRAINING AT\t{}".format(datetime.now().strftime('%H:%M:%S.%f'), print_on_screen=True))

        for epoch in range(epochs):

            self._reset_metrics()

            if tr_vl_steps[0] is None:
                print('Epoch {} out of {}   '.format(epoch + 1, epochs), end="\r")

            # Iterate over the batches of the dataset
            for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
                self._train_step(x_batch_train, y_batch_train)
                if tr_vl_steps[0] is not None:
                    print("{}loss:{:.3f} acc:{:.3f}    ".format(progr_bar(step, tr_vl_steps[0]), self.train_loss.result(),
                                                                self.train_accuracy.result()), end="\r")
                else:
                    # ALTERNATIVE TRAINING INFO, EVERY 100 STEPS
                    # Log every 100 batches/steps -> Seen so far: ((step + 1) * batch_size) samples
                    if step % 50 == 0:
                        print("STEP {} loss:{:.3f} acc:{:.3f}    ".format(step, self.train_loss.result(),
                              self.train_accuracy.result()), end="\r")

            clean_progr_bar()

            # Update data on Tensorboard for training
            if tb_logs[0] is not None:
                with tb_logs[0].as_default():
                    tf.summary.scalar('train_loss', self.train_loss.result(), step=epoch+1)
                    tf.summary.scalar('train_accuracy', self.train_accuracy.result(), step=epoch+1)

            template = 'Ep. {}\tTRAINING -> Loss: {:.4f}, Accuracy: {:.4f}        '
            print_log(template.format(epoch + 1, self.train_loss.result(), self.train_accuracy.result()),
                      print_on_screen=True, print_on_file=(val_ds is not None))
            print_log("\tTIME: {}".format(datetime.now().strftime('%H:%M:%S.%f')), print_on_screen=True)

            # save train result for this epoch
            # N.B. Needs to be casted cause 'metrics' return tf.Tensor
            res['train_acc'].append(float(self.train_accuracy.result()))
            res['train_loss'].append(float(self.train_loss.result()))

            # Run a validation loop at the end of each epoch, if validation dataset provided
            if val_ds is not None:
                for step, (x_batch_val, y_batch_val) in enumerate(val_ds):
                    self._test_val_step(x_batch_val, y_batch_val, validation=True)
                    if tr_vl_steps[1] is not None:
                        print("{}loss:{:.3f} acc:{:.3f}    ".format(progr_bar(step, tr_vl_steps[1]),
                                                                    self.val_loss.result(),
                                                                    self.val_accuracy.result()), end="\r")
                    else:
                        # ALTERNATIVE TRAINING INFO, EVERY 100 STEPS
                        # Log every 100 batches/steps -> Seen so far: ((step + 1) * batch_size) samples
                        if step % 10 == 0:
                            print("STEP {} loss:{:.3f} acc:{:.3f}    ".format(step, self.val_loss.result(),
                                                                              self.val_accuracy.result()), end="\r")

                clean_progr_bar()

                # Update data on Tensorboard for validation
                if tb_logs[1] is not None:
                    with tb_logs[1].as_default():
                        tf.summary.scalar('val_loss', self.val_loss.result(), step=epoch+1)
                        tf.summary.scalar('val_accuracy', self.val_accuracy.result(), step=epoch+1)

                template = '\tVALIDATE -> Loss: {:.4f}, Accuracy: {:.4f}        '
                print_log(template.format(self.val_loss.result(), self.val_accuracy.result()), print_on_screen=True)
                print_log("\tTIME: {}".format(datetime.now().strftime('%H:%M:%S.%f')), print_on_screen=True)

                # save validation result for this epoch
                # N.B. Needs to be casted cause 'metrics' return tf.Tensor
                res['val_acc'].append(float(self.val_accuracy.result()))
                res['val_loss'].append(float(self.val_loss.result()))

                # Update Confusion Matrix if fw_cm is provided
                # NB -> Very inefficient, suggest to disable for model assessment, useful only for printing graphs
                # TODO Improve efficiency
                if fw_cm[0] is not None:
                    print("Generating Confusion Matrix...    ", end="\r")
                    complete_validation = val_ds.unbatch()
                    test_pred = []
                    y_test_int = []
                    for images, labels in complete_validation.take(-1):
                        y_test_int.append(np.argmax(labels.numpy(), axis=0))
                        test_pred.append(np.argmax(self.model(np.reshape(images.numpy(), (-1, self.input_width_height,
                                                                                    self.input_width_height,
                                                                                    self.channels)),
                                                        training=False).numpy(), axis=1)[0])
                    log_confusion_matrix(test_pred, y_test_int, fw_cm[0], fw_cm[1], epoch=epoch)

        end_training = time.perf_counter()
        print_log("FINISH TRAINING AT\t{}".format(datetime.now().strftime('%H:%M:%S.%f')), print_on_screen=True)
        print_log("TRAINING EX. TIME: {} ".format(str(timedelta(seconds=end_training - start_training))),
                  print_on_screen=True)

        # Set model as trained
        self.trained = True

        return res

    '''
    Methods to test the model.
    IF train_ds and epochs are provided, it train the model before running the test
    '''
    def test(self, test_ds=None, train_ds=None, epochs=None, tr_te_steps=(None, None), tb_logs=None,
             fw_cm=(None, None)):
        if test_ds is None:
            print("test_ds not provided, exiting...")
            exit()

        if train_ds is not None and epochs is not None:
            print("Model reinitialized, start training over the entire training set before testing")
            self.training(train_ds=train_ds, epochs=epochs, tr_vl_steps=(tr_te_steps[0], None),
                          tb_logs=(tb_logs, None))

        if self.trained:

            start_testing = time.perf_counter()
            print_log("START TESTING AT\t{}".format(datetime.now().strftime('%H:%M:%S.%f')), print_on_screen=True)

            for step, (x_batch_test, y_batch_test) in enumerate(test_ds):
                self._test_val_step(x_batch_test, y_batch_test)
                if tr_te_steps[1] is not None:
                    print("{}loss:{:.3f} acc:{:.3f}    ".format(progr_bar(step, tr_te_steps[1]),
                                                                self.test_loss.result(),
                                                                self.test_accuracy.result()), end="\r")
                else:
                    # ALTERNATIVE TRAINING INFO, EVERY 100 STEPS
                    # Log every 100 batches/steps -> Seen so far: ((step + 1) * batch_size) samples
                    if step % 10 == 0:
                        print("STEP {} loss:{:.3f} acc:{:.3f}    ".format(step, self.test_loss.result(),
                                                                          self.test_accuracy.result()), end="\r")

            clean_progr_bar()

            # TODO Improve efficiency
            if fw_cm[0] is not None:
                print("Generating Confusion Matrix...    ", end="\r")
                complete_test = test_ds.unbatch()
                test_pred = []
                y_test_int = []
                for images, labels in complete_test.take(-1):
                    y_test_int.append(np.argmax(labels.numpy(), axis=0))
                    test_pred.append(np.argmax(self.model(np.reshape(images.numpy(), (-1, self.input_width_height,
                                                                                self.input_width_height,
                                                                                self.channels)),
                                                    training=False).numpy(), axis=1)[0])
                log_confusion_matrix(test_pred, y_test_int, fw_cm[0], fw_cm[1])

            template = '\tTEST -> Loss: {:.4f}, Accuracy: {:.4f}'
            print_log(template.format(self.test_loss.result(), self.test_accuracy.result()), print_on_screen=True)
            print_log("FINISH TESTING AT\t{}".format(datetime.now().strftime('%H:%M:%S.%f')), print_on_screen=True)
            end_testing = time.perf_counter()
            print_log("TESTING EX. TIME: {} ".format(str(timedelta(seconds=end_testing - start_testing))),
                      print_on_screen=True)

        else:
            print("TEST FAILED! Model is not trained...")
            exit()

    def _train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_train, training=True)
            loss = self.loss_object(y_train, predictions)
        # Use the gradient tape to automatically retrieve the gradients of the trainable variables
        # with respect to the loss.
        grads = tape.gradient(loss, self.model.trainable_variables)
        # Run one step of gradient descent by updating the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        # Update training metric
        self.train_loss(loss)
        self.train_accuracy(y_train, predictions)

    def _test_val_step(self, x_test, y_test, validation=False):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(x_test, training=False)
        loss = self.loss_object(y_test, predictions)
        # Update training metric
        if validation:
            self.val_loss(loss)
            self.val_accuracy(y_test, predictions)
        else:
            self.test_loss(loss)
            self.test_accuracy(y_test, predictions)

    def _reset_metrics(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
