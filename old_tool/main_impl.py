import argparse
import datetime
import tensorflow.keras.callbacks as callbacks
import kerastuner as kt
from old_tool.models_impl.myModel import MyModel as BASIC_EXP
from old_tool.models_impl.VGG16 import VGG16_19
from old_tool.models_impl.ResNet152V2 import ResNet
# from old_tool.utils_backup.dir_util import copy_tree
from old_tool.utils_backup.preprocessing_data import *
from old_tool.utils_backup.training_utils import *
from old_tool.utils_backup.generic_utils import *
from old_tool.utils_backup.config import *


def main(arguments):

    print_log("STARTING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)

    print("LOADING AND PRE-PROCESSING DATA")

    dataset_base = main_path + arguments.dataset

    # Create dataset of filepaths
    train_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/train/*/*")
    val_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/val/*/*")
    final_training_paths_ds = tf.data.Dataset.list_files(dataset_base + "/training/*/*/*")
    test_paths_ds = tf.data.Dataset.list_files(dataset_base + "/test/*/*")

    # STATS
    size_train = sum([len(files) for r, d, files in os.walk(dataset_base + "/training/train")])
    size_val = sum([len(files) for r, d, files in os.walk(dataset_base + "/training/val")])
    size_test = sum([len(files) for r, d, files in os.walk(dataset_base + "/test")])
    nclasses = len(CLASS_NAMES)

    # SELECTING MODELS
    model_class, logs_mode = _model_selection(arguments, nclasses)

    # Print information on log
    # EXECUTION Info
    print_log("INFO EXECUTION:"
              "\nmodel = {}\ndataset = {}\nmonitoring = {}"
              "\noutput_model = {}\nepochs = {}\nbatch_size = {}"
              "\n----------------"
              .format(arguments.model, arguments.dataset, arguments.monitoring,
                      arguments.output_model, arguments.epochs, arguments.batch_size))
    # DATA Info
    print_log("INFO DATA:"
              "\nnum_classes = {}\nsize_img= {}x{}\nDATA SPLIT= 0.2\nCLASS_NAMES= {}\nSize train-val-test= {}-{}-{}"
              "\n----------------"
              .format(nclasses, arguments.image_size, arguments.image_size, CLASS_NAMES, size_train, size_val, size_test))

    # delete previous logs
    shutil.rmtree(main_path + "tensorboard_logs/exp", ignore_errors=True)
    # initialize logs for Tensorboard
    log_dir = main_path + "tensorboard_logs/exp/{}".format(timeExec)

    # --------------  TRAINING and VALIDATION part  --------------------

    #  Use Dataset.map to create a dataset of image, label pairs
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    lab_train_ds = train_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    lab_val_ds = val_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    if arguments.caching:
        # delete previous cache files and store for this execution
        caching_file_base = main_path + "temp/"
        for f in os.listdir(caching_file_base):
            if "train.tfcache" in f or "val.tfcache" in f:
                os.remove(caching_file_base + f)
        train_ds = prepare_for_training(lab_train_ds, cache=caching_file_base + "train.tfcache")
        val_ds = prepare_for_training(lab_val_ds, cache=caching_file_base + "val.tfcache")
    else:
        train_ds = prepare_for_training(lab_train_ds)
        val_ds = prepare_for_training(lab_val_ds)

    print_log('Start Training for {} epochs  '.format(arguments.epochs), print_on_screen=True)

    # Initialize writers or callback for keeping track of results on tensorboard
    if logs_mode == "writers":
        tb_logs = (tf.summary.create_file_writer(log_dir + "/train"), tf.summary.create_file_writer(log_dir + "/val"))
    elif logs_mode == "callbacks":
        tb_logs = callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch')
    else:
        tb_logs = None

    # If monitoring set, initialize writer for keeping track cm during validation
    file_writer_cm_val = tf.summary.create_file_writer(log_dir + "/cm_val") \
        if arguments.monitoring else None

    tot_steps_tr = (size_train // arguments.batch_size) + 1
    tot_steps_vl = (size_val // arguments.batch_size) + 1

    if arguments.tuning is not None:
        print("TUNING MODEL HYPER-PARAMETERS with '{}'".format(arguments.tuning))
        tuner = None
        # https://github.com/keras-team/keras-tuner
        if arguments.tuning == "hyperband":
            tuner = kt.Hyperband(
                model_class.build_tuning,
                # `tune_new_entries=False` prevents unlisted parameters from being tuned
                tune_new_entries=False,
                objective='val_accuracy',
                max_epochs=30,
                hyperband_iterations=2,
                directory='tuning',
                project_name='wt_hyperband'
            )
        elif arguments.tuning == "bayesian":
            tuner = kt.tuners.BayesianOptimization(
                model_class.build_tuning,
                objective='val_accuracy',
                max_trials=50,
                directory='tuning',
                project_name='wt_bayesian'
            )
        elif arguments.tuning == "random":
            tuner = kt.tuners.RandomSearch(
                model_class.build_tuning,
                objective='val_accuracy',
                max_trials=5,
                executions_per_trial=3,
                directory='tuning',
                project_name='wt_random'
            )

        # # """Case #4:
        # # - We restrict the search space
        # # - This means that default values are being used for params that are left out
        # # """
        #
        # hp = HyperParameters()
        # hp.Choice('learning_rate', [1e-1, 1e-3])
        #
        # tuner = RandomSearch(
        #     build_model,
        #     max_trials=5,
        #     hyperparameters=hp,
        #     tune_new_entries=False,
        #     objective='val_accuracy')

        tuner.search_space_summary()
        tuner.search(train_ds, validation_data=val_ds)
        tuner.results_summary()
        # best_model = tuner.get_best_models(1)[0]
        # best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        exit()

    # Training the model, check performances over validation set
    _ = model_class.training(train_ds=train_ds, epochs=arguments.epochs,  val_ds=val_ds, tb_logs=tb_logs,
                             tr_vl_steps=(tot_steps_tr, tot_steps_vl), fw_cm=(file_writer_cm_val, CLASS_NAMES))

    del train_ds, val_ds

    # --------------  FINAL TRAINING and TEST part  --------------------

    #  Use Dataset.map to create a dataset of image, label pairs
    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    lab_final_train_ds = final_training_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    lab_test_ds = test_paths_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    if arguments.caching:
        # delete previous cache files and store for this execution
        caching_file_base = main_path + "temp/"
        for f in os.listdir(caching_file_base):
            if "fin_tr.tfcache" in f or "test.tfcache" in f:
                os.remove(caching_file_base + f)
        fin_train_ds = prepare_for_training(lab_final_train_ds, cache=caching_file_base + "fin_tr.tfcache")
        test_ds = prepare_for_training(lab_test_ds, cache=caching_file_base + "test.tfcache")
    else:
        fin_train_ds = prepare_for_training(lab_final_train_ds)
        test_ds = prepare_for_training(lab_test_ds)

    # Initialize writers and callbacks for keeping track of results on Tensorboard
    file_writer_cm = tf.summary.create_file_writer(log_dir + "/cm_test")
    if logs_mode == "writers":
        tb_final_logs = tf.summary.create_file_writer(log_dir + "/final_train")
    elif logs_mode == "callbacks":
        tb_final_logs = callbacks.TensorBoard(log_dir=log_dir + "/final_train", update_freq='epoch')
    else:
        tb_final_logs = None

    tot_steps_te = (size_test // arguments.batch_size) + 1
    tot_steps_fintr = tot_steps_tr + tot_steps_vl

    # Train the model over the entire total_training set and then test
    model_class.test(test_ds=test_ds, train_ds=fin_train_ds, epochs=arguments.epochs, tb_logs=tb_final_logs,
                     tr_te_steps=(tot_steps_fintr, tot_steps_te), fw_cm=(file_writer_cm, CLASS_NAMES))

    #test_set = ['{}/{}'.format("/home/djack/local_repositories/grad-CAMalware/DATASETS/dataset1", name)
    #            for name in os.listdir("/home/djack/local_repositories/grad-CAMalware/DATASETS/dataset1")]
    #ind = 10
    #for ti in test_set:
    #    img, _ = process_path(ti)
    #    img = tf.expand_dims(img, 0)
    #    preds = model_class.model.predict(img)
    #    i = np.argmax(preds[0])
    #    print("lab: {} pred_vals: {} pred: {}".format(ti.split("/")[-1].split("_")[0], preds, i))
    #    ind -= 1
    #    if ind <= 0:
    #        break

    del fin_train_ds, test_ds

    # save model and architecture to single file
    if arguments.output_model is not None:
        model_class.model.save(main_path + 'model_saved/{}_m{}'.format(arguments.output_model, arguments.model))
        model_class.model.save_weights(main_path + 'model_saved/{}_m{}_weights'.format(arguments.output_model,
                                                                                          arguments.model))
        with open(main_path + 'model_saved/{}_m{}.info'.format(arguments.output_model, arguments.model), 'wb') \
                as filehandle:
            store_data = {"CLASS_NAMES": CLASS_NAMES, "CHANNELS": CHANNELS, "IMG_DIM": IMG_DIM}
            pickle.dump(store_data, filehandle)
        print_log("Model, Weights and Info saved to 'model_saved/{}_m{}[_weights|.info]'"
                  .format(arguments.output_model, arguments.model), print_on_screen=True)

        # Copy the tensorboard information folder into a backup folder (it keeps graph and cm)
        # print_log("AND tensorboard graphs/images stored to 'results/backup_tensorboard/{}".format(timeExec),
        #           print_on_screen=True)
        # copy_tree(log_dir, main_path + "results/backup_tensorboard/{}".format(timeExec))

    print_log("ENDING EXECUTION AT\t{}".format(time.strftime("%d-%m %H:%M:%S")), print_on_screen=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deep Learning Image Malware Classification')
    group = parser.add_argument_group('Arguments')
    # REQUIRED Arguments
    group.add_argument('-m', '--model', required=True, type=str,
                       help='[0] DATA, [1] BASIC_EXP, [2] VGG[16|19], [3] ResNet, [4] InceptionResNet, '
                            '[5] NASNet, [6] DenseNet')
    group.add_argument('-d', '--dataset', required=True, type=str,
                       help='the dataset to be used')
    # OPTIONAL Arguments
    group.add_argument('-o', '--output_model', required=False, type=str, default=None,
                       help='Name of model to output and store')
    group.add_argument('-e', '--epochs', required=False, type=int, default=20,
                       help='number of epochs')
    group.add_argument('-b', '--batch_size', required=False, type=int, default=32)
    group.add_argument('-i', '--image_size', required=False, type=str, default="250x1",
                       help='FORMAT ACCEPTED = SxC , the Size (SIZExSIZE) and channel of the images in input '
                            '(reshape will be applied)')
    group.add_argument('-t', '--tuning', required=False, type=str, default=None,
                       help='Run Keras Tuner for choosing hyperparameters [hyperband, random, bayesian]')
    group.add_argument('-w', '--weights', required=False, type=str, default=None,
                       help="If you do not want random initialization of the model weights "
                            "(ex. 'imagenet' or path to weights to be loaded)")
    # FLAGS
    group.add_argument('--monitoring', dest='monitoring', action='store_true',
                       help='Monitor train/val with confusion matrix per epoch (reduce speed)')
    group.set_defaults(monitoring=False)
    group.add_argument('--caching', dest='caching', action='store_true',
                       help='Caching dataset on file and loading per batches (IF db too big for memory)')
    group.set_defaults(caching=False)
    group.add_argument('--exclude_top', dest='include_top', action='store_false',
                       help='Exclute the fully-connected layer at the top pf the network (default INCLUDE)')
    group.set_defaults(include_top=True)
    arguments = parser.parse_args()
    return arguments


def _check_args(arguments):
    if arguments.model != "BASIC_EXP" \
            and arguments.model != "VGG16" and arguments.model != "VGG19" \
            and arguments.model != "ResNet" \
            and arguments.model != "InceptionResNet" \
            and arguments.model != "NASNet" \
            and arguments.model != "DenseNet" \
            and arguments.model != "DATA":
        print('Invalid model choice, exiting...')
        exit()
    if re.match(r"^\d{2,4}x([13])$", arguments.image_size):
        img_size = arguments.image_size.split("x")[0]
        channels = arguments.image_size.split("x")[1]
        setattr(arguments, "image_size", int(img_size))
        setattr(arguments, "channels", int(channels))
    else:
        print('Invalid image_size, exiting...')
        exit()
    if not os.path.isdir(main_path + arguments.dataset):
        print('Cannot find dataset in {}, exiting...'.format(arguments.dataset))
        exit()
    # Check Dataset struct: should be in folder tree training/[train|val] e test
    if not os.path.isdir(main_path + arguments.dataset + "/test") or \
            not os.path.isdir(main_path + arguments.dataset + "/training/val") or \
            not os.path.isdir(main_path + arguments.dataset + "/training/train"):
        print("Dataset '{}' should contain folders 'test, training/train and training/val'...".format(arguments.dataset))
        exit()
    if arguments.tuning is not None:
        if arguments.tuning != "random" and arguments.tuning != "bayesian" and arguments.tuning != "hyperband":
            print("INVALID option '' for tuning, allowed option are [random, bayesian, hyperband]', exiting..."
                  .format(arguments.tuning))
            exit()


def _model_selection(arguments, nclasses):
    model = None
    logs_mode = None
    print("INITIALIZING MODEL")
    if arguments.model == "BASIC_EXP":
        model = BASIC_EXP(nclasses, arguments.image_size, arguments.channels)
        logs_mode = "writers"
    elif arguments.model == "VGG16" or arguments.model == "VGG19":
        # NB. Setting include_top=True and thus accepting the entire struct, the input Shape MUST be 224x224x3
        # and in any case, channels has to be 3
        if arguments.channels != 3:
            print("VGG requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        model = VGG16_19(nclasses, arguments.image_size, arguments.channels,
                      weights=arguments.weights, include_top=arguments.include_top)
        logs_mode = "callbacks"
    elif arguments.model == "ResNet":
        # NB. Setting include_top=True and thus accepting the entire struct, the input Shape MUST be 224x224x3
        # and in any case, channels has to be 3
        if arguments.channels != 3:
            print("ResNet requires images with channels 3, please set --image_size <YOUR_IMAGE_SIZE>x3, exiting...")
            exit()
        model = ResNet(nclasses, arguments.image_size, arguments.channels,
                       weights=arguments.weights, include_top=arguments.include_top)
        logs_mode = "callbacks"
    else:
        print("model {} not implemented yet...".format(arguments.model))
        exit()

    return model, logs_mode


def get_label(file_path):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    # The second to last is the class-directory
    # cast to float32 for one_hot encode (otherwise TRUE/FALSE tensor)
    return tf.cast(parts[-2] == CLASS_NAMES, tf.float32)


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=CHANNELS)  # tf.image.decode_jpeg(img, channels=CHANNELS)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_DIM, IMG_DIM])


def process_path(file_path):
    label = get_label(file_path)
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img, label


def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000, loop=False):
    # IF it is a small dataset, only load it once and keep it in memory.
    # OTHERWISE use `.cache(filename)` to cache preprocessing work for datasets that don't fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    if loop:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


def build_model_tuning(hp):
    inputs = tf.keras.Input(shape=(250, 250, 1))
    x = tf.keras.layers.Conv2D(hp.Int('filters_1', 16, 128, step=16), (3, 3), activation='relu',
                        input_shape=(250, 250, 1))(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)

    for i in range(hp.Int('conv_blocks', 2, 5, default=3)):
        x = tf.keras.layers.Conv2D(hp.Int('filters_' + str(i), 32, 256, step=32), kernel_size=(3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        # if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
        #    x = tf.keras.layers.MaxPool2D()(x)
        # else:
        #    x = tf.keras.layers.AvgPool2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(hp.Float('dropout', 0, 0.7, step=0.1, default=0.5))(x)
    x = tf.keras.layers.Dense(hp.Int('hidden_size', 256, 1024, step=128, default=512),
                       activation=hp.Choice('act_1', ['relu', 'tanh']))(x)
    output = tf.keras.layers.Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-5, 1e-1)),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model


if __name__ == '__main__':
    start = time.perf_counter()
    args = parse_args()
    _check_args(args)

    # GLOBAL SETTINGS FOR THE EXECUTIONS
    # Reduce verbosity for Tensorflow Warnings and set dtype for layers
    # tf.keras.backend.set_floatx('float64')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    CHANNELS = args.channels
    IMG_DIM = args.image_size
    CLASS_NAMES = np.array([item.name for item in pathlib.Path(main_path + args.dataset + "/training/train").glob('*')])
    BATCH_SIZE = args.batch_size

    # Check if tensorflow can access the GPU
    device_name = tf.test.gpu_device_name()
    if not device_name:
        print('GPU device not found...')
        exit()
    print('Found GPU at: {}'.format(device_name))

    main(args)
    end = time.perf_counter()
    print()
    print("EX. TIME: {} ".format(str(datetime.timedelta(seconds=end-start))))

