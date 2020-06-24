
#
# I. LICENSE CONDITIONS
#
# Copyright (c) 2019 by Blue Sky Studios, Inc.
# Permission is hereby granted to use this software solely for non-commercial
# applications and purposes including academic or industrial research,
# evaluation and not-for-profit media production. All other rights are retained
# by Blue Sky Studios, Inc. For use for or in connection with commercial
# applications and purposes, including without limitation in or in connection
# with software products offered for sale or for-profit media production,
# please contact Blue Sky Studios, Inc. at
#  tech-licensing@blueskystudios.com<mailto:tech-licensing@blueskystudios.com>.
#
# THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
# EVENT SHALL BLUE SKY STUDIOS, INC. OR ITS AFFILIATES BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import numpy as np
import shutil
import os
import pandas
from collections import namedtuple
import matplotlib.pyplot as plt
import ml.neuralRig.prep.rigTrainingData as rigTrainingData
import logging
import datetime
import json

# this should probably be part of the saved config file:
try:
    import maya.cmds as mc
    maya_env = True
except:
    maya_env = False

try:
    from sklearn.decomposition import PCA
    import ml.model.mlp as mlp
except ImportError as e:
    print('error: cannot import required basic modules', e)

try:
    import tensorflow as tf
    from tensorflow.keras.optimizers import SGD, Adam
    from tensorflow.keras.losses import Huber
    from tensorflow.python.tools import freeze_graph
    from tensorflow.tools.graph_transforms import TransformGraph
except ImportError as e:
    print('error: cannot import required tensorflow modules', e)


try:
    import configparser
except ImportError as e:
    # python 2 backwards compatibility
    import ConfigParser as configparser

logger = logging.getLogger(__name__)


TFModel = namedtuple('TFModel',
                     ['graph', 'session', 'input_tensor', 'output_tensor',
                      'normalized', 'trans_max', 'trans_min', 'verts_max', 'verts_min'])


class TrainingSession(object):
    """
    a class for setting up training
    :param str config: config file path for current training session
    :param dict training_params: parameters for training
    :param str output_path: where to save the training result
    :param rigTrainingData.RigTrainingData training_data: object that represents the training data
    :param list training_set: a list of integers indicating which files should be used for training
    :param list evaluation_set: a list of integers indicating which files should be used for evaluation
    :param np.ndarray joints: array of joints as input for the network
    :param np.ndarray controls: array of numeric controls as input for the network
    :param dict offset_data: a dictionary containing different kinds of label data
    :param np.ndarray inputs: all network inputs combined
    :param dict models: a dictionary contains all the model information used for evaluation
    :param dict column_dict: a dictionary contains info for column names of different training types
    """

    FRAMEWORK_TYPES = ['tensorflow']

    #: training types this training session can handle
    TRAINING_TYPES = ['differential', 'anchor', 'local_offset', 'world_offset']

    #: network param names that are of type int
    INT_PARAMS = ['epoch', 'batch_size', 'pc_count', 'unit_count', 'layer_count']

    #: network param names that are of type float
    FLOAT_PARAMS = ['regularizer_decay', 'dropout', 'momentum', 'learning_rate_decay', 'learning_rate', 'min', 'max',
                    'joint_translate_min', 'joint_translate_max', 'loss', 'prediction_error']

    #: network param names that are of type bool
    BOOLEAN_PARAMS = ['normalize', 'batch_norm']

    def __init__(self, config):
        """
        initialize session
        :param config: config file path, should be in .ini format
        """
        if not os.path.exists(config):
            raise IOError("%s doesn't exist" % config)
        self.config = config
        self.training_params = {}
        for training_type in self.TRAINING_TYPES:
            self.training_params['%s_read' % training_type] = False
        self.output_path = ''
        self.training_data = rigTrainingData.RigTrainingData()
        self.evaluation_data = None
        self.training_set = None
        self.evaluation_set = None
        self.joints = None
        self.controls = None
        self.anchorids = None   # translated to column numbers from ids (ids may not be sequential nor continuous!)
        self.idxlate = None
        self.joint_names = None
        self.control_names = None
        self.offset_data = {}
        self.inputs = None
        self._config_handle = None
        self.models = {}
        self.column_dict = {}
        self.parse_config()

    def __get_network_param(self, param, training_type):
        full_param_name = '%s_%s' % (training_type, param)
        exists = self._config_handle.has_option('network_parameters', full_param_name)
        if not exists:
            logger.info('Warning: %s is not defined in the config file %s' % (full_param_name, self.config))
        if param in self.INT_PARAMS:
            if not exists:
                self.training_params[full_param_name] = 0
            else:
                self.training_params[full_param_name] = self._config_handle.getint('network_parameters',
                                                                                   full_param_name)
        elif param in self.FLOAT_PARAMS:
            if not exists:
                self.training_params[full_param_name] = 0.0
            else:
                self.training_params[full_param_name] = self._config_handle.getfloat('network_parameters',
                                                                                     full_param_name)
        elif param in self.BOOLEAN_PARAMS:
            if not exists:
                self.training_params[full_param_name] = False
            else:
                self.training_params[full_param_name] = self._config_handle.getboolean('network_parameters',
                                                                                       full_param_name)
        else:
            if not exists:
                self.training_params[full_param_name] = ''
            else:
                self.training_params[full_param_name] = self._config_handle.get('network_parameters', full_param_name)

    def parse_config(self):
        """parse the config file and get all the training parameters
        """
        self._config_handle = configparser.ConfigParser()
        self._config_handle.read(self.config)

        for training_type in self.TRAINING_TYPES:
            self.__get_network_param('epoch', training_type)
            self.__get_network_param('batch_size', training_type)
            self.__get_network_param('regularizer', training_type)
            self.__get_network_param('regularizer_decay', training_type)
            self.__get_network_param('pc_count', training_type)
            self.__get_network_param('activation_func', training_type)
            self.__get_network_param('unit_count', training_type)
            self.__get_network_param('layer_count', training_type)
            self.__get_network_param('dropout', training_type)
            self.__get_network_param('optimizer', training_type)
            self.__get_network_param('batch_norm', training_type)

            if self.training_params['%s_optimizer' % training_type] == 'SGD':
                self.__get_network_param('momentum', training_type)
                self.__get_network_param('learning_rate_decay', training_type)
            self.__get_network_param('learning_rate', training_type)
            self.__get_network_param('normalize', training_type)
            self.__get_network_param('loss_metric', training_type)
            self.__get_network_param('val_metric', training_type)
            self.__get_network_param('network_meta', training_type)
            self.__get_network_param('network_input', training_type)
            self.__get_network_param('network_output', training_type)
            self.__get_network_param('min', training_type)
            self.__get_network_param('max', training_type)
            self.__get_network_param('joint_translate_min', training_type)
            self.__get_network_param('joint_translate_max', training_type)
            self.__get_network_param('control_mapping', training_type)
            self.__get_network_param('framework', training_type)
            self.__get_network_param('loss', training_type)
            self.__get_network_param('prediction_error', training_type)

        if self._config_handle.has_option('input_parameters', 'evaluation_data_path'):
            self.evaluation_data = rigTrainingData.RigTrainingData()
            self.evaluation_data.path = self._config_handle.get('input_parameters', 'evaluation_data_path')
        else:
            self.evaluation_data = self.training_data

        self.training_data.path = self._config_handle.get('input_parameters', 'training_data_path')
        training_set = self._config_handle.get('input_parameters', 'training_data_set')
        tokens = training_set.split('-')
        if len(tokens) == 1:
            self.training_set = [int(tokens[0])]
        else:
            self.training_set = range(int(tokens[0]), int(tokens[-1]) + 1)
        evaluation_set = self._config_handle.get('input_parameters', 'evaluation_data_set')
        tokens = evaluation_set.split('-')
        if len(tokens) == 1:
            self.evaluation_set = [int(tokens[0])]
        else:
            self.evaluation_set = range(int(tokens[0]), int(tokens[-1]) + 1)

        self.output_path = self._config_handle.get('output_parameters', 'output_path')
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def read_training_data(self, training_types=['differential', 'anchor', 'local_offset'], evaluation=False):
        """read in training data
        :param list training_types: list of training types. Only reading in data specified in training types
        :param bool evaluation: if True, does not read in label data, and only reads in input data from evaluation set
        """
        data_size = self.training_data.size()
        if evaluation:
            data_set = self.evaluation_set
            cur_data = self.evaluation_data
        else:
            data_set = self.training_set
            cur_data = self.training_data
        max_file_num = max(data_set)
        if max_file_num >= data_size:
            raise ValueError("Training data only has %i files. Can't read in file %i" % (data_size, max_file_num))

        self.joints = None
        self.controls = None
        self.joint_names = None
        self.control_names = None
        dfs = {}
        data_arrays = {}
        for index, i in enumerate(data_set):
            df_joint = cur_data.get_joint_data(i)
            df_control = cur_data.get_control_data(i)

            self.joint_names = list(df_joint.columns)
            if df_control is not None:
                self.control_names = list(df_control.columns)
            else:
                self.control_names = []

            for training_type in training_types:
                dfs[training_type] = cur_data.get_data(data_type=training_type, batch_num=i)
            logging.info("Finish loading batch #%i" % i)

            # convert dataFrame to a flat numpy array
            joints = df_joint.to_numpy()
            joints = joints.tolist()
            joints = np.array(joints, dtype=np.float32)
            logging.info("Joint shape: %s" % str(joints.shape))  # [samples, joints number, dimension]

            if df_control is not None:
                controls = df_control.to_numpy()
                controls = controls.tolist()
                controls = np.array(controls, dtype=np.float32)
                logging.info("control shape: %s" % str(controls.shape))  # [samples, joints number, dimension]
            else:
                controls = None

            for training_type in training_types:
                self.column_dict[training_type] = list(dfs[training_type].columns)
                data_arrays[training_type] = dfs[training_type].to_numpy()
                data_arrays[training_type] = data_arrays[training_type].tolist()
                data_arrays[training_type] = np.array(data_arrays[training_type], dtype=np.float32)
                # [samples, vertices number, dimension]
                logging.info("%s data shape: %s" % (training_type, str(data_arrays[training_type].shape)))

            if index == 0:
                self.joints = joints
                self.controls = controls
                for training_type in training_types:
                    self.offset_data[training_type] = data_arrays[training_type]
                    if training_type == 'anchor' and not evaluation:
                        self.anchorids = dfs[training_type].keys()
                        self.anchorids = sorted(self.anchorids)
                        if self.idxlate:
                            self.anchorids = [self.idxlate[x] for x in self.anchorids]
                    if training_type in {'local_offset', 'world_offset'} and not evaluation:
                        # map from any id to the place in the array:
                        self.idxlate = {a: i for i, a in enumerate(self.column_dict[training_type])}
                        if self.anchorids:
                            self.anchorids = [self.idxlate[x] for x in self.anchorids]
            else:
                self.joints = np.concatenate((self.joints, joints), axis=0)
                if controls is not None:
                    self.controls = np.concatenate((self.controls, controls), axis=0)
                for training_type in training_types:
                    self.offset_data[training_type] = np.concatenate((self.offset_data[training_type],
                                                                      data_arrays[training_type]),
                                                                     axis=0)
        for training_type in training_types:
            self.training_params['%s_read' % training_type] = True

    @staticmethod
    def _normalize_features(df, df_max=None, df_min=None):
        if df_max is None:
            df_max = df.max()
        if df_min is None:
            df_min = df.min()
        df_norm = (df - df_min) / (df_max - df_min)
        df_norm = np.nan_to_num(df_norm)
        return df_norm, df_max.tolist(), df_min.tolist()

    @staticmethod
    def _denormalize_features(df, df_max, df_min):
        df = (df * (df_max - df_min)) + df_min
        df = np.nan_to_num(df)
        return df

    def prep_data(self, training_type, evaluation=False):
        """
        prep data for training. Performing normalization, pca, etc
        :param str training_type: type of training
        :param bool evaluation: if True, operates in evaluation mode
        """
        if self.training_params['%s_normalize' % training_type]:
            if evaluation:
                model = self.models[training_type]
                df_min = model.verts_min
                df_max = model.verts_max
                normalize_result = self._normalize_features(self.offset_data[training_type],
                                                            df_max=df_max,
                                                            df_min=df_min)
                self.offset_data[training_type], df_max, df_min = normalize_result
            else:
                normalize_result = self._normalize_features(self.offset_data[training_type])
                self.offset_data[training_type], df_max, df_min = normalize_result
                self.training_params['%s_min' % training_type] = df_min
                self.training_params['%s_max' % training_type] = df_max

            joints = np.copy(self.joints)
            joint_translate = joints[:, :, -3:]
            if evaluation:
                model = self.models[training_type]
                trans_min = model.trans_min
                trans_max = model.trans_max
                joint_translate, trans_max, trans_min = self._normalize_features(joint_translate,
                                                                                 df_max=trans_max,
                                                                                 df_min=trans_min)
            else:
                joint_translate, trans_max, trans_min = self._normalize_features(joint_translate)
                self.training_params['%s_joint_translate_min' % training_type] = trans_min
                self.training_params['%s_joint_translate_max' % training_type] = trans_max
            joints[:, :, -3:] = joint_translate

        self.inputs = joints.reshape(self.joints.shape[0], -1)
        if self.controls is not None:
            controls = self.controls.reshape(self.controls.shape[0], -1)
            self.inputs = np.concatenate((self.inputs, controls), axis=1)

    def make_plot(self, history, path, training_type):
        """
        util function to generate a graph based on training error
        :param tf.keras.History history: training history
        :param str path: where to save the image
        :param str training_type: the type of the training
        """
        if self.training_params['%s_loss_metric' % training_type] == 'huber':
            plt.plot(history.history['mean_squared_error'])
        else:
            plt.plot(history.history[self.training_params['%s_loss_metric' % training_type]])
        plt.plot(history.history[self.training_params['%s_val_metric' % training_type]])
        plt.ylabel(self.training_params['%s_loss_metric' % training_type])
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(path)

    def train(self, training_type, cpu=False, log=False):
        """
        function to train the network
        :param str training_type: what type of training this is. Valid options are differential, anchor, local_offset
        :param bool cpu: if True, use cpu for training
        :param bool log: if True, generates log for TensorBoard
        """
        if not self.training_params['%s_read' % training_type]:
            self.read_training_data([training_type])
        self.prep_data(training_type)
        self._train(training_type=training_type, cpu=cpu, log=log)

    def _train(self, training_type, cpu=False, log=False):
        outputs = self.offset_data[training_type].reshape(self.offset_data[training_type].shape[0], -1)

        df_min = self.training_params['%s_min' % training_type]
        df_max = self.training_params['%s_max' % training_type]
        normalized_zero = -1 * df_min / (df_max - df_min)

        logger.info('input dimension: %s, %s output dimension: %s' %
                    (str(self.inputs.shape), training_type, str(outputs.shape)))

        pca = PCA(n_components=self.training_params['%s_pc_count' % training_type])
        pca.fit(outputs)

        if cpu:
            tf_config = tf.compat.v1.ConfigProto(device_count={'GPU': 0})
        else:
            tf_config = tf.compat.v1.ConfigProto()

        with tf.compat.v1.Session(graph=tf.Graph(), config=tf_config) as session:
            activation = self.training_params['%s_activation_func' % training_type]

            if self.training_params['%s_control_mapping' % training_type]:
                outputs = outputs.reshape(self.offset_data[training_type].shape[0], -1, 3)
                temp_outputs = outputs.swapaxes(0, 1)
                outputs = []
                for temp in temp_outputs:
                    outputs.append(temp)
                with open(self.training_params['%s_control_mapping' % training_type], 'r') as file_handle:
                    input_dimensions = []
                    control_mapping = json.load(file_handle)
                    for vtx in self.column_dict[training_type]:
                        cur_dimensions = []
                        input_indices = []
                        if str(vtx) not in control_mapping:
                            print("vertex %i doesn't have secondary deformation" % vtx)
                            input_dimensions.append([])
                            continue
                        controls = control_mapping[str(vtx)]
                        for control in controls:
                            if control in self.joint_names:
                                cur_index = self.joint_names.index(control)
                                for i in range(12):
                                    input_indices.append(12*cur_index + i)
                            else:
                                cur_index = self.control_names.index(control)
                                input_indices.append(12*len(self.joint_names) + cur_index)

                        input_indices.sort()

                        start = input_indices[0]
                        previous = start
                        for index in input_indices[1:]:
                            if index > previous + 1:
                                end = previous + 1
                                cur_dimensions.append((start, end))
                                start = index
                            previous = index
                        cur_dimensions.append((start, previous+1))
                        input_dimensions.append(cur_dimensions)

                model = mlp.get_concatenated_model(12*len(self.joint_names) + len(self.control_names),
                                                    input_dimensions,
                                                    normalized_zero,
                                                    unit_count=self.training_params['%s_unit_count' % training_type],
                                                    activation=activation,
                                                    layer_count=self.training_params['%s_layer_count' % training_type])

            else:
                model = mlp.DeformNetModelTF(output_dimension=outputs.shape[1],
                                             unit_count=self.training_params['%s_unit_count' % training_type],
                                             activation=activation,
                                             pca=pca,
                                             dropout=self.training_params['%s_dropout' % training_type],
                                             layer_count=self.training_params['%s_layer_count' % training_type],
                                             batch_norm=self.training_params['%s_batch_norm' % training_type],
                                             regularizer=self.training_params['%s_regularizer' % training_type],
                                             regularizer_l=self.training_params['%s_regularizer_decay' % training_type])

            if self.training_params['%s_optimizer' % training_type] == 'SGD':
                optimizer = SGD(lr=self.training_params['%s_learning_rate' % training_type],
                                decay=self.training_params['%s_learning_rate_decay' % training_type],
                                momentum=self.training_params['%s_momentum' % training_type],
                                nesterov=True)
            else:
                optimizer = Adam(lr=self.training_params['%s_learning_rate' % training_type], epsilon=1e-08)

            loss_func = self.training_params['%s_loss_metric' % training_type]

            if loss_func == 'huber':
                loss_func = Huber(delta=0.01)
                metrics = ['mean_squared_error']
            else:
                metrics = [loss_func]

            model.compile(loss=loss_func,
                          optimizer=optimizer,
                          metrics=metrics)

            export_path = os.path.join(self.output_path, training_type)
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            start_time = datetime.datetime.now()
            self.training_params['%s_start_time' % training_type] = start_time.strftime('%m/%d/%Y, %H:%M:%S')

            if log:
                log_dir = os.path.join(export_path, 'log')
                tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
                history = model.fit(self.inputs,
                                    outputs,
                                    epochs=self.training_params['%s_epoch' % training_type],
                                    batch_size=self.training_params['%s_batch_size' % training_type],
                                    validation_split=0.1,
                                    shuffle=True,
                                    verbose=2,
                                    callbacks=[tensorboard_callback])
            else:
                history = model.fit(self.inputs,
                                    outputs,
                                    epochs=self.training_params['%s_epoch' % training_type],
                                    batch_size=self.training_params['%s_batch_size' % training_type],
                                    validation_split=0.1,
                                    verbose=2,
                                    shuffle=True)

            end_time = datetime.datetime.now()
            loss = history.history['loss'][-1]
            validation_loss = history.history['val_loss'][-1]
            plot_image = os.path.join(export_path, '%s.png' % training_type)
            try:
                self.make_plot(history, plot_image, training_type)
            except:
                pass

            # Save the keras model as a tensorflow model
            saver = tf.compat.v1.train.Saver(save_relative_paths=True)
            saver.save(session, os.path.join(export_path, training_type))

        # Save the training parameters to the output dir:
        self.training_params['%s_network_meta' % training_type] = os.path.join(export_path, training_type + '.meta')

        self.training_params['%s_network_input' % training_type] = model.input_name
        self.training_params['%s_network_output' % training_type] = model.output_name

        self.training_params['%s_start_time' % training_type] = start_time.strftime('%m/%d/%Y, %H:%M:%S')
        self.training_params['%s_end_time' % training_type] = end_time.strftime('%m/%d/%Y, %H:%M:%S')
        self.training_params['%s_training_time' % training_type] = str(end_time - start_time)

        config_record = os.path.join(self.output_path, 'config.ini')
        if not os.path.exists(config_record):
            shutil.copy(self.config, config_record)
        config_handle = configparser.ConfigParser()
        config_handle.read(config_record)
        config_handle.set('network_parameters',
                          '%s_network_meta' % training_type,
                          self.training_params['%s_network_meta' % training_type])
        config_handle.set('network_parameters',
                          '%s_network_input' % training_type,
                          self.training_params['%s_network_input' % training_type])
        config_handle.set('network_parameters',
                          '%s_network_output' % training_type,
                          self.training_params['%s_network_output' % training_type])
        config_handle.set('network_parameters',
                          '%s_min' % training_type,
                          str(self.training_params['%s_min' % training_type]))
        config_handle.set('network_parameters',
                          '%s_max' % training_type,
                          str(self.training_params['%s_max' % training_type]))
        config_handle.set('network_parameters',
                          '%s_joint_translate_min' % training_type,
                          str(self.training_params['%s_joint_translate_min' % training_type]))
        config_handle.set('network_parameters',
                          '%s_joint_translate_max' % training_type,
                          str(self.training_params['%s_joint_translate_max' % training_type]))
        config_handle.set('network_parameters',
                          '%s_start_time' % training_type,
                          str(self.training_params['%s_start_time' % training_type]))
        config_handle.set('network_parameters',
                          '%s_end_time' % training_type,
                          str(self.training_params['%s_end_time' % training_type]))
        config_handle.set('network_parameters',
                          '%s_training_time' % training_type,
                          str(self.training_params['%s_training_time' % training_type]))
        config_handle.set('network_parameters',
                          '%s_loss' % training_type,
                          str(loss))
        config_handle.set('network_parameters',
                          '%s_validation_loss' % training_type,
                          str(validation_loss))

        f = open(config_record, 'w')
        config_handle.write(f)
        f.close()
        model.summary()

    def load_model(self, training_type):
        """
        load the tensorflow model
        :param str training_type: what type of training this is. Valid options are differential, anchor, local_offset
        """

        graph = tf.Graph()
        config = tf.compat.v1.ConfigProto(
            device_count={'GPU': 0}
        )
        with graph.as_default():
            session = tf.compat.v1.Session(config=config)
            with session.as_default():
                meta = self.training_params['%s_network_meta' % training_type]
                root = meta.rsplit(os.sep, 1)[0]

                if not os.path.exists(meta):
                    raise ValueError('Could not find meta file: %s' % meta)

                saver = tf.train.import_meta_graph(meta)
                saver.restore(session, tf.train.latest_checkpoint(root))

                in_tensor = session.graph.get_tensor_by_name(self.training_params['%s_network_input' % training_type])
                if self.training_params['%s_network_output' % training_type].find(',') > -1:
                    tokens = self.training_params['%s_network_output' % training_type].split(',')
                    out_tensor = []
                    for token in tokens:
                        out_tensor.append(session.graph.get_tensor_by_name(token))
                else:
                    out_tensor = session.graph.get_tensor_by_name(self.training_params['%s_network_output' % training_type])

                normalized = self.training_params['%s_normalize' % training_type]
                verts_max, verts_min, trans_max, trans_min = None, None, None, None
                if normalized:
                    trans_max = np.array(self.training_params['%s_joint_translate_max' % training_type], dtype=np.float32)
                    trans_min = np.array(self.training_params['%s_joint_translate_min' % training_type], dtype=np.float32)
                    verts_max = np.array(self.training_params['%s_max' % training_type], dtype=np.float32)
                    verts_min = np.array(self.training_params['%s_min' % training_type], dtype=np.float32)

                tf_model = TFModel(graph=session.graph,
                                   session=session,
                                   input_tensor=in_tensor,
                                   output_tensor=out_tensor,
                                   normalized=normalized,
                                   trans_max=trans_max,
                                   trans_min=trans_min,
                                   verts_max=verts_max,
                                   verts_min=verts_min)

                self.models[training_type] = tf_model

    def evaluate(self, training_types):
        """
        evaluate the model on evaluation data set
        :param list training_types: what types of evaluation. Valid options are differential, anchor, local_offset
        """
        self.read_training_data(evaluation=True)

        config_record = os.path.join(self.output_path, 'config.ini')
        if not os.path.exists(config_record):
            raise RuntimeError("Could not locate config file")

        config_handle = configparser.ConfigParser()
        config_handle.read(config_record)

        for training_type in training_types:
            self.load_model(training_type)
            self.prep_data(training_type, evaluation=True)

            model = self.models[training_type]

            columns = self.column_dict[training_type]
            data_dict = {}

            with model.graph.as_default():
                with model.session.as_default():
                    results = model.session.run(model.output_tensor,
                                                feed_dict={model.input_tensor: self.inputs})
                    if isinstance(model.output_tensor, list):
                        results = np.array(results)
                        results = results.swapaxes(0, 1)
                    else:
                        results = results.reshape(results.shape[0], -1, 3)
                    mean_square_error = np.square(results - self.offset_data[training_type]).mean()

                    if model.normalized:
                        results = self._denormalize_features(results,
                                                             df_max=model.verts_max,
                                                             df_min=model.verts_min)
                    results = np.swapaxes(results, 0, 1)
                    for index, data in enumerate(results):
                        data_dict[columns[index]] = list(data)

                    data_frame = pandas.DataFrame(data_dict)

            prediction_file = os.path.join(self.output_path, training_type, 'prediction.json')
            data_frame.to_json(prediction_file)

            config_handle.set('network_parameters',
                              '%s_prediction_error' % training_type,
                              str(mean_square_error))

        f = open(config_record, 'w')
        config_handle.write(f)
        f.close()

    def save_pb_file(self, training_types):
        """
        saves a .pb file for inference
        :param list training_types: what types of evaluation. Valid options are differential, anchor, local_offset
        """

        for training_type in training_types:
            self.load_model(training_type)
            export_path = os.path.join(self.output_path, training_type)

            input_node = self.training_params['%s_network_input' % training_type].split(':')[0]
            output_nodes = []
            if self.training_params['%s_network_output' % training_type].find(',') > -1:
                tokens = self.training_params['%s_network_output' % training_type].split(',')
                for token in tokens:
                    output_node = token.split(':')[0]
                    output_nodes.append(output_node)
            else:
                output_node = self.training_params['%s_network_output' % training_type].split(':')[0]
                output_nodes.append(output_node)

            model = self.models[training_type]
            graph = model.graph

            with graph.as_default():
                freeze_var_names = list(set(v.op.name for v in tf.global_variables()))
                output_names = output_nodes
                output_names += [v.op.name for v in tf.global_variables()]
                input_graph_def = graph.as_graph_def()
                for node in input_graph_def.node:
                    node.device = ""
                frozen_graph = tf.graph_util.convert_variables_to_constants(model.session,
                                                                            input_graph_def,
                                                                            output_names,
                                                                            freeze_var_names)

                transforms = ['strip_unused_nodes(type=float, shape="1,299,299,3")',
                              'remove_nodes(op=Identity, op=CheckNumerics)',
                              'fold_constants(ignore_errors=true)',
                              'fold_batch_norms']

                optimized_graph = TransformGraph(frozen_graph, [input_node], output_nodes, transforms)
                tf.io.write_graph(optimized_graph, export_path, 'model.pb', as_text=False)

