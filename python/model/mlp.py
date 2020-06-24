
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



tf_parent_model = object

try:
    from sklearn.decomposition import PCA
    import numpy as np
except:
    print('error: cannot import required scikit-learn or numpy 7modules', e)

try:
    import maya.cmds as mc
    maya_env = True
except:
    maya_env = False

# for some reason importing modules below inside maya causes issues
if not maya_env:
    try:
        import tensorflow as tf
        from tensorflow.keras.layers import Dense
        from tensorflow.keras.layers import Activation
        from tensorflow.keras.layers import Dropout
        from tensorflow.keras.layers import BatchNormalization
        from tensorflow.keras.regularizers import l1, l2
        tf_parent_model = tf.keras.Model
    except ImportError as e:
        print('error: cannot import required tensorflow modules', e)


class DeformNetModelTF(tf_parent_model):
    """
    A fully connected MLP network defined using tensorflow
    :param output_name: output name for the network
    :param input_name: input name for the network
    :param layer_count: how many layers are in the network
    """

    TOL = 0.001

    def __init__(self,
                 output_dimension,
                 unit_count=512,
                 activation='relu',
                 pca=PCA(0),
                 dropout=0.5,
                 layer_count=4,
                 batch_norm=True,
                 regularizer=None,
                 regularizer_l=0.1):
        """
        init function
        :param int output_dimension: dimension for output data
        :param int unit_count: number of nodes in a layer
        :param str activation: activation function for hidden layers
        :param PCA pca: object representing principal component analysis result on output
        :param float dropout: dropout ratio for network
        :param int layer_count: number of layers in the network
        :param bool batch_norm: whether to perform batch normalization
        :param str regularizer: what regularizer to use for Dense layers
        :param float regularizer_l: the l parameter for the regularizer
        """
        super(DeformNetModelTF, self).__init__(name='DeformNetModelTF')

        if regularizer == 'l1':
            regularizer = l1(regularizer_l)
        elif regularizer == 'l2':
            regularizer = l2(regularizer_l)
        else:
            regularizer = None

        self._input_layer = Dense(unit_count, input_shape=(1,), kernel_regularizer=regularizer)

        if batch_norm:
            self._input_batch_norm = BatchNormalization()
        else:
            self._input_batch_norm = None

        self._input_activation = Activation(activation)

        if dropout > self.TOL:
            self._input_dropout = Dropout(dropout)
        else:
            self._input_dropout = None

        self._hidden_layers = []
        self._hidden_lyr_dropouts = []
        self._hidden_lyr_activations = []
        self._hidden_lyr_batch_norms = []

        for i in range(layer_count-2):
            self._hidden_layers.append(Dense(unit_count, kernel_regularizer=regularizer))
            if batch_norm:
                self._hidden_lyr_batch_norms.append(BatchNormalization())
            self._hidden_lyr_activations.append(Activation(activation))
            if dropout > self.TOL:
                self._hidden_lyr_dropouts.append(Dropout(dropout))

        self._output_layer = Dense(output_dimension, kernel_regularizer=regularizer)
        self._pca = pca

        self.output_name = None
        self.input_name = None
        self.layer_count = layer_count

    def call(self, input_tensor):
        self.input_name = input_tensor.name
        x = self._input_layer(input_tensor)
        if self._input_batch_norm is not None:
            x = self._input_batch_norm(x)
        x = self._input_activation(x)

        if self._input_dropout:
            x = self._input_dropout(x)

        for i in range(self.layer_count-2):
            x = self._hidden_layers[i](x)
            if self._hidden_lyr_batch_norms:
                x = self._hidden_lyr_batch_norms[i](x)
            x = self._hidden_lyr_activations[i](x)
            if self._hidden_lyr_dropouts:
                x = self._hidden_lyr_dropouts[i](x)

        x = self._output_layer(x)

        components = tf.cast(self._pca.components_, tf.float32)
        x = tf.matmul(x - self._pca.mean_, components, transpose_a=False, transpose_b=True)
        x = tf.matmul(x, components) + self._pca.mean_

        self.output_name = x.name
        return x


def get_concatenated_model(num_of_inputs, model_input_dimensions, model_input_constant, unit_count=128, activation='relu', layer_count=4):
    """
    generate a model that concatenates several small models together
    :param int num_of_inputs: the dimension of the input vector
    :param list model_input_dimensions: a list of input dimensions for the models
    :param int unit_count: how many units per layer
    :param str activation: activation function
    :param int layer_count: how many layers
    :return: a model for training and inference
    :rtype: tf.keras.Model
    """

    ipt = tf.keras.Input(shape=(num_of_inputs, ))
    outputs = []
    #model_input_dimensions = model_input_dimensions[10:20]
    for dimensions in model_input_dimensions:
        inputs = []
        # for points with no secondary deformation, put in a constant 0 value
        if not dimensions:
            cur_output = tf.keras.layers.Lambda(lambda x: model_input_constant + 0.0 * x[:, 0:3])(ipt)
            outputs.append(cur_output)
            continue
        for dimension in dimensions:
            start, end = dimension
            cur_input = tf.keras.layers.Lambda(lambda x: x[:, start:end])(ipt)
            inputs.append(cur_input)
        if len(inputs) > 1:
            input_concat = tf.keras.layers.Concatenate()(inputs)
        else:
            input_concat = inputs[0]
        x = input_concat
        for i in range(layer_count-1):
            x = Dense(unit_count, )(x)
            x = Activation(activation)(x)
        cur_output = Dense(3)(x)
        outputs.append(cur_output)

    full_model = tf.keras.Model(inputs=ipt, outputs=outputs)
    full_model.input_name = ipt.name
    full_model.output_name = ','.join([i.name for i in outputs])
    print(full_model.summary())
    return full_model

