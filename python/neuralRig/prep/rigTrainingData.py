
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


"""
module for managing training data for fast rig eval
"""
import os
import shutil
import pandas
import ml.trainingData as td
import ml.neuralRig.prep.dataPrep as dataPrep


class RigTrainingData(td.TrainingData):
    """
    class for managing training data for fast rig eval
    """
    FILE_SET = ('controls', 'moverValues', 'differentialOffset', 'localOffset',
                'jointLocalMatrix', 'anchorPoints')

    def size(self):
        """
        find out the training data size
        :return: data set size
        :rtype: int
        """
        mover_files = [f for f in self.files if f.find('moverValues') > -1]
        return len(mover_files)

    def get_mover_data(self, batch_num):
        """
        retrieve mover data
        :param int batch_num: the batch number on the file name
        :return: mover data
        :rtype: pandas.DataFrame
        """
        fl = os.path.join(self.path, 'moverValues%i.pkl' % batch_num)
        return pandas.read_pickle(fl)

    def get_joint_data(self, batch_num):
        """
        retrieve joint data
        :param int batch_num: the batch number on the file name
        :return: joint data
        :rtype: pandas.DataFrame
        """
        fl = os.path.join(self.path, 'jointLocalMatrix%i.pkl' % batch_num)
        return pandas.read_pickle(fl)

    def get_control_data(self, batch_num):
        """
        retrieve numerical control data
        :param int batch_num: the batch number on the file name
        :return: control data
        :rtype: pandas.DataFrame
        """
        fl = os.path.join(self.path, 'controls%i.pkl' % batch_num)
        if os.path.exists(fl):
            return pandas.read_pickle(fl)
        else:
            return None

    def get_differential_data(self, batch_num):
        """
        retrieve differential data
        :param int batch_num: the batch number on the file name
        :return: differential vertex data
        :rtype: pandas.DataFrame
        """
        fl = os.path.join(self.path, 'differentialOffset%i.pkl' % batch_num)
        return pandas.read_pickle(fl)

    def get_anchor_data(self, batch_num):
        """
        retrieve anchor points data
        :param int batch_num: the batch number on the file name
        :return: anchor points data
        :rtype: pandas.DataFrame
        """
        fl = os.path.join(self.path, 'anchorPoints%i.pkl' % batch_num)
        return pandas.read_pickle(fl)

    def get_local_offset_data(self, batch_num):
        """
        retrieve local offsets data
        :param int batch_num: the batch number on the file name
        :return: local offset data
        :rtype: pandas.DataFrame
        """
        fl = os.path.join(self.path, 'localOffset%i.pkl' % batch_num)
        return pandas.read_pickle(fl)

    def get_world_offset_data(self, batch_num):
        """
        retrieve local offsets data
        :param int batch_num: the batch number on the file name
        :return: local offset data
        :rtype: pandas.DataFrame
        """
        fl = os.path.join(self.path, 'worldOffset%i.pkl' % batch_num)
        return pandas.read_pickle(fl)

    def get_data(self, data_type, batch_num):
        """
        retrieve given type of data
        :param str data_type: data type. Valid options are differential, anchor and local_offset
        :param int batch_num: the batch number on the file name
        :return: requested data
        :rtype: pandas.DataFrame
        """
        if data_type == 'differential':
            return self.get_differential_data(batch_num)
        elif data_type == 'anchor':
            return self.get_anchor_data(batch_num)
        elif data_type == 'local_offset':
            return self.get_local_offset_data(batch_num)
        elif data_type == 'world_offset':
            return self.get_world_offset_data(batch_num)
        else:
            raise ValueError("Given data type %s is not in current data set" % data_type)





