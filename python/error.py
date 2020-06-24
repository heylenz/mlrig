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



import numpy


class ErrorHistory(object):
    """
    a class to track prediction errors
    """

    ERROR_METRICS = ['mean_square_error', 'max_square_error', 'min_square_error',
                     'mean_absolute_error', 'max_absolute_error', 'min_absolute_error']

    def __init__(self):
        """
        init function
        """
        self.history = {}
        for metric in self.ERROR_METRICS:
            self.history[metric] = []

    def add_sample(self, ground_truth, prediction):
        """
        add a sample, and calculate related errors
        :param numpy.ndarray ground_truth: data representing ground truth
        :param numpy.ndarray prediction: data representing prediction
        """
        absolute_value = numpy.absolute(prediction - ground_truth)
        square_value = numpy.square(prediction - ground_truth)
        self.history['mean_square_error'].append(square_value.mean())
        self.history['max_square_error'].append(square_value.max())
        self.history['min_square_error'].append(square_value.min())
        self.history['mean_absolute_error'].append(absolute_value.mean())
        self.history['max_absolute_error'].append(absolute_value.max())
        self.history['min_absolute_error'].append(absolute_value.min())

    def mean_square_error(self):
        """returns the mean square error across the whole data set"""
        mse_list = numpy.array(self.history['mean_square_error'])
        return mse_list.mean()

    def max_square_error(self):
        """returns the max square error across the whole data set"""
        mse_list = numpy.array(self.history['max_square_error'])
        return mse_list.max()

    def min_square_error(self):
        """returns the min square error across the whole data set"""
        mse_list = numpy.array(self.history['min_square_error'])
        return mse_list.min()

    def mean_absolute_error(self):
        """returns the mean absolute error across the whole data set"""
        mae_list = numpy.array(self.history['mean_absolute_error'])
        return mae_list.mean()

    def max_absolute_error(self):
        """returns the max absolute error across the whole data set"""
        mae_list = numpy.array(self.history['max_absolute_error'])
        return mae_list.max()

    def min_absolute_error(self):
        """returns the min absolute error across the whole data set"""
        mae_list = numpy.array(self.history['min_absolute_error'])
        return mae_list.min()

    def best_data_by_metric(self, metric):
        """
        returns the index of the best data sample in history based on metric
        :param str metric: which error metric to use. Should be an element in ERROR_METRICS
        :return: tuple: the index of the best sample, the error value
        """
        if metric not in self.ERROR_METRICS:
            raise RuntimeError("Invalid error metric: %s" % metric)

        val, idx = min((val, idx) for (idx, val) in enumerate(self.history[metric]))
        return idx, val

    def metric_range(self, metric):
        """returns the range of the given error metric"""
        if metric not in self.ERROR_METRICS:
            raise RuntimeError("Invalid error metric: %s" % metric)
        mse_list = numpy.array(self.history[metric])
        return min(mse_list), max(mse_list)

    def worst_data_by_metric(self, metric):
        """
        returns the index of the worst data sample in history based on metric
        :param str metric: which error metric to use. Should be an element in ERROR_METRICS
        :return: tuple: the index of the best sample, the error value
        """
        if metric not in self.ERROR_METRICS:
            raise RuntimeError("Invalid error metric: %s" % metric)

        val, idx = max((val, idx) for (idx, val) in enumerate(self.history[metric]))
        return idx, val


