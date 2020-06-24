
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
module for managing training data, training data is managed per folder. 
The folder should have a .ini file which contains metadata information
"""
import os
import logging
logger = logging.getLogger(__name__)


class TrainingData(object):
    """
    class for managing all training data generated for a training process
    :param str path: where the data is stored
    :param list files: a list of files that contain data
    """
    _path = None

    #: the list of legitimate files
    FILE_SET = ()

    def __init__(self):
        self._path = None
        self.files = []

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, p):
        self._path = p
        self.get_files()

    def get_files(self):
        """retrieve all files in the data path
        """
        all_files = os.listdir(self.path)
        all_files = [os.path.join(self.path, f) for f in all_files]
        
        self.files = [f for f in all_files if os.path.isfile(f)]
        self.files.sort()

    def get_config(self):
        """
        retrieve the config file
        :return: config file path
        :rtype: str
        """
        for fl in self.files:
            # assume the config file is .ini format
            if fl.endswith('.ini'):
                return fl

        return ''
