
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
module for various data used to generating training data.
The module is meant to be used inside of maya. All the methods are 
utility functions to aid training data generation. 
"""
import os
import math
import numpy
import pandas
try:
    import maya.cmds as mc
except ImportError as e:
    print(e)

import logging
logger = logging.getLogger(__name__)


class PrepData(object):
    def import_(self, path):
        pass

    @classmethod
    def from_path(cls, path):
        """
        generates an instance based on a csv file
        :return: a cls object
        :rtype: cls
        """
        obj = cls()
        obj.import_(path)
        return obj


class AnchorPoints(PrepData):
    """
    class to manage anchor points

    :param list pts: anchor points
    """
    def __init__(self):
        self.pts = []

    def retrieve_from_scene(self):
        """retrieve anchor points from scene selection
        """
        selected = mc.ls(sl=True, flatten=True)
        selected = [i.split('[')[-1] for i in selected]
        selected = [i.split(']')[0] for i in selected]
        selected = [int(i) for i in selected]
        self.pts = selected
        self.pts.sort()

    def export(self, path):
        """export selected points to a csv file
        """
        if not path.endswith('.csv'):
            path = path + '.csv'
        anchor_dict = {'anchors': self.pts}
        data_frame = pandas.DataFrame(anchor_dict)
        data_frame.to_csv(path)
        logger.info("anchor points exported to %s" % path)

    def import_(self, path):
        """import anchor points data from a file
        """
        data = pandas.read_csv(path)
        self.pts = list(data['anchors'])
        self.pts.sort()

    def select_points(self, mesh):
        """select anchor points on a mesh, used for debugging
        """
        if not mc.objExists(mesh):
            logger.error("%s doesn't exist" % mesh)
        vertices = []
        for v in self.pts:
            vertices.append('%s.vtx[%i]' % (mesh, v))
        mc.select(vertices, r=True)

    def __iter__(self):
        return self.pts.__iter__()


class JointRelations(PrepData):
    """
    class to manage joint parenting relations. 
    :param dict parent_dict: a dictionary recording joint parenting relationship
    :param dict category_dict: a dictionary categorizing joints
    """
    def __init__(self):
        self.parent_dict = {}
        self.category_dict = {}

    def retrieve_from_scene(self, skin_clusters=None):
        """
        retrieve joint parenting relationship from current scene
        :param skin_clusters: a list of skinClusters to consider
        :type skin_clusters: list
        """
        raise NotImplementedError("This method relies on specific tagging mechanism we use,\
                                   so it cannot be open sourced. The idea is to retrieve\
                                   joint parenting relationship from a maya scene. The method\
                                   is a utility function and is non-critical for the system\
                                   to work. Users can write their own version or generate\
                                   the csv file manually.")

    def export(self, path):
        """export joint relations to a csv file
        """
        joint_data = {'joint': [], 'parent': []}
        joints = self.parent_dict.keys()
        joints.sort()
        for jnt in joints:
            joint_data['joint'].append(jnt)
            joint_data['parent'].append(None)

        jnt_data_frame = pandas.DataFrame.from_dict(joint_data)
        jnt_data_frame.to_csv(path)

    def import_(self, path):
        """import joint data from a file
        """
        joint_data = pandas.read_csv(path)
        joint_data.dropna(inplace=True, thresh=2)
        self.parent_dict = {}
        self.category_dict = {'all': []}

        for row in joint_data.iterrows():
            joint = row[1]['joint']
            parent = row[1]['parent']
            if isinstance(parent, basestring):
                self.parent_dict[joint] = parent

            self.category_dict['all'].append(joint)

            if 'category' in row[1]:
                category = row[1]['category']
                if category not in self.category_dict:
                    self.category_dict[category] = []
                self.category_dict[category].append(joint)

    def get_all_joints(self):
        """return all joints defined in this relation obj
        """
        return sorted(self.category_dict['all'])


class MoverRanges(PrepData):
    """
    class to manage mover ranges. The code here only does a best guess,
    the result needs to be manually checked. 
    :param dict limits: a dictionary recording mover ranges
    """

    #: tolerance
    TOL = 0.00001

    _numeric_ctls = []

    def __init__(self):
        self.limits = {}
        self._numeric_ctls = []

    def retrieve_from_pose_lib(self, path):
        """retrieve mover range based on library poses stored in a path
        """
        raise NotImplementedError("This method relies on our custom format for storing animation data,\
                                   so it cannot be open sourced. The method is a utility function and\
                                   is not critical for the system to work. It should be easy to write\
                                   your own or define the mover range manually.")

    def export(self, path):
        """export mover range data
        """
        data_dict = {}
        full_attrs = ['rotateX', 'rotateY', 'rotateZ', 
                      'translateX', 'translateY', 'translateZ',
                      'scaleX', 'scaleY', 'scaleZ']
        attributes = ['rx', 'ry', 'rz', 'tx', 'ty', 'tz', 'sx', 'sy', 'sz', '']

        data_dict['mover'] = []
        for attr in attributes:
            for suffix in ['_min', '_max']:
                fa = attr + suffix
                data_dict[fa] = []

        reordered_dict = {}
        for mover_attr in self.limits:
            mover, attr = mover_attr.split('.', 1)
            if attr in full_attrs:
                ctrl_name = mover
                index = full_attrs.index(attr)
                final_attr = attributes[index]
            else:
                ctrl_name = mover_attr
                final_attr = ''

            if ctrl_name not in reordered_dict:
                reordered_dict[ctrl_name] = {}
            reordered_dict[ctrl_name][final_attr] = self.limits[mover_attr]

        movers = reordered_dict.keys()
        # the sorting key below is specific to our naming convention.
        # it helps debugging but sorting doesn't affect how the system works
        movers.sort(key=lambda x: x.replace('Lf', '').replace('Rt', ''))

        for mover in movers:
            data_dict['mover'].append(mover)
            for attr in attributes:
                for index, suffix in enumerate(['_min', '_max']):
                    final_attr = attr + suffix
                    if attr in reordered_dict[mover]:
                        data_dict[final_attr].append(reordered_dict[mover][attr][index])
                    else:
                        data_dict[final_attr].append(numpy.NAN)

        data_frame = pandas.DataFrame(data_dict)
        data_frame.to_csv(path, 
                          columns=['mover', 'rx_min', 'rx_max', 'ry_min', 
                                   'ry_max', 'rz_min', 'rz_max', 'tx_min', 
                                   'tx_max', 'ty_min', 'ty_max', 'tz_min', 
                                   'tz_max', 'sx_min', 'sx_max', 'sy_min', 
                                   'sy_max', 'sz_min', 'sz_max', '_min', 
                                   '_max'], 
                          index=False)   

    def import_(self, path):
        """import range info from a csv file
        """
        self.limits = {}
        limit_data = pandas.read_csv(path)
        # fill in 0s for blank cells
        limit_data.fillna(0, inplace=True)

        # get all available attributes
        attributes = list(limit_data.columns[1:])
        attributes = [i.replace('_min', '') for i in attributes]
        attributes = [i.replace('_max', '') for i in attributes]
        attributes = set(attributes)

        mover_names = limit_data['mover']

        for index, mover in enumerate(mover_names):
            for attr in attributes:
                min_v = limit_data['%s_min' % attr][index]
                max_v = limit_data['%s_max' % attr][index]
                if (max_v - min_v) > self.TOL:
                    if attr:
                        full_attr = '%s.%s' % (mover, attr)
                    else:
                        full_attr = mover
                        self._numeric_ctls.append(full_attr)
                    self.limits[full_attr] = (min_v, max_v)

    @property
    def controls(self):
        return self.limits.keys()

    @property
    def numeric_controls(self):
        if self._numeric_ctls:
            self._numeric_ctls.sort()
            return self._numeric_ctls
        else:
            xform_attrs = ['rotateX', 'rotateY', 'rotateZ', 
                           'translateX', 'translateY', 'translateZ',
                           'scaleX', 'scaleY', 'scaleZ']
            nctls = []
            ctls = self.limits.keys()
            for ctl in ctls:
                attr = ctl.split('.', 1)[-1]
                if attr not in xform_attrs:
                    nctls.append(ctl)
            return nctls

    def __getitem__(self, key):
        return self.limits[key]


class DataInclusion(PrepData):
    """
    class to manage which objects are included in the training.
    :param list included: list of objects to be included for training
    """
    def __init__(self):
        self.included = []
    
    def add_selected(self):
        """add included objects based on scene selection
        """
        selected = mc.ls(selection=True, flatten=True)
        self.included.extend(selected)

    def add(self, include_list):
        """add included objects based on given list. This is for attributes
        """
        self.included.extend(include_list)

    def export(self, path):
        """export data to a csv file
        """
        include_dict = {'included': self.included}
        data = pandas.DataFrame(include_dict)
        data.to_csv(path)

    def import_(self, path):
        """import data from a csv file
        """
        if path:
            data = pandas.read_csv(path)
            self.included = set(data['included'])
        else:
            self.included = []

    def is_included(self, obj):
        """checks if the obj should be included for training
        """
        if not self.included:
            return True
        return obj in self.included
