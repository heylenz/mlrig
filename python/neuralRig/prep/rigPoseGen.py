
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


import os
import shutil
import pandas
import logging
import scipy.stats as stats
import copy

try:
    import maya.cmds as mc
    import maya.api.OpenMaya as om
    import ml.neuralRig.prep.correctiveBlendShape as cbs
except ImportError as e:
    print e

try:
    import configparser
except ImportError as e:
    # python 2 backwards compatibility
    import ConfigParser as configparser

import ml.neuralRig.prep.dataPrep as dp
import ml.neuralRig.prep.rigTrainingData as rigTrainingData

logger = logging.getLogger(__name__)


class RigPoseGenerator(object):
    """
    class to manage training pose data generation
    :param str conifg_file: .ini file containing paths for all other files
    :param str limit_file: csv file with mover range information
    :param dp.MoverRanges mover_ranges: object defining mover ranges
    :param str joint_file: csv file with joint parent relationship information
    :param dp.JointRelations joint_relations: object defining joint relations
    :param str maya_file: maya file used as the rig
    :param str inclusion_file: csv file with information regarding what's to include for training
    :param dp.DataInclusion inclusion: obj managing what's to include for training
    :param str anchor_file: csv file with anchor points information
    :param dp.AnchorPoints anchors: obj managing anchor points
    :param dict cur_data_set: dictionary storing the pose data
    :param int vertex_count: number of vertices on the mesh
    :param str mesh: mesh name to retrieve data from
    :param dict data_frames: a list of pandas.DataFrames object, used for exporting data
    """
    TEMP_BS_NODE = 'mlBlendShape'
    TEMP_TARGET = 'mlTarget'

    SKIN_TYPES = ['skinCluster']

    TOL = 0.00001

    def __init__(self, 
                 limit_file, 
                 joint_file, 
                 maya_file=None, 
                 inclusion_file=None,
                 anchor_file=None,
                 mesh='main'):
        self.limit_file = limit_file
        self.mover_ranges = dp.MoverRanges.from_path(self.limit_file)
        self.joint_file = joint_file
        self.joint_relations = dp.JointRelations.from_path(self.joint_file)
        self.maya_file = maya_file
        self.inclusion_file = inclusion_file
        self.inclusion = dp.DataInclusion.from_path(self.inclusion_file)
        self.anchor_file = anchor_file
        self.anchors = dp.AnchorPoints.from_path(self.anchor_file)
        self.mesh = mesh
        self.data_frames = {}
        self._mesh_analyzed = False
        self.cur_data_set = {}
        self.vertex_count = 0
        self._deformers = []
        self._connection_map = {}
        self.config_file = None
        self._mover_controls = []
        self._movers = []


    @classmethod
    def __aug_path(cls, config_path, path):
        if path.startswith(os.sep):
            return path
        else:
            dirname = os.path.dirname(config_path)
            return os.path.join(dirname, path)

    @classmethod
    def from_config(cls, config):
        """
        generates an instance base on a config file
        :param str config: path to the .ini config file
        :return: a RigPoseGenerator object
        :rtype: RigPoseGenerator
        """
        config_handle = configparser.ConfigParser()
        config_handle.read(config)
        limit_file = config_handle.get('training_config', 'limit_file')
        limit_file = cls.__aug_path(config, limit_file)
        joint_file = config_handle.get('training_config', 'joint_relation_file')
        joint_file = cls.__aug_path(config, joint_file)
        maya_file = config_handle.get('training_config', 'maya_file')
        maya_file = cls.__aug_path(config, maya_file)
        inclusion_file = config_handle.get('training_config', 'inclusion_file')
        inclusion_file = cls.__aug_path(config, inclusion_file)
        anchor_file = config_handle.get('training_config', 'anchor_file')
        anchor_file = cls.__aug_path(config, anchor_file)

        mesh = config_handle.get('training_config', 'mesh')
        obj = cls(limit_file, joint_file, maya_file, inclusion_file, anchor_file, mesh)
        obj.config_file = config
        return obj

    def __get_vtx_ct(self, mesh):
        self.vertex_count = len(mc.ls(mesh + '.cp[*]', flatten=True))

    def __get_connection_info(self, mesh):

        self._connection_map = {}
        selection = om.MSelectionList()
        if mc.nodeType(mesh) == 'transform':
            mesh = mc.listRelatives(mesh, children=True)[0]
        selection.add(mesh)
        geom = selection.getDependNode(0)
        vtx_iter = om.MItMeshVertex(geom)
        while not vtx_iter.isDone():
            index = vtx_iter.index()
            connected_vtx = vtx_iter.getConnectedVertices()
            self._connection_map[index] = connected_vtx
            vtx_iter.next()

    def pose_rig(self,
                 rand_generator=stats,
                 distribution_scale=1.0,
                 pose_data=None,
                 read_frame=None):
        """
        method to pose the rig
        :param module rand_generator: module for random number generation
        :param float distribution_scale: value to scale the distribution, making it wider or narrower
        :param dict pose_data: if given, use given data to pose the rig instead of random numbers
        :param int read_frame: if given, instead of random values, use the pose from given frame
        """
        self.cur_data_set['moverValues'] = {}
        if read_frame is not None:
            mc.currentTime(read_frame)
            for ctrl in self._mover_controls:
                self.cur_data_set['moverValues'][ctrl] = [mc.getAttr(ctrl)]
            return

        distribution = rand_generator.truncnorm(-1 * distribution_scale,
                                                distribution_scale,
                                                loc=0.0,
                                                scale=1.0)

        dis = distribution.rvs(len(self.mover_ranges.controls))

        for index, ctrl in enumerate(self.mover_ranges.controls):
            lock_state = mc.getAttr(ctrl, lock=True)
            if lock_state:
                logger.debug("warning: can't set %s" % ctrl)
                continue

            connect_state = mc.listConnections(ctrl, destination=False)
            if connect_state:
                logger.debug("warning: can't set %s because of connection" % ctrl)
                continue

            if pose_data is not None:
                value = pose_data[ctrl]
                mc.setAttr(ctrl, value)
                self.cur_data_set['moverValues'][ctrl] = [value]
            else:
                num = dis[index]

                min_v, max_v = self.mover_ranges[ctrl]
                value = min_v + (max_v - min_v) * (num + distribution_scale) / (distribution_scale * 2.0)
                logger.debug("setting %s to %i" % (ctrl, value))
                mc.setAttr(ctrl, value)
                self.cur_data_set['moverValues'][ctrl] = [value]
    
    def __prep_mesh(self, mesh):
        mc.deformer(mesh,
                    frontOfChain=True, 
                    type='blendShape', 
                    name=self.TEMP_BS_NODE)
        history = mc.listHistory(mesh, interestLevel=1)
        history = [i for i in history if 'geometryFilter' in mc.nodeType(i, inherited=True)]
        
        self._deformers = []
        for his in history:
            if his == self.TEMP_BS_NODE:
                break
            self._deformers.append(his)

    def retrieve_data(self,
                      precision=8):
        """
        retrieve pose data from current pose
        :param int precision: result precision
        """

        if not self._mesh_analyzed:
            self.__get_vtx_ct(self.mesh)
            self._mesh_analyzed = True
            self._deformers = {}

        joints = self.joint_relations.get_all_joints()
        world_mats = {}
        
        self.cur_data_set['jointWorldMatrix'] = {}
        self.cur_data_set['jointWorldQuaternion'] = {}
        self.cur_data_set['jointLocalMatrix'] = {}
        self.cur_data_set['jointLocalQuaternion'] = {}
        self.cur_data_set['controls'] = {}
        self.cur_data_set['worldPos'] = {}
        self.cur_data_set['worldOffset'] = {}
        self.cur_data_set['localOffset'] = {}
        self.cur_data_set['differentialOffset'] = {}
        self.cur_data_set['anchorPoints'] = {}
        if not self._connection_map:
            self.__get_connection_info(self.mesh)

        for jnt in joints:
            if not self.inclusion.is_included(jnt):
                continue
            world_mat = mc.getAttr(jnt + '.worldMatrix[0]')
            world_mats[jnt] = world_mat
            
            wm = om.MTransformationMatrix(om.MMatrix(world_mat))
            quaternion = wm.rotation(asQuaternion=True)

            temp = [world_mat[0], world_mat[1], world_mat[2],
                    world_mat[4], world_mat[5], world_mat[6],
                    world_mat[8], world_mat[9], world_mat[10],
                    world_mat[12], world_mat[13], world_mat[14]]

            self.cur_data_set['jointWorldMatrix'][jnt] = [[round(i, precision) for i in temp]]

            temp = [quaternion[0], quaternion[1], quaternion[2], quaternion[3],
                    world_mat[12], world_mat[13], world_mat[14]]
            self.cur_data_set['jointWorldQuaternion'][jnt] = [[round(i, precision) for i in temp]]

        for jnt in joints:
            if not self.inclusion.is_included(jnt):
                continue
            parent = self.joint_relations.parent_dict.get(jnt, None)
            if parent:
                if parent not in world_mats:
                    world_mats[parent] = mc.getAttr(parent + '.worldMatrix[0]')
                parent_mat = om.MMatrix(world_mats[parent])
                world_mat = om.MMatrix(world_mats[jnt])
                local_mat = world_mat * parent_mat.inverse()

            else:
                local_mat = om.MMatrix(world_mats[jnt])
            lm = om.MTransformationMatrix(local_mat)
            quaternion = lm.rotation(asQuaternion=True)

            temp = [local_mat[0], local_mat[1], local_mat[2],
                    local_mat[4], local_mat[5], local_mat[6],
                    local_mat[8], local_mat[9], local_mat[10],
                    local_mat[12], local_mat[13], local_mat[14]]
            self.cur_data_set['jointLocalMatrix'][jnt] = [[round(i, precision) for i in temp]]

            temp = [quaternion[0], quaternion[1], quaternion[2], quaternion[3],
                    local_mat[12], local_mat[13], local_mat[14]]
            self.cur_data_set['jointLocalQuaternion'][jnt] = [[round(i, precision) for i in temp]]

        for nc in self.mover_ranges.numeric_controls:
            if not self.inclusion.is_included(nc):
                continue
            value = mc.getAttr(nc)
            min_v, max_v = self.mover_ranges[nc]
            value = (value - min_v) / (max_v - min_v)
            self.cur_data_set['controls'][nc] = [[value]]

        if not self._deformers:
            self.__prep_mesh(mesh=self.mesh)

        meshShape = mc.listRelatives(self.mesh, children=True)
        meshShape = mc.ls(meshShape, ni=True)
        meshShape = meshShape[0]
        positions = cbs.getPositions(meshShape)
        for i, p in enumerate(positions):
            vtx = '%s.vtx[%i]' % (self.mesh, i)
            if not self.inclusion.is_included(vtx):
                continue
            self.cur_data_set['worldPos'][i] = [[round(p[0], precision),
                                                 round(p[1], precision),
                                                 round(p[2], precision)]]

        duplicate = mc.duplicate(self.mesh,
                                 name=self.mesh + 'Dup',
                                 upstreamNodes=False,
                                 returnRootsOnly=True)[0]

        deformer_env_dict = {}
        # shut down all deformers except for skin clusters
        for deformer in self._deformers:
            dtype = mc.nodeType(deformer)
            if dtype not in self.SKIN_TYPES:
                deformer_env_dict[deformer] = mc.getAttr(deformer + '.envelope')
                mc.setAttr(deformer + '.envelope', 0.0)

        linear_positions = cbs.getPositions(meshShape)

        for i in range(self.vertex_count):
            vtx = '%s.vtx[%i]' % (self.mesh, i)
            if not self.inclusion.is_included(vtx):
                continue
            offset = positions[i] - linear_positions[i]
            self.cur_data_set['worldOffset'][i] = [[round(offset[0], precision),
                                                    round(offset[1], precision),
                                                    round(offset[2], precision)]]

        offsets = cbs.makeCorrectiveFromSculpt(self.mesh,
                                               duplicate,
                                               self.TEMP_BS_NODE,
                                               self.TEMP_TARGET)

        for i in range(self.vertex_count):
            vtx = '%s.vtx[%i]' % (self.mesh, i)
            if not self.inclusion.is_included(vtx):
                continue
            offset = offsets.get(i, [0.0, 0.0, 0.0])
            self.cur_data_set['localOffset'][i] = [[round(data, precision) for data in offset]]

        for i in range(self.vertex_count):
            vtx = '%s.vtx[%i]' % (self.mesh, i)
            if not self.inclusion.is_included(vtx):
                continue
            neighbors = self._connection_map[i]
            valence = float(len(neighbors))
            new_coord = offsets.get(i, [0.0, 0.0, 0.0])

            neighbor_values = [0.0, 0.0, 0.0]
            for neighbor in neighbors:
                nb_coord = offsets.get(neighbor, [0.0, 0.0, 0.0])
                x = neighbor_values[0] + nb_coord[0]
                y = neighbor_values[1] + nb_coord[1]
                z = neighbor_values[2] + nb_coord[2]

                neighbor_values = [x, y, z]
            x = new_coord[0] - neighbor_values[0] / valence
            y = new_coord[1] - neighbor_values[1] / valence
            z = new_coord[2] - neighbor_values[2] / valence

            offset = [x, y, z]
            self.cur_data_set['differentialOffset'][i] = [[round(data, precision) for data in offset]]

            for anchor in self.anchors:
                index = int(anchor)
                offset = offsets.get(anchor, [0.0, 0.0, 0.0])
                self.cur_data_set['anchorPoints'][index] = [[round(data, precision) for data in offset]]

        # restore the rig
        for deformer in self._deformers:
            dtype = mc.nodeType(deformer)
            if dtype not in self.SKIN_TYPES:
                mc.setAttr(deformer + '.envelope', deformer_env_dict[deformer])
        mc.delete(duplicate)
    
    def batch_export_farm(self,
                          export_path,
                          num_of_poses,
                          data_format='pkl',
                          batch_size=500,
                          data_path=None,
                          log_dir=None):
        """
        export the training data on the farm
        :param str export_path: export path
        :param int num_of_poses: how many poses to generate
        :param str data_format: pkl or csv
        :param int batch_size: how big each batch be. Each batch goes into a single file
        :param str data_path: if given, use existing mover values in given data path instead of random mover values
        :param str log_dir: log directory
        """
        raise NotImplementedError("This method works with apis specific to our render farm,\
                                   so it cannot be open sourced. Users can use batch_export method\
                                   or write their own batch_export_farm if a render farm is avaliable.\
                                   Using a render farm greatly reduces the time needed to generate\
                                   training samples, as the tasks are parallelizable")

    def batch_export(self,
                     export_path,
                     num_of_poses,
                     data_format='pkl',
                     batch_size=500,
                     data_path=None,
                     read_frame=None):
        """
        export the training data as different batches
        :param str export_path: export path
        :param int num_of_poses: how many poses to generate
        :param str data_format: pkl or csv
        :param int batch_size: how big each batch be. Each batch goes into a single file
        :param str data_path: if given, use existing mover values in given data path instead of random mover values
        """
        if num_of_poses <= 0:
            return
        iter_num = 0

        cur_frame = read_frame
        while num_of_poses > 0:
            num_of_poses = num_of_poses - batch_size

            cur_batch_num = batch_size
            if num_of_poses < 0:
                cur_batch_num = num_of_poses + batch_size
            
            if data_path is not None:
                existing_data = rigTrainingData.RigTrainingData()
                existing_data.path = data_path
                mover_data = existing_data.get_mover_data(iter_num)
            else:
                mover_data = None

            if read_frame:
                self.gen_pose_data(cur_batch_num,
                                   mover_data=mover_data,
                                   read_frame=cur_frame)
                cur_frame += cur_batch_num
            else:
                self.gen_pose_data(cur_batch_num,
                                   mover_data=mover_data)
            self.export(export_path,
                        data_format=data_format,
                        iter_num=iter_num)
            iter_num += 1

    def gen_pose_data(self,
                      num_of_poses,
                      mover_data=None,
                      read_frame=None):
        """
        generate pose data
        :param int num_of_poses: number of poses
        :param pandas.DataFrame mover_data: if given, uses existing control info instead of random poses
        :param int read_frame: if given, read pose data from given frame number instead
        """
        if num_of_poses <= 0:
            return

        if read_frame is not None:
            # TODO: bssMover is a custom node type used by us for animation controls. Please 
            # replace the line beline with the appropriate logic for your workflow 
            # the idea here is to gather all the controls used to animate the rig
            movers = mc.ls(type='bssMover')
            self._movers = []
            self._mover_controls = []
            for mover in movers:
                # TODO: skippint namespaced movers. Again, this is specific for our workflow. Please 
                # replace it or remove it based on your workflow
                if mover.find(':') > -1:
                    continue
                parent = mc.listRelatives(mover, parent=True, path=True)[0]
                self._movers.append(parent)
                attrs = mc.listAttr(parent, keyable=True) or []
                for attr in attrs:
                    self._mover_controls.append(parent + '.' + attr)

        self.data_frames = {}
        for i in range(num_of_poses):
            if mover_data is not None:
                cur_pose_data = mover_data.iloc[i]
            else:
                cur_pose_data = None
            if read_frame is not None:
                cur_frame = read_frame + i
                self.pose_rig(pose_data=cur_pose_data, read_frame=cur_frame)
            else:
                self.pose_rig(pose_data=cur_pose_data)
            self.retrieve_data()

            for key, value in self.cur_data_set.items():
                if not value:
                    continue
                if key not in self.data_frames:
                    self.data_frames[key] = pandas.DataFrame(value)
                else:
                    df = pandas.DataFrame(value)
                    self.data_frames[key] = self.data_frames[key].append(df, ignore_index=True)
    
    def export(self, export_path, data_format='pkl', iter_num=0):
        """
        export pose data to files
        :param str export_path: path for export
        :param str data_format: pkl or csv
        :param int iter_num: the batch number that should appear on the file name
        """
        if self.config_file:
            config_copy = os.path.join(export_path, self.config_file.split(os.sep)[-1])
            if not os.path.exists(config_copy):
                shutil.copy(self.config_file, config_copy)

        for key, dataFrame in self.data_frames.items():
            path = os.path.join(export_path,
                                '%s%i.%s' % (key, iter_num, data_format))
            if data_format == 'pkl':
                dataFrame.to_pickle(path)
            elif data_format == 'csv':
                dataFrame.to_csv(path)

    def gen_control_mapping(self, num_of_poses=5):
        """
        generates a mapping which shows which controls affect each vertex
        :param int num_of_poses: how many poses to use to generate the mapping
        :return: a dictionary with the mapping info
        """
        control_mapping = {}
        for i in range(num_of_poses):
            self.pose_rig()
            self.retrieve_data()
            cur_data_set = copy.deepcopy(self.cur_data_set)

            for ctrl in self.mover_ranges.controls:
                min_v, max_v = self.mover_ranges[ctrl]
                cur_value = mc.getAttr(ctrl)
                mc.setAttr(ctrl, max_v)
                self.retrieve_data()

                vertices = []
                joints = []
                controls = []
                for vtx in cur_data_set['localOffset']:
                    orig_offset = cur_data_set['localOffset'][vtx][0]
                    new_offset = self.cur_data_set['localOffset'][vtx][0]

                    diff = abs(orig_offset[0] - new_offset[0]) + \
                           abs(orig_offset[1] - new_offset[1]) + \
                           abs(orig_offset[2] - new_offset[2])
                    if diff > self.TOL:
                        vertices.append(vtx)

                for jnt in cur_data_set['jointLocalMatrix']:
                    orig_mtx = cur_data_set['jointLocalMatrix'][jnt][0]
                    new_mtx = self.cur_data_set['jointLocalMatrix'][jnt][0]
                    diff = 0.0
                    for ii in range(12):
                        diff += abs(orig_mtx[ii] - new_mtx[ii])
                    if diff > self.TOL:
                        joints.append(jnt)

                for ctl in cur_data_set['controls']:
                    orig_ctl = cur_data_set['controls'][ctl][0]
                    new_ctl = self.cur_data_set['controls'][ctl][0]
                    diff = abs(orig_ctl[0] - new_ctl[0])

                    if diff > self.TOL:
                        controls.append(ctl)

                for vtx in vertices:
                    if vtx not in control_mapping:
                        control_mapping[vtx] = []

                    for jnt in joints:
                        if jnt not in control_mapping[vtx]:
                            control_mapping[vtx].append(jnt)
                    for ctl in controls:
                        if ctl not in control_mapping[vtx]:
                            control_mapping[vtx].append(ctl)
                mc.setAttr(ctrl, cur_value)
        return control_mapping

