
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
import maya.cmds as mc
import maya.api.OpenMaya as om
import ml.neuralRig.prep.blendShapes as bs
import pandas
import numpy
import scipy
import scipy.sparse.linalg as linalg
from scipy.sparse import csr_matrix
import ml.neuralRig.train.setup as setup
import ml.error as error
import ml.neuralRig.prep.dataPrep as dp
import ml.neuralRig.prep.correctiveBlendShape as cbs

try:
    import configparser
except ImportError as e:
    # python 2 backwards compatibility
    import ConfigParser as configparser

import logging
logger = logging.getLogger(__name__)


class RigPoseApplier(setup.TrainingSession):
    """
    class to apply training result in maya as static poses
    :param str config: path to the .init file containing configuration information
    :param pandas.DataFrames pose_data: DataFrame containing pose data
    :param dict prediction_data: a dictionary containing prediction data
    :param str blendshape: name of the blendshape node to apply the prediction
    :param dict vertex_ids: a dict of vertices that are affected by the prediction. training_type is used as key
    :param float anchor_weight: how much weight should be given to anchor points for differential coordinates solve.
                                Does nothing for local_offset prediction
    :param error.ErrorHistory error_history: history of all the prediction errors
    """
    TARGET = 'mlTarget'

    def __init__(self, config, blendshape='blendShapeMain', anchor_weight=1.0):
        """
        init function
        :param str config: .ini file containing configuration information
        :param str blendshape: name of the blendshape node to apply the prediction
        :param float anchor_weight: how much weight should be given to anchor points for differential coordinates
                                    solve. Does nothing for local_offset prediction
        """
        super(RigPoseApplier, self).__init__(config)
        self.pose_data = None
        self.prediction_data = {}
        self.blendshape = blendshape
        self.vertex_ids = {}
        self.anchor_weight = anchor_weight
        self.error_history = error.ErrorHistory()
        self._laplacian_row = None
        self._laplacian_col = None
        self._laplacian_val = None
        self._laplacian_matrix = None
        self._valences = []
        self._cholesky_matrix = None
        self._reverse_cuthill_mckee_order = None
        self._inverse_cmo = None

        try:
            self._mesh = mc.deformer(self.blendshape, query=True, geometry=True)[0]
        except ValueError:
            self._mesh = None

    def gen_deformer(self, training_type='differential', deformer_type='tensorflow'):
        """
        generates a bssNNDeform deformer
        :param str training_type: the type of training. Valid options are differential and local_offset
        """
        if not mc.pluginInfo("bssNNDeform.so", loaded=True, q=True):
            mc.loadPlugin("bssNNDeform.so")

        input_config = self.training_data.get_config()
        input_handle = configparser.ConfigParser()
        input_handle.read(input_config)

        range_file = input_handle.get('training_config', 'limit_file')
        joint_relation_file = input_handle.get('training_config', 'joint_relation_file')
        inclusion_file = input_handle.get('training_config', 'inclusion_file')
        anchor_file = input_handle.get('training_config', 'anchor_file')
        if not self._mesh:
            self._mesh = input_handle.get('training_config', 'mesh')

        mover_ranges = dp.MoverRanges()
        mover_ranges.import_(range_file)

        joint_relations = dp.JointRelations()
        joint_relations.import_(joint_relation_file)

        data_inclusion = dp.DataInclusion()
        data_inclusion.import_(inclusion_file)

        anchor_points = dp.AnchorPoints()
        anchor_points.import_(anchor_file)

        joint_dict = joint_relations.parent_dict

        deformers = mc.listHistory(self._mesh)
        deformers = [d for d in deformers if 'geometryFilter' in mc.nodeType(d, inherited=True)]
        deformers = [d for d in deformers if mc.nodeType(d) != 'tweak']

        cur_skin = None
        if deformers:
            cur_skin = deformers[0]
        deformer = mc.deformer(self._mesh, type="bssNNDeform")[0]

        if cur_skin:
            mc.reorderDeformers(cur_skin, deformer, self._mesh)

        i = 0

        for joint in joint_relations.get_all_joints():
            if not mc.objExists(joint):
                logger.warning("%s doesn't exist" % joint)
                continue

            if not data_inclusion.is_included(joint):
                continue
            joint_parent = joint_relations.parent_dict.get(joint, None)
            if not joint_parent or not mc.objExists(joint_parent):
                mc.connectAttr('%s.worldMatrix[0]' % joint, '%s.matrixInfluences[%i]' % (deformer, i))
            else:
                mult_node = 'mult_matrix_%s' % joint
                mc.createNode('multMatrix', name=mult_node)
                mc.connectAttr('%s.worldMatrix[0]' % joint, '%s.matrixIn[0]' % mult_node)
                mc.connectAttr('%s.worldInverseMatrix[0]' % joint_parent, '%s.matrixIn[1]' % mult_node)
                mc.connectAttr('%s.matrixSum' % mult_node, '%s.matrixInfluences[%i]' % (deformer, i))
            i = i + 1

        i = 0
        numeric_min = []
        numeric_max = []
        for inc in mover_ranges.numeric_controls:
            if not data_inclusion.is_included(inc):
                continue
            mc.connectAttr(inc, '%s.numericInfluences[%i]' % (deformer, i))
            cmin, cmax = mover_ranges.limits[inc]
            numeric_min.append(cmin)
            numeric_max.append(cmax)
            i = i + 1

        mc.setAttr('%s.numericInputMin' % deformer, numeric_min, type='floatArray')
        mc.setAttr('%s.numericInputMax' % deformer, numeric_max, type='floatArray')

        if training_type == 'differential':

            mc.setAttr('%s.jointInputMin' % deformer, self.training_params['differential_joint_translate_min'])
            mc.setAttr('%s.jointInputMax' % deformer, self.training_params['differential_joint_translate_max'])
            mc.setAttr('%s.normalizeInput' % deformer, self.training_params['differential_normalize'])

            mc.setAttr('%s.differentialOutputMin' % deformer, self.training_params['differential_min'])
            mc.setAttr('%s.differentialOutputMax' % deformer, self.training_params['differential_max'])
            mc.setAttr('%s.differentialNormalizeOutput' % deformer, self.training_params['differential_normalize'])

            diff_model_path = os.path.join(self.output_path, 'differential', 'model.pb')
            mc.setAttr('%s.differentialNetworkPath' % deformer, diff_model_path, type='string')

            mc.setAttr('%s.anchorOutputMin' % deformer, self.training_params['anchor_min'])
            mc.setAttr('%s.anchorOutputMax' % deformer, self.training_params['anchor_max'])
            mc.setAttr('%s.anchorNormalizeOutput' % deformer, self.training_params['anchor_normalize'])

            anchor_model_path = os.path.join(self.output_path, 'anchor', 'model.pb')
            mc.setAttr('%s.anchorNetworkPath' % deformer, anchor_model_path, type='string')

            # assuming the training type is the same for differential and anchor:
            try:
                mc.setAttr('%s.networkType' % deformer,
                            1 if self.training_params['differential_framework'] == 'mxnet' else 0)
            except:
                print("Warning: couldn't set network type")


            self._cal_matrices()

            mc.setAttr('%s.affectedVertices' % deformer, self.vertex_ids['differential'], type='Int32Array')
            mc.setAttr('%s.laplacianMatrixRows' % deformer, self._laplacian_row, type='Int32Array')
            mc.setAttr('%s.laplacianMatrixColumns' % deformer, self._laplacian_col, type='Int32Array')
            mc.setAttr('%s.laplacianMatrixValues' % deformer, self._laplacian_val, type='floatArray')

            cholesky_row = []
            cholesky_col = []
            cholesky_val = []

            row_ct, col_ct = self._cholesky_matrix.shape

            tol = 0.00000001
            for i in range(row_ct):
                for j in range(i+1):
                    val = self._cholesky_matrix[i][j]
                    if abs(val) >= tol:
                        cholesky_row.append(i)
                        cholesky_col.append(j)
                        cholesky_val.append(float(val))

            mc.setAttr('%s.choleskyMatrixRows' % deformer, cholesky_row, type='Int32Array')
            mc.setAttr('%s.choleskyMatrixColumns' % deformer, cholesky_col, type='Int32Array')

            mc.setAttr('%s.choleskyMatrixValues' % deformer, cholesky_val, type='floatArray')

            rcm_list = list(self._reverse_cuthill_mckee_order)
            rcm_list = [int(i) for i in rcm_list]
            mc.setAttr('%s.reverseCuthillMckeeOrder' % deformer, rcm_list, type='Int32Array')

            augmented_valences = self._valences[:]
            augmented_valences.extend([self.anchor_weight] * len(anchor_points.pts))

            mc.setAttr('%s.valences' % deformer, augmented_valences, type='floatArray')
            mc.setAttr('%s.differentialInputName' % deformer,
                       self.training_params['differential_network_input'].split(':')[0],
                       type='string')

            mc.setAttr('%s.differentialOutputName' % deformer,
                       self.training_params['differential_network_output'].split(':')[0],
                       type='string')

            mc.setAttr('%s.anchorInputName' % deformer,
                       self.training_params['anchor_network_input'].split(':')[0],
                       type='string')

            anchor_names = self.training_params['anchor_network_output'].split(',')
            anchor_names = [i.split(':')[0] for i in anchor_names]

            mc.setAttr('%s.anchorOutputName' % deformer,
                       ','.join(anchor_names),
                       type='string')

            mc.setAttr('%s.isDifferential' % deformer, True)

        else:
            mc.setAttr('%s.isDifferential' % deformer, False)
            mc.setAttr('%s.jointInputMin' % deformer, self.training_params['local_offset_joint_translate_min'])
            mc.setAttr('%s.jointInputMax' % deformer, self.training_params['local_offset_joint_translate_max'])
            mc.setAttr('%s.normalizeInput' % deformer, self.training_params['local_offset_normalize'])

            self.read_prediction_data(training_types=['local_offset'], ground_truth=True)
            mc.setAttr('%s.anchorOutputMin' % deformer, self.training_params['local_offset_min'])
            mc.setAttr('%s.anchorOutputMax' % deformer, self.training_params['local_offset_max'])
            mc.setAttr('%s.anchorNormalizeOutput' % deformer, self.training_params['local_offset_normalize'])

            anchor_model_path = os.path.join(self.output_path, 'local_offset', 'model.pb')
            mc.setAttr('%s.anchorNetworkPath' % deformer, anchor_model_path, type='string')

            mc.setAttr('%s.affectedVertices' % deformer, self.vertex_ids['local_offset'], type='Int32Array')

            mc.setAttr('%s.anchorInputName' % deformer,
                       self.training_params['local_offset_network_input'].split(':')[0],
                       type='string')

            anchor_names = self.training_params['local_offset_network_output'].split(',')
            anchor_names = [i.split(':')[0] for i in anchor_names]
            mc.setAttr('%s.anchorOutputName' % deformer,
                       ','.join(anchor_names),
                       type='string')

    def read_pose_data(self):
        """reads in mover data"""
        self.pose_data = None
        for i in self.evaluation_set:
            if self.pose_data is None:
                self.pose_data = self.evaluation_data.get_mover_data(i)
            else:
                pose_data = self.evaluation_data.get_mover_data(i)
                self.pose_data = self.pose_data.append(pose_data, ignore_index=True)

    def read_prediction_data(self, training_types=('differential', 'anchor'), ground_truth=False):
        """
        read in prediction data
        :param list training_types: a list of training types that have result predicted
        :param bool ground_truth: if True, read ground truth data instead
        """
        for training_type in training_types:
            self.prediction_data[training_type] = None

            if ground_truth:
                for i in self.evaluation_set:
                    if self.prediction_data[training_type] is None:
                        self.prediction_data[training_type] = self.evaluation_data.get_data(training_type, i)
                    else:
                        new_data = self.evaluation_data.get_data(training_type, i)
                        old_data = self.prediction_data[training_type]
                        self.prediction_data[training_type] = old_data.append(new_data, ignore_index=True)
            else:
                prediction_file = os.path.join(self.output_path, training_type, 'prediction.json')
                self.prediction_data[training_type] = pandas.read_json(prediction_file)
            self.vertex_ids[training_type] = sorted(list(self.prediction_data[training_type].columns))

    def _cal_matrices(self):
        """
        calculate laplacian matrix, cholesky matrix and _reverse_cuthill_mckee_order
        """
        if not self.vertex_ids or 'differential' not in self.vertex_ids:
            self.read_prediction_data(ground_truth=True)

        anchors = self.vertex_ids['anchor']

        final_indices = self.vertex_ids['differential']

        connection_map = {}
        self._valences = []
        selection_list = om.MSelectionList()
        selection_list.add(self._mesh)
        geom_obj = selection_list.getDependNode(0)
        vtx_iter = om.MItMeshVertex(geom_obj)
        while not vtx_iter.isDone():
            index = vtx_iter.index()
            connected_vtx = vtx_iter.getConnectedVertices()
            connection_map[index] = connected_vtx
            vtx_iter.next()

        num_vtx = len(final_indices)

        row = []
        col = []
        sparse_v = []

        col_ct = num_vtx
        row_ct = num_vtx + len(anchors)
        for i, index in enumerate(final_indices):
            connected_vertices = connection_map[index]
            valence = float(len(connected_vertices))
            self._valences.append(valence)

            for j in range(num_vtx):
                if i == j:
                    row.append(i)
                    col.append(j)
                    sparse_v.append(valence)
                elif final_indices[j] in connected_vertices:
                    row.append(i)
                    col.append(j)
                    sparse_v.append(-1.0)

        for i, cur_anchor in enumerate(anchors):
            for j, index in enumerate(final_indices):
                if index == cur_anchor:
                    row.append(i + num_vtx)
                    col.append(j)
                    sparse_v.append(self.anchor_weight)

        sparse_v = numpy.array(sparse_v)
        self._laplacian_val = list(sparse_v)
        row = numpy.array(row)
        self._laplacian_row = list(row)
        col = numpy.array(col)
        self._laplacian_col = list(col)
        self._laplacian_matrix = csr_matrix((sparse_v, (row, col)),
                                            shape=(row_ct, col_ct))

        # calculating the normal matrix
        normal_matrix = self._laplacian_matrix.transpose() * self._laplacian_matrix

        # reverse cuthill mckee reordering to reduce filling
        self._reverse_cuthill_mckee_order = scipy.sparse.csgraph.reverse_cuthill_mckee(normal_matrix,
                                                                                       symmetric_mode=True)
        self._inverse_cmo = [0] * col_ct

        reverse_cuthill_mckee_mtx = numpy.ndarray(shape=(col_ct, col_ct))

        normal_matrix_non_sparse = normal_matrix.toarray()
        for i in range(col_ct):
            row_index = self._reverse_cuthill_mckee_order[i]
            self._inverse_cmo[row_index] = i
            for j in range(col_ct):
                col_index = self._reverse_cuthill_mckee_order[j]
                reverse_cuthill_mckee_mtx[i][j] = normal_matrix_non_sparse[row_index][col_index]

        self._cholesky_matrix = scipy.linalg.cholesky(reverse_cuthill_mckee_mtx,
                                                      lower=True,
                                                      overwrite_a=True,
                                                      check_finite=False)

    def apply_pose(self, pose_index, training_type='differential', ground_truth=False):
        """
        apply the pose to rigs in the scene
        :param int pose_index: the index of the pose to apply
        :param str training_type: the type of training. Valid options are differential and local_offset
        :param bool ground_truth: if True, read ground truth data instead
        """
        if self.pose_data is None:
            self.read_pose_data()
        if training_type not in self.prediction_data:
            training_types = [training_type]
            if training_type == 'differential':
                training_types.append('anchor')
            self.read_prediction_data(training_types, ground_truth=ground_truth)

        cur_pose_data = self.pose_data.iloc[pose_index]

        for mover in cur_pose_data.index:

            value = cur_pose_data[mover]

            all_movers = mc.ls(mover, recursive=True)
            for m in all_movers:
                try:
                    mc.setAttr(m, value)
                except RuntimeError as error_msg:
                    pass

        shape_data = bs.getEmptyShapeData()
        shape_data['shapes'][self.TARGET] = bs.getEmptyShape()

        if training_type == 'local_offset':
            cur_offset_data = self.prediction_data[training_type].loc[pose_index]

            shape_data['shapes'][self.TARGET] = bs.getEmptyShape()

            exclude_list = []

            for vtx_id in cur_offset_data.index:
                if int(vtx_id) in exclude_list:
                    continue
                shape_data['shapes'][self.TARGET]['offsets'][int(vtx_id)] = cur_offset_data[vtx_id]


        elif training_type == 'differential':
            cur_differential_data = self.prediction_data[training_type].loc[pose_index]
            cur_anchor_data = self.prediction_data['anchor'].loc[pose_index]

            if self._laplacian_matrix is None:
                self._cal_matrices()

            col_ct = len(self.vertex_ids['differential'])
            diff_coord = numpy.ndarray(shape=(col_ct + len(self.vertex_ids['anchor']), 1))

            real_coords = []
            for axis in range(3):
                test_index = []
                test_coord = []
                for i, di in enumerate(self.vertex_ids['differential']):
                    test_index.append(di)
                    test_coord.append(cur_differential_data[di][axis])
                    diff_coord[i] = self._valences[i] * cur_differential_data[di][axis]
                for j, anchor in enumerate(self.vertex_ids['anchor']):
                    if cur_anchor_data is not None:
                        test_coord.append(cur_anchor_data[anchor][axis])
                        diff_coord[j + col_ct] = self.anchor_weight * cur_anchor_data[anchor][axis]
                    else:
                        test_coord.append(0.0)
                        diff_coord[j + col_ct] = 0.0

                # calculates (L*) * d, where D* is the transpose of the laplacian matrix,
                # d is the differential coordinates scaled with valence
                morphed_diff_coord = self._laplacian_matrix.transpose() * diff_coord

                # reorder the coordinates based on reverse cuthill mckee reordering
                reordered_diff_coord = numpy.ndarray(shape=(col_ct, 1))
                for i in range(col_ct):
                    reordered_diff_coord[i] = morphed_diff_coord[self._reverse_cuthill_mckee_order[i]]

                # back substitution to solve two trianglar matrices
                theta_coord = scipy.linalg.solve_triangular(self._cholesky_matrix,
                                                            reordered_diff_coord,
                                                            lower=True,
                                                            overwrite_b=True,
                                                            check_finite=False)
                reordered_real_coord = scipy.linalg.solve_triangular(self._cholesky_matrix.T,
                                                                     theta_coord,
                                                                     lower=False,
                                                                     overwrite_b=True,
                                                                     check_finite=False)

                # reorder the coordinates to the original order

                real_coord = []
                for i in range(col_ct):
                    real_coord.append(reordered_real_coord[self._inverse_cmo[i]][0])

                real_coords.append(real_coord)

            for i, di in enumerate(self.vertex_ids[training_type]):
                offset = [real_coords[0][i], real_coords[1][i], real_coords[2][i]]
                shape_data['shapes'][self.TARGET]['offsets'][di] = offset

        else:
            raise NotImplementedError("The training_type %s is not implemented" % training_type)

        bs.setShapeData(self.blendshape, shape_data, shapes=[self.TARGET])
        mc.setAttr(self.blendshape + '.' + self.TARGET, 1.0)

    def calculate_error(self, training_type):
        """
        calculates the error between the prediction and ground truth. This is calculated with an assumption
        that in the current scene, there is one ground_truth rig, one rig using ml prediction
        :param str training_type: type of training. Valid options are differential and local_offset
        """
        all_meshes = mc.ls(self._mesh, recursive=True)

        ground_truth_mesh = None
        for mesh in all_meshes:
            if mesh != self._mesh:
                ground_truth_mesh = mesh
                break

        vertex_ids = self.vertex_ids[training_type]

        gt_pos = cbs.getPositions(ground_truth_mesh)
        pt_pos = cbs.getPositions(self._mesh)

        gt_data = []
        pt_data = []
        for vertex in vertex_ids:
            vector = gt_pos[vertex] - pt_pos[vertex]
            gt_data.append(vector.length())
            pt_data.append(0.0)

        gt_data = numpy.array(gt_data)
        pt_data = numpy.array(pt_data)

        self.error_history.add_sample(gt_data, pt_data)

    @property
    def mean_square_error(self):
        return self.error_history.mean_square_error()

    @property
    def max_square_error(self):
        return self.error_history.max_square_error()

    @property
    def mean_absolute_error(self):
        return self.error_history.mean_absolute_error()

    @property
    def max_absolute_error(self):
        return self.error_history.max_absolute_error()

    def go_through_all_poses(self, training_type, ground_truth=False):
        """go through all poses"""
        if self.pose_data is None:
            self.read_pose_data()
        pose_ct = len(self.pose_data.index)
        for i in range(pose_ct):
            self.apply_pose(i, training_type=training_type, ground_truth=ground_truth)
            self.calculate_error(training_type)

        config_handle = configparser.ConfigParser()
        config_handle.read(self.config)

        if not config_handle.has_section('prediction_result'):
            config_handle.add_section('prediction_result')

        scene_name = mc.file(sceneName=True, q=True)

        config_handle.set('prediction_result',
                          '%s_prediction_scene_file' % training_type,
                          scene_name)

        config_handle.set('prediction_result',
                          '%s_prediction_mean_square_error' % training_type,
                          str(self.mean_square_error))

        config_handle.set('prediction_result',
                          '%s_prediction_max_square_error' % training_type,
                          str(self.max_square_error))

        config_handle.set('prediction_result',
                          '%s_prediction_mean_absolute_error' % training_type,
                          str(self.mean_absolute_error))

        config_handle.set('prediction_result',
                          '%s_prediction_max_absolute_error' % training_type,
                          str(self.max_absolute_error))

        f = open(self.config, 'w')
        config_handle.write(f)
        f.close()



