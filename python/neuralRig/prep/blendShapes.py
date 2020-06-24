
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


import maya.cmds as mc
import maya.OpenMaya as om
import maya.OpenMayaAnim as oma
import re
import cPickle
import json
import copy as copy

import time
import os

def getEmptyShapeData():
    return {'shapes':{}, 'setMembers':True, 'baseWeights':{}, 'nodeType':'blendShape'}
def getEmptyShape():
    return {'offsets':{}, 'weights':{}, 'shapeIndex':None}

def setBlendShapeData(node,
                      shapeData,
                      inputIndex=0,
                      shapes=None):
    """
    sets the shape data onto a blendShape node.
    :param str node: the blendShape to add the shapes to.
    :param dict shapeData: the shape data to apply to the node.
    :param list shapes: if given, only modify given target names
    """
    nodeType = mc.nodeType(node)
    inputShape = mc.deformer(node, g=True, q=True)
    shapeAliasLookup = getShapeAliasLookup(node)

    if not 'shapes' in shapeData:
        print(procedureName + ':  shapeData does not have a "shapes" key.  Returning now...')
        return

    for shapeAlias in shapeData['shapes']:
        if shapes and shapeAlias not in shapes:
            continue

        # read the information stored for this shape
        targetData = shapeData['shapes'][shapeAlias]
        targetOffsets = targetData["offsets"]
        targetWeights = targetData["weights"]
        shapeIndex = shapeAliasLookup.get(shapeAlias, None)

        # if the shape doesn't already exist, create it at the end of the list
        newShape = False
        if shapeIndex is None:
            newShape = True
            weightIndices = mc.getAttr(node + ".weight", mi=True)
            if weightIndices is None:
                shapeIndex = 0
            else:
                shapeIndex = weightIndices[-1] + 1

            mc.addMultiInstance(node + '.weight[' + str(shapeIndex) + ']')
            if shapeAlias[0] != '[':
                mc.aliasAttr(shapeAlias.strip(), node + ".weight[" + str(shapeIndex) + "]")

        # iterate through the offset dictionary
        pointList = []
        componentList = []
        #speed optimization
        shapeComponentsToUse = {}
        
        for pointIndex in targetOffsets:
            pointData = targetOffsets[pointIndex]                       
            pointList.append((pointData[0], pointData[1], pointData[2], 1.0))
            componentList.append('vtx[' + str(pointIndex) + ']')
        
        # create the element by calling getAttr
        try:
            mc.getAttr(node + '.inputTarget[' + str(inputIndex) + ']')
        except:
            pass
        try:
            mc.getAttr(node + '.inputTarget[' + str(inputIndex) + '].inputTargetGroup[' + str(shapeIndex) + ']')
        except:
            pass

        shapeAttr = node + ".inputTarget[" + str(inputIndex) + "].inputTargetGroup[" + str(shapeIndex) + "]"
        mc.setAttr(shapeAttr + ".inputTargetItem[6000].inputPointsTarget", len(componentList), type="pointArray", *pointList)
        mc.setAttr(shapeAttr + ".inputTargetItem[6000].inputComponentsTarget", len(componentList), type="componentList", *componentList)

        tAttrs = mc.listAttr(shapeAttr, m=True, string='targetWeights')
        if tAttrs != None:
            for a in tAttrs:
                mc.removeMultiInstance((node + '.' + a), b=True)
        # set the weights
        for weight in targetWeights:
            mc.setAttr(shapeAttr + ".targetWeights[" + str(weight) + "]", targetWeights[weight])

def getShapeAliasLookup(node):
    """
    Builds a lookup dictionary that maps a blendShape node's weight
    attribute alias name to the weight attribute's index.
    """

    aliasLookup = {}
    weightIndices = mc.getAttr(node + ".weight", mi=True)
    if weightIndices:
        for weightIndex in weightIndices:
            attributeAlias = mc.aliasAttr(node + ".weight[" + str(weightIndex) + "]", q=True)
            if attributeAlias:
                aliasLookup[attributeAlias] = weightIndex
            else:
                aliasLookup['[' + str(weightIndex) + ']'] = weightIndex

    return aliasLookup