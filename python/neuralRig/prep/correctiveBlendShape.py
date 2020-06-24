
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
import maya.OpenMaya as OpenMaya
import maya.api.OpenMaya as om
import copy
import ml.neuralRig.prep.blendShapes as bs
import itertools
import numpy


def makeCorrectiveFromSculpt(resultMesh,
                             sculptMesh,
                             blendShapeNode,
                             targetName,
                             tol=0.0001):
    """
    calculates offset in local space before linear skin blending
    :param str resultMesh: the mesh obj that has only the linear skin blending node and a blendShape node to store local offset
    :param str scupltMesh: the deformed mesh resulted from a full rig with all deformers
    :param str blendShapeNode: name of the blendShape node
    :param str targetName: name of the blendShape target to temporarily store local offset 
    :param float tol: float comparison tolerance 
    """
    
    if mc.nodeType(resultMesh) == 'transform':
        resultMesh = mc.listRelatives(resultMesh, children=True, path=True)
        resultMesh = mc.ls(resultMesh, noIntermediate=True)[0]
    
    if mc.nodeType(sculptMesh) == 'transform':
        sculptMesh = mc.listRelatives(sculptMesh, children=True, path=True)
        sculptMesh = mc.ls(sculptMesh, noIntermediate=True)[0]

    # get the result and sculpt positions
    # and the offset vectors between the two
    targetWeight = None
    weightAttr = '%s.%s' % (blendShapeNode, targetName)
    if mc.objExists(weightAttr):
        targetWeight = mc.getAttr(weightAttr)
        mc.setAttr(weightAttr, 0.0)
    resultPos = getPositions(resultMesh)
    sculptPos = getPositions(sculptMesh)
    if targetWeight:
        mc.setAttr(weightAttr, targetWeight)
    # if the number of points dont match, raise an Exception
    if len(resultPos) != len(sculptPos):
        raise RuntimeError('the number of result components (' + len(resultPos) + ') does not match the number of sculpt components (' + len(sculptPos) + ')')

    offsetVec = [e[0] - e[1] for e in zip(sculptPos, resultPos)]    
    
    prunedResultPos = []
    prunedResultInd = []
    prunedOffsetVec = []

    resultInd = None

    for index, vec in enumerate(offsetVec):
        if vec.length() <= tol:
            continue
        prunedResultPos.append(resultPos[index])
        prunedResultInd.append(index)
        prunedOffsetVec.append(vec)

    if not prunedResultInd:
        print("makeCorrectiveFromSculpt: Source and Destination geometries are in sync....No action taken!")
        return {}

    # run makeLinearCorrective on the computed offsets
    sData = makeLinearCorrective(prunedOffsetVec,
                                 blendShapeNode,
                                 targetName,
                                 deformedGeometry=resultMesh,
                                 resultPos=prunedResultPos,
                                 resultInd=prunedResultInd)
    return sData

def makeLinearCorrective(offsetVec,
                         blendShapeNode,
                         targetName,
                         deformedGeometry=None,
                         resultPos=None,
                         resultInd=None):

    overrideTargetName = None
    shapeData = None
        
    matricesResult = genMatrix(blendShapeNode,
                               deformedGeometry=deformedGeometry,
                               resultPos=resultPos,
                               resultInd=resultInd,
                               targetName=overrideTargetName)

    invShapeMatrices = matricesResult['invShapeMatrices']
    nonInvertablePointsSet = matricesResult['nonInvertablePointsSet']
    
    # remove any invalid inverseMatrices from consideration:
    offsetVecToUse = []
    invShapeMatricesToUse = {}
    resultIndToUse = []
    for index, ov in itertools.izip(resultInd, offsetVec):
        
        if index in nonInvertablePointsSet:
            continue
        
        invShapeMatricesToUse[index] = invShapeMatrices[index]
        offsetVecToUse.append(ov)
        resultIndToUse.append(index)
    
    optionalData = {}
    optionalData[targetName] = dict(zip(resultIndToUse, offsetVecToUse))
    
    applyMatricesResult = applyMatricesToTransformBlendShapeNodeOffsets(
        blendShapeNode,
        invShapeMatricesToUse,
        shapeData=shapeData,
        targets=[targetName],
        optionalData=optionalData,
        matrixOp=matrixOp_makeLinearCorrective
    )
    
    indicesToUse = applyMatricesResult['matrixOpResults'][targetName]
    deltaOffsetsModificationsValues = [applyMatricesResult['matrixOpResults'][targetName][x]['multResult'] for x in indicesToUse]
    
    deltaOffsetsModifications = dict(zip(indicesToUse, deltaOffsetsModificationsValues))
    
    return deltaOffsetsModifications


def getPositions(obj):
    """get the positions of all points of given geometry.
    """
    selectionList = om.MSelectionList()
    selectionList.add(obj)
    geomObj = selectionList.getDependNode(0)
    pos = None
    geomIt = om.MFnMesh(geomObj)
    pos = geomIt.getPoints()
    return pos
    

def invertMatrices(matrices):
    invMatrices = {}
        
    nonInvertablePoints = []
    nonInvertablePointsSet = set()
    
    for ii in matrices:
        invMatrix = None
        
        try:
            invMatrix = numpy.linalg.inv(matrices[ii])
        except:
            nonInvertablePoints.append(str(ii))
            nonInvertablePointsSet.add(ii)
        else:
            invMatrices[ii] = invMatrix
        
    return invMatrices, nonInvertablePointsSet, nonInvertablePoints


def genMatrix(blendShapeNode,
              deformedGeometry=None,
              resultPos=None,
              resultInd=None,
              targetName=None):
    """
    This procedure is meant to return transform matrices which describe 
    how a 1-unit offset in each X, Y, Z direction actually 
    transforms to an offset at the end of the subsequent deformer stack.
    
    :param str blendShapeNode:  the blendShape node whose offsets are in question
    :param str deformedGeometry: the name of the mesh that's deformed by blendShapeNode
    :param str targetName: the name of the blendShape target
    :param list resultPos:  the default positions of the points in resultCmp
    :param list resultInd:  the indices of the points in resultCmp.  For resultCmp element "<mesh>.vtx[5]", its resultInd element would be 5, for example. 
    """
    removeTarget = False

    if not targetName:
        targetName = 'CORRECTIVE_DUMMY_TARGET'
        removeTarget = True
    # initialize empty shape data
    shapeData = bs.getEmptyShapeData()
    shape = bs.getEmptyShape()
    shapeData['shapes'][targetName] = shape
    shape['offsets'] = {}

    if not mc.objExists('%s.%s' % (blendShapeNode, targetName)):
        shapeIndex = 0
        weightIndices = mc.getAttr(blendShapeNode + ".weight", multiIndices=True)
        if weightIndices:
            shapeIndex = weightIndices[-1] + 1
        attr = '%s.weight[%i]' % (blendShapeNode, shapeIndex)
        mc.getAttr(attr)
        mc.aliasAttr(targetName, attr)
        mc.setAttr(attr, 1.0)

    # get the x, y and z basis vectors for each point
    perAxisUnitOffsetVectors = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)] # X, Y, Z.
    axes = []
    
    for ii in range(3):
        for ind in resultInd:
            shape['offsets'][ind] = perAxisUnitOffsetVectors[ii]

        bs.setBlendShapeData(blendShapeNode, shapeData)
        currentAxisOffsetPos = getPositions(deformedGeometry)
        
        newCurrentAxisOffsetPos = [currentAxisOffsetPos[j] for j in resultInd]
        currentAxes = [e[0] - e[1] for e in zip(newCurrentAxisOffsetPos, resultPos)]    
        axes.append(currentAxes)
    
    if removeTarget:
        mc.aliasAttr(blendShapeNode + '.' + targetName, remove=True)
        mc.removeMultiInstance(blendShapeNode + '.weight[' + str(shapeIndex) + ']', b=True)
        mc.removeMultiInstance(blendShapeNode + '.inputTarget[0].inputTargetGroup[' + str(shapeIndex) + '].inputTargetItem[6000]')
        mc.removeMultiInstance(blendShapeNode + '.inputTarget[0].inputTargetGroup[' + str(shapeIndex) + ']')

    xAxes = axes[0]
    yAxes = axes[1]
    zAxes = axes[2]
    
    nonInvertablePoints = []
    nonInvertablePointsSet = set()

    # use the basis vectors to compute the final position
    
    # calculate the shapeMatrix first:
    matrices = {}
    for index, xAxis, yAxis, zAxis in itertools.izip(resultInd, xAxes, yAxes, zAxes):
        
        shapeMatrix = numpy.array([[xAxis[0], xAxis[1], xAxis[2], 0],
                                   [yAxis[0], yAxis[1], yAxis[2], 0],
                                   [zAxis[0], zAxis[1], zAxis[2], 0],
                                   [0,        0,        0,        1]])
        matrices[index] = shapeMatrix
       
    result = {}
    result['matrices'] = matrices

    invShapeMatrices = {}
    # calculate the invShapeMatrix first:
    invShapeMatrices, nonInvertablePointsSet, nonInvertablePoints = invertMatrices(matrices)
            
    result['invShapeMatrices'] = invShapeMatrices
    result['nonInvertablePoints'] = nonInvertablePoints
    result['nonInvertablePointsSet'] = nonInvertablePointsSet  
    return result

     
def matrixOp_makeLinearCorrective(currentMatrix, 
                                  cv, 
                                  optionalData=None):
     
    ov = optionalData
    
    multResult = None
    addResult = None
    if(ov):
        multResult = numpy.dot(currentMatrix, numpy.append(ov, 0))
        addResult = multResult + numpy.append(cv, 0)
    else:
        # pass-through if ov is not supplied:
        multResult = None
        addResult = cv
    
    result = {'multResult':multResult[:-1], 'addResult':addResult[:-1]}
        
    # in this case:  the goal is to modify the cv by the offsetValue modifed by the matrix:  so the addResult is the shapeOffsetReplacement result:
    result['shapeOffsetReplacement'] = addResult
    return result
     
def applyMatricesToTransformShapeDataOffsets(
    shapeData,
    matrices,
    targets=None,
    matrixOp=matrixOp_makeLinearCorrective,
    optionalData=None
    ):
    targetsToUse = None
    if targets:
        targetsSet = set(targets)
        shapeDataTargetsSet = set(shapeData['shapes'])
        commonTargetsSet = targetsSet & shapeDataTargetsSet
        targetsToUse = list(commonTargetsSet)
    else:
        targetsToUse = shapeData['shapes']
        
    matrixOpResults = {}
    errorPoints = {}
    shapeOffsetsAll = {}
    for currentShape in targetsToUse:
        shapeOffsetsAll[currentShape] = {}
        matrixOpResults[currentShape] = {}

        shapeOffsets = shapeData['shapes'][currentShape]['offsets']
        
        for index, currentMatrix in matrices.iteritems():        
            cv = shapeOffsets.get(index, (0, 0, 0))
            if(optionalData):
                currentOptionalData = optionalData[currentShape][index]
            else:
                currentOptionalData = None
                
            currentMatrixOpResult = matrixOp(currentMatrix, cv, optionalData=currentOptionalData)
            matrixOpResults[currentShape][index] = currentMatrixOpResult
            
            # modify the current shapeOffset with the replacement value from the matrixOp:
            shapeOffsets[index] = currentMatrixOpResult['shapeOffsetReplacement']
    
        shapeOffsetsAll[currentShape] = shapeOffsets
    
    result = {}
    result['shapeOffsetsAll'] = shapeOffsetsAll
    result['errorPoints'] = errorPoints          
    result['matrixOpResults'] = matrixOpResults      
    result['targetsToUse'] = targetsToUse     
         
    return result
     
def applyMatricesToTransformBlendShapeNodeOffsets(
    blendShapeNode,
    matrices,
    shapeData=None,
    targets=None,
    optionalData=None,
    matrixOp=matrixOp_makeLinearCorrective
    ):
    applyShapeData = bs.getEmptyShapeData()
    for target in targets:
        shape = bs.getEmptyShape()
        applyShapeData['shapes'][target] = shape

    applyMatricesResult = applyMatricesToTransformShapeDataOffsets(
        applyShapeData,
        matrices,
        targets=targets,
        optionalData=optionalData,
        matrixOp=matrixOp
    )

    return applyMatricesResult

