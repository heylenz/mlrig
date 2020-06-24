////////////////////////////////////////////////////////////////////////////////
//
// I. LICENSE CONDITIONS
//
// Copyright (c) 2019 by Blue Sky Studios, Inc.
// Permission is hereby granted to use this software solely for non-commercial
// applications and purposes including academic or industrial research,
// evaluation and not-for-profit media production. All other rights are retained
// by Blue Sky Studios, Inc. For use for or in connection with commercial
// applications and purposes, including without limitation in or in connection
// with software products offered for sale or for-profit media production,
// please contact Blue Sky Studios, Inc. at
//  tech-licensing@blueskystudios.com<mailto:tech-licensing@blueskystudios.com>.
//
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
// INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY,
// NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL BLUE SKY STUDIOS, INC. OR ITS AFFILIATES BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE,EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
////////////////////////////////////////////////////////////////////////////////

#include "bssNNDeform.h"

//	Input & Output Attribute MObjects

//	Input Attribute MObjects
MObject bssNNDeform::matrixInfluences_;
MObject bssNNDeform::numericInfluences_;
MObject bssNNDeform::jointInputMin_;
MObject bssNNDeform::jointInputMax_;
MObject bssNNDeform::numericInputMin_;
MObject bssNNDeform::numericInputMax_;
MObject bssNNDeform::normalizeInput_;
MObject bssNNDeform::differentialOutputMin_;
MObject bssNNDeform::differentialOutputMax_;
MObject bssNNDeform::differentialNormalizeOutput_;
MObject bssNNDeform::differentialNetworkOutputSize_;
MObject bssNNDeform::anchorOutputMin_;
MObject bssNNDeform::anchorOutputMax_;
MObject bssNNDeform::anchorNormalizeOutput_;
MObject bssNNDeform::anchorNetworkOutputSize_;
MObject bssNNDeform::affectedVertices_;
MObject bssNNDeform::valences_;
MObject bssNNDeform::differentialNetworkPath_;
MObject bssNNDeform::anchorNetworkPath_;
MObject bssNNDeform::differentialInputName_;
MObject bssNNDeform::anchorInputName_;
MObject bssNNDeform::differentialOutputName_;
MObject bssNNDeform::anchorOutputName_;
MObject bssNNDeform::networkType_;
MObject bssNNDeform::laplacianMatrixValues_;
MObject bssNNDeform::laplacianMatrixRows_;
MObject bssNNDeform::laplacianMatrixColumns_;
MObject bssNNDeform::choleskyMatrixValues_;
MObject bssNNDeform::choleskyMatrixRows_;
MObject bssNNDeform::choleskyMatrixColumns_;
MObject bssNNDeform::reverseCuthillMckeeOrder_;
MObject bssNNDeform::isDifferential_;

bssNNDeform::bssNNDeform()
{
    diffNetworkPath = "";
    anchorNetworkPath = "";
    diffInputName = "";
    diffOutputName = "";
    anchorInputName = "";
    anchorOutputName = "";
    laplacianRowCt = 0;
    laplacianColCt = 0;
    choleskyRowCt = 0;
    choleskyColCt = 0;
    rcmRowCt = 0;
    rcmColCt = 0;
    networkType = Tensorflow;
    isDifferential = true;
}

bssNNDeform::~bssNNDeform() {};

void* bssNNDeform::creator() {
	return(new bssNNDeform());
}

auto bssNNDeform::calCoord(std::vector<float> &diffCoord) {
    float* ptr = &diffCoord[0];
    auto rVector = Eigen::Map<Eigen::VectorXf>(ptr, diffCoord.size());

    auto tempCoord = Eigen::MatrixXf(mulMatrix * rVector);
    choleskyMatrixDense.triangularView<Eigen::Lower>().solveInPlace(tempCoord);
    choleskyMatrixDense.transpose().triangularView<Eigen::Upper>().solveInPlace(tempCoord);
    auto result = rcmMatrixInverse * tempCoord;
    return Eigen::VectorXf(result);
}

MStatus	bssNNDeform::deform (MDataBlock &data, 
	                         MItGeometry &ptItr,
	                         const MMatrix & /*worldMat */, 
	                         unsigned int /*multiIndex*/) {
	MStatus status;

	float env = data.inputValue(MPxDeformerNode::envelope, &status).asFloat();
	if (!status) {
		return fail ("bssNNDeform::deform: no envelope");
	}

	float jntMin = data.inputValue(jointInputMin_, &status).asFloat();
	CHECK_MSTATUS(status);

    float jntMax = data.inputValue(jointInputMax_, &status).asFloat();
    CHECK_MSTATUS(status);

    bool normalizeInput = data.inputValue(normalizeInput_, &status).asFloat();
    CHECK_MSTATUS(status);

    bool diffNormalize = data.inputValue(differentialNormalizeOutput_, &status).asBool();
    CHECK_MSTATUS(status);

    bool anchorNormalize = data.inputValue(anchorNormalizeOutput_, &status).asBool();
    CHECK_MSTATUS(status);

    float diffMin = data.inputValue(differentialOutputMin_, &status).asFloat();
    CHECK_MSTATUS(status);

    float diffMax = data.inputValue(differentialOutputMax_, &status).asFloat();
    CHECK_MSTATUS(status);

    float anchorMin = data.inputValue(anchorOutputMin_, &status).asFloat();
    CHECK_MSTATUS(status);

    float anchorMax = data.inputValue(anchorOutputMax_, &status).asFloat();
    CHECK_MSTATUS(status);

	if (env < TOL)
	    return MStatus::kSuccess;

	// pull in the joint transforms and the other inputs:
	auto matrixInfluencesDH = data.inputValue(matrixInfluences_, &status);
	CHECK_MSTATUS(status);
	
    auto numericInfluencesDH = data.inputValue(numericInfluences_, &status);
    CHECK_MSTATUS(status);

	auto matrixInfluencesADH = MArrayDataHandle(matrixInfluencesDH);
    auto numericInfluencesADH = MArrayDataHandle(numericInfluencesDH);

    uint networkSize = 12 * matrixInfluencesADH.elementCount() +
                       numericInfluencesADH.elementCount();
    if (networkInput.size() != networkSize)
        networkInput.resize(networkSize, 0.0);

    std::vector<float>::iterator inputIter = networkInput.begin();
    if (matrixInfluencesADH.elementCount() > 0) {
        do {
            MDataHandle dh = matrixInfluencesADH.inputValue(&status);
            MFloatMatrix inputMatrix = dh.asFloatMatrix();
            for (int i=0; i<4; i++) {
                for (int j=0; j<3; j++) {
                    assert(inputIter != networkInput.end());
                    if (i != 3 || !normalizeInput) {
                        //rotation doesn't need normalization
                        *inputIter = inputMatrix[i][j];
                    }
                    else {
                        *inputIter = (inputMatrix[i][j] - jntMin) / (jntMax - jntMin);
                    }
                    inputIter ++;
                }
            }
        } while(matrixInfluencesADH.next());
	}

	if (numericInfluencesADH.elementCount() > 0) {

	    if (normalizeInput && numericInputMin.size() != numericInfluencesADH.elementCount()) {
	        MGlobal::displayError("Incomplete parameters. numericInputMin needs to be the same length as numericInfluences");
	        return MStatus::kSuccess;
	    }

        if (normalizeInput && numericInputMax.size() != numericInfluencesADH.elementCount()) {
            MGlobal::displayError("Incomplete parameters. numericInputMax needs to be the same length as numericInfluences");
            return MStatus::kSuccess;
        }
	    auto minIter = numericInputMin.begin();
	    auto maxIter = numericInputMax.begin();
        do {
            MDataHandle dh = numericInfluencesADH.inputValue(&status);
            float inputValue = dh.asFloat();
            assert(inputIter != networkInput.end());

            if (normalizeInput) {
                assert(minIter != numericInputMin.end());
                assert(maxIter != numericInputMax.end());
                *inputIter = (inputValue - *minIter) / (*maxIter - *minIter);
                minIter ++;
                maxIter ++;
                inputIter ++;
            }
            else {
                *inputIter = inputValue;
                inputIter ++;
            }
        } while(numericInfluencesADH.next());
	}

    MPointArray meshPos;
    ptItr.allPositions(meshPos);;

    if (isDifferential) {
        if (diffNetworkPath == "" ||
            anchorNetworkPath == "" ||
            diffInputName == "" ||
            anchorInputName == "" ||
            diffOutputName == "" ||
            anchorOutputName == "")
            return MStatus::kSuccess;

        if (diffModel == nullptr || anchorModel == nullptr)
            return MStatus::kSuccess;

    }
    else {
        if (anchorNetworkPath == "" ||
            anchorInputName == "" ||
            anchorOutputName == "")
            return MStatus::kSuccess;

        if (anchorModel == nullptr)
            return MStatus::kSuccess;
    }

    if (isDifferential) {
        std::vector<std::string> diffOutputNames;
        diffOutputNames.push_back(diffOutputName);

        diffModel->inference(networkInput,
                             diffOutput,
                             diffInputName,
                             diffOutputNames);
        assert(diffOutput.size() != 0);
    }

    anchorModel->inference(networkInput,
                           anchorOutput,
                           anchorInputName,
                           anchorOutputNames);


    assert(anchorOutput.size() != 0);

    if (diffNormalize) {
        std::for_each(diffOutput.begin(),
                      diffOutput.end(),
                      [&] (float& n) { n = n * (diffMax - diffMin) + diffMin;});
    }
    if (anchorNormalize) {
        std::for_each(anchorOutput.begin(),
                      anchorOutput.end(),
                      [&] (float& n) { n = n * (anchorMax - anchorMin) + anchorMin;});
    }

    if (!isDifferential) {
        std::vector<int>::iterator vtxIter = affectedVertices.begin();

        int i = 0;
        while (vtxIter != affectedVertices.end()) {
            meshPos[*vtxIter][0] = meshPos[*vtxIter][0] + env * anchorOutput[3*i];
            meshPos[*vtxIter][1] = meshPos[*vtxIter][1] + env * anchorOutput[3*i + 1];
            meshPos[*vtxIter][2] = meshPos[*vtxIter][2] + env * anchorOutput[3*i + 2];
            i ++;
            vtxIter ++;
        }

        ptItr.setAllPositions(meshPos);
        return MStatus::kSuccess;
    }

    std::vector<float>::iterator diffIt, anchorIt;
    std::vector<float> xcoord, ycoord, zcoord;

    xcoord.reserve(diffOutput.size()/3 + anchorOutput.size()/3);
    ycoord.reserve(diffOutput.size()/3 + anchorOutput.size()/3);
    zcoord.reserve(diffOutput.size()/3 + anchorOutput.size()/3);
    for (diffIt=diffOutput.begin(); diffIt!=diffOutput.end();) {
        xcoord.push_back(*diffIt);
        diffIt ++;
        ycoord.push_back(*diffIt);
        diffIt ++;
        zcoord.push_back(*diffIt);
        diffIt ++;
    }
    for (anchorIt=anchorOutput.begin(); anchorIt!=anchorOutput.end();) {
        xcoord.push_back(*anchorIt);
        anchorIt ++;
        ycoord.push_back(*anchorIt);
        anchorIt ++;
        zcoord.push_back(*anchorIt);
        anchorIt ++;
    }

    Eigen::VectorXf finalXCoord = calCoord(xcoord);
    Eigen::VectorXf finalYCoord = calCoord(ycoord);
    Eigen::VectorXf finalZCoord = calCoord(zcoord);

    std::vector<int>::iterator vtxIter = affectedVertices.begin();

    int i = 0;
    while (vtxIter != affectedVertices.end()) {
        meshPos[*vtxIter][0] = meshPos[*vtxIter][0] + env * finalXCoord[i];
        meshPos[*vtxIter][1] = meshPos[*vtxIter][1] + env * finalYCoord[i];
        meshPos[*vtxIter][2] = meshPos[*vtxIter][2] + env * finalZCoord[i];
        i ++;
        vtxIter ++;
    }

    ptItr.setAllPositions(meshPos);
	return MStatus::kSuccess;
}

MStatus bssNNDeform::fail (const char * err) {
	// Wrap this node
	MStatus status;
	MObject thisObj = thisMObject();
	MFnDependencyNode thisFn(thisObj, &status);
	MString name;
	if (status)
		name = thisFn.name ();
	else
		name = "(unknown)";

	MString errmsg("bssNNDeform node: ");
	errmsg += name;
	errmsg += " : ";
	if (err) {
		errmsg += err;
	}

	status = MStatus::kFailure;
	status.perror (errmsg);

	return status;
}

MPxNode::SchedulingType bssNNDeform::schedulingType() const {
    return MPxNode::kParallel;
}

//replace the line below with proper maya node id
MTypeId bssNNDeform::id (0001, 0001);


MStatus bssNNDeform::initialize() {
	MStatus status = MStatus::kSuccess;
	
	// set up attributes/plugs/connections:

	MFnNumericAttribute nAttrFn;
	MFnMatrixAttribute mAttrFn;
    MFnTypedAttribute tAttrFn;
    MFnEnumAttribute eAttrFn;

	/***************************************************************************
	 *
	 *	Create static output attributes for this node.
     *
	 **************************************************************************/
	matrixInfluences_ = mAttrFn.create("matrixInfluences",
	                                   "mif",
	                                   MFnMatrixAttribute::kFloat);
	CHECK_MSTATUS(mAttrFn.setDefault(MFloatMatrix()));
	CHECK_MSTATUS(mAttrFn.setArray(true));
	CHECK_MSTATUS(mAttrFn.setConnectable(true));
	CHECK_MSTATUS(addAttribute(matrixInfluences_));

	numericInfluences_ = nAttrFn.create("numericInfluences",
	                                    "nif",
	                                    MFnNumericData::kFloat,
	                                    0.0);
	CHECK_MSTATUS(nAttrFn.setArray(true));
    CHECK_MSTATUS(nAttrFn.setConnectable(true));
    CHECK_MSTATUS(addAttribute(numericInfluences_));

    jointInputMin_ = nAttrFn.create("jointInputMin",
                                    "jmn",
                                    MFnNumericData::kFloat,
                                    0.0);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(jointInputMin_));

    jointInputMax_ = nAttrFn.create("jointInputMax",
                                    "jmx",
                                    MFnNumericData::kFloat,
                                    1.0);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(jointInputMax_));

    numericInputMin_ = tAttrFn.create("numericInputMin",
                                      "nmn",
                                      MFnData::kFloatArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(numericInputMin_));

    numericInputMax_ = tAttrFn.create("numericInputMax",
                                      "nmx",
                                      MFnData::kFloatArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(numericInputMax_));

    differentialOutputMin_ = nAttrFn.create("differentialOutputMin",
                                            "dmn",
                                            MFnNumericData::kFloat,
                                            0.0);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(differentialOutputMin_));

    differentialOutputMax_ = nAttrFn.create("differentialOutputMax",
                                            "dmx",
                                            MFnNumericData::kFloat,
                                            1.0);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(differentialOutputMax_));

    normalizeInput_ = nAttrFn.create("normalizeInput",
                                     "nit",
                                     MFnNumericData::kBoolean,
                                     false);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(normalizeInput_));

    isDifferential_ = nAttrFn.create("isDifferential",
                                     "idl",
                                     MFnNumericData::kBoolean,
                                     true);
    CHECK_MSTATUS(nAttrFn.setInternal(true));
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(isDifferential_));

    differentialNormalizeOutput_ = nAttrFn.create("differentialNormalizeOutput",
                                                  "dml",
                                                  MFnNumericData::kBoolean,
                                                  false);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(differentialNormalizeOutput_));

    differentialNetworkOutputSize_ = nAttrFn.create("differentialNetworkOutputSize",
                                                    "dms",
                                                    MFnNumericData::kInt,
                                                    0);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(differentialNetworkOutputSize_));

    anchorOutputMin_ = nAttrFn.create("anchorOutputMin",
                                      "amn",
                                      MFnNumericData::kFloat,
                                      0.0);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(anchorOutputMin_));

    anchorOutputMax_ = nAttrFn.create("anchorOutputMax",
                                      "amx",
                                      MFnNumericData::kFloat,
                                      1.0);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(anchorOutputMax_));

    anchorNormalizeOutput_ = nAttrFn.create("anchorNormalizeOutput",
                                            "aml",
                                            MFnNumericData::kBoolean,
                                            false);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(anchorNormalizeOutput_));

    anchorNetworkOutputSize_ = nAttrFn.create("anchorNetworkOutputSize",
                                              "ams",
                                              MFnNumericData::kInt,
                                              0);
    CHECK_MSTATUS(nAttrFn.setKeyable(false));
    CHECK_MSTATUS(nAttrFn.setStorable(true));
    CHECK_MSTATUS(nAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(anchorNetworkOutputSize_));

    affectedVertices_ = tAttrFn.create("affectedVertices",
                                       "afv",
                                       MFnData::kIntArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(affectedVertices_));

    laplacianMatrixValues_ = tAttrFn.create("laplacianMatrixValues",
                                            "lmv",
                                            MFnData::kFloatArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(laplacianMatrixValues_));

    laplacianMatrixRows_ = tAttrFn.create("laplacianMatrixRows",
                                          "lmr",
                                          MFnData::kIntArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(laplacianMatrixRows_));

    laplacianMatrixColumns_ = tAttrFn.create("laplacianMatrixColumns",
                                             "lmc",
                                             MFnData::kIntArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(laplacianMatrixColumns_));

    choleskyMatrixValues_ = tAttrFn.create("choleskyMatrixValues",
                                           "cmv",
                                           MFnData::kFloatArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(choleskyMatrixValues_));

    choleskyMatrixRows_ = tAttrFn.create("choleskyMatrixRows",
                                         "cmr",
                                         MFnData::kIntArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(choleskyMatrixRows_));

    choleskyMatrixColumns_ = tAttrFn.create("choleskyMatrixColumns",
                                            "cmc",
                                            MFnData::kIntArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(choleskyMatrixColumns_));

    reverseCuthillMckeeOrder_ = tAttrFn.create("reverseCuthillMckeeOrder",
                                               "rcm",
                                               MFnData::kIntArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(reverseCuthillMckeeOrder_));

    valences_ = tAttrFn.create("valences",
                               "vcs",
                               MFnData::kFloatArray);
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setStorable(true));
    CHECK_MSTATUS(tAttrFn.setWritable(true));
    CHECK_MSTATUS(addAttribute(valences_));

    differentialNetworkPath_ = tAttrFn.create("differentialNetworkPath",
                                              "dwp",
                                              MFnData::kString);
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(addAttribute(differentialNetworkPath_));

    anchorNetworkPath_ = tAttrFn.create("anchorNetworkPath",
                                        "awp",
                                        MFnData::kString);
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(addAttribute(anchorNetworkPath_));

    differentialInputName_ = tAttrFn.create("differentialInputName",
                                            "din",
                                            MFnData::kString);
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(addAttribute(differentialInputName_));

    anchorInputName_ = tAttrFn.create("anchorInputName",
                                      "ain",
                                      MFnData::kString);
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(addAttribute(anchorInputName_));

    differentialOutputName_ = tAttrFn.create("differentialOutputName",
                                             "don",
                                             MFnData::kString);
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(addAttribute(differentialOutputName_));

    anchorOutputName_ = tAttrFn.create("anchorOutputName",
                                       "aon",
                                       MFnData::kString);
    CHECK_MSTATUS(tAttrFn.setKeyable(false));
    CHECK_MSTATUS(tAttrFn.setInternal(true));
    CHECK_MSTATUS(addAttribute(anchorOutputName_));

    networkType_ = eAttrFn.create("networkType", "nwt", 0, &status);
    eAttrFn.addField("tensorflow", 0);
    eAttrFn.addField("mxnet", 1);
    CHECK_MSTATUS(eAttrFn.setInternal(true));
    CHECK_MSTATUS(addAttribute(networkType_));

	/***************************************************************************
	 *
	 *	Define our attributeAffects properties
	 *
	 **************************************************************************/
	CHECK_MSTATUS(attributeAffects(matrixInfluences_, outputGeom));
	CHECK_MSTATUS(attributeAffects(numericInfluences_, outputGeom));
	CHECK_MSTATUS(attributeAffects(jointInputMin_, outputGeom));
	CHECK_MSTATUS(attributeAffects(jointInputMax_, outputGeom));
    CHECK_MSTATUS(attributeAffects(numericInputMin_, outputGeom));
    CHECK_MSTATUS(attributeAffects(numericInputMax_, outputGeom));
	CHECK_MSTATUS(attributeAffects(differentialOutputMin_, outputGeom));
	CHECK_MSTATUS(attributeAffects(differentialOutputMax_, outputGeom));
    CHECK_MSTATUS(attributeAffects(differentialNormalizeOutput_, outputGeom));
    CHECK_MSTATUS(attributeAffects(differentialNetworkOutputSize_, outputGeom));
    CHECK_MSTATUS(attributeAffects(anchorOutputMin_, outputGeom));
    CHECK_MSTATUS(attributeAffects(anchorOutputMax_, outputGeom));
    CHECK_MSTATUS(attributeAffects(anchorNormalizeOutput_, outputGeom));
    CHECK_MSTATUS(attributeAffects(anchorNetworkOutputSize_, outputGeom));

	CHECK_MSTATUS(attributeAffects(affectedVertices_, outputGeom));
	CHECK_MSTATUS(attributeAffects(differentialNetworkPath_, outputGeom));
	CHECK_MSTATUS(attributeAffects(anchorNetworkPath_, outputGeom));
	CHECK_MSTATUS(attributeAffects(laplacianMatrixValues_, outputGeom));
    CHECK_MSTATUS(attributeAffects(laplacianMatrixRows_, outputGeom));
    CHECK_MSTATUS(attributeAffects(laplacianMatrixColumns_, outputGeom));
    CHECK_MSTATUS(attributeAffects(choleskyMatrixValues_, outputGeom));
    CHECK_MSTATUS(attributeAffects(choleskyMatrixRows_, outputGeom));
    CHECK_MSTATUS(attributeAffects(choleskyMatrixColumns_, outputGeom));
    CHECK_MSTATUS(attributeAffects(reverseCuthillMckeeOrder_, outputGeom));
    CHECK_MSTATUS(attributeAffects(valences_, outputGeom));
	CHECK_MSTATUS(attributeAffects(networkType_, outputGeom));

	//  Return the status back to Maya
	return(status);
}

void bssNNDeform::setSparseMatrix(const MDataHandle &dh,
                                  MatrixType mType,
                                  MatrixValueType vType) {
    //TODO(stevens): maya's MDataHandle::data is not a const function
    //this is a workaround
    MDataHandle copyHandle(dh);
    MObject dataObj = copyHandle.data();

    //TODO(stevens): maya's MFnFloatArrayData and MFnIntArrayData have made
    //operator & private, so I can't use a MFnData pointer, and have to create
    //both even though only one of them is used
    MFnFloatArrayData valData(dataObj);
    MFnIntArrayData indexData(dataObj);
    uint length;
    std::vector<Triplet>* triplet = NULL;
    TripletFlag* flag = NULL;
    int* rowCt = NULL;
    int* colCt = NULL;
    SparseMatrix* matrix;

    if (vType == Value) {
        if (!dataObj.hasFn(MFn::kFloatArrayData))
            return;
        length = valData.length();
    }
    else {
        if (!dataObj.hasFn(MFn::kIntArrayData))
            return;
        length = indexData.length();
    }

    if (mType == Laplacian) {
        triplet = &laplacianMatrixTriplet;
        flag = &laplacianFlag;
        rowCt = &laplacianRowCt;
        colCt = &laplacianColCt;
        matrix = &laplacianMatrix;
    }
    else {
        triplet = &choleskyMatrixTriplet;
        flag = &choleskyFlag;
        rowCt = &choleskyRowCt;
        colCt = &choleskyColCt;
        matrix = &choleskyMatrix;
    }

    if (triplet->size() != length) {
        triplet->resize(length);
        *flag = None;
    }

    for (uint i=0; i<length; i++) {
        Triplet curTriplet = (*triplet)[i];
        int row = curTriplet.row();
        int col = curTriplet.col();
        float value = curTriplet.value();

        if (vType == Value) {
            value = valData[i];
            *flag = (*flag) | HasValue;
        }
        else if (vType == Row) {
            row = indexData[i];
            *flag = (*flag) | HasRow;
            if (indexData[i] >= *rowCt) {
                *rowCt = indexData[i] + 1;
            }
        }
        else {
            col = indexData[i];
            *flag = (*flag) | HasColumn;
            if (indexData[i] >= *colCt) {
                *colCt = indexData[i] + 1;
            }
        }

        (*triplet)[i] = Triplet(row, col, value);
    }

    if ((*flag) == AllSet) {
        matrix->resize(*rowCt, *colCt);
        matrix->setFromTriplets(triplet->begin(), triplet->end());

        if (mType == Laplacian) {
            if (rcmMatrix.size() != 0 && valenceMatrix.size() != 0)
                mulMatrix = rcmMatrix * matrix->transpose() * valenceMatrix;
        }
        else {
            choleskyMatrixDense = Eigen::MatrixXf(*matrix);
        }
    }
}

bool bssNNDeform::setInternalValue (const MPlug &plug, const MDataHandle &dh) {
    MStatus status;
    if (plug == differentialNetworkPath_) {
        if (networkType == Tensorflow) {
            MString str = dh.asString();
            diffNetworkPath = std::string(str.asChar());
            diffModel = std::make_unique<TensorflowModel>(diffNetworkPath);
        }
        else {
            // to be implemented
        }
    }
    else if (plug == anchorNetworkPath_) {
        if (networkType == Tensorflow) {
            MString str = dh.asString();
            anchorNetworkPath = std::string(str.asChar());
            anchorModel = std::make_unique<TensorflowModel>(anchorNetworkPath);
        }
        else {
            // to be implemented
        }
    }
    else if (plug == differentialInputName_) {
        MString str = dh.asString();
        diffInputName = std::string(str.asChar());
    }
    else if (plug == differentialOutputName_) {
        MString str = dh.asString();
        diffOutputName = std::string(str.asChar());
    }
    else if (plug == anchorInputName_) {
        MString str = dh.asString();
        anchorInputName = std::string(str.asChar());
    }
    else if (plug == anchorOutputName_) {
        MString str = dh.asString();
        anchorOutputName = std::string(str.asChar());
        MStringArray tokens;
        str.split(',', tokens);

        anchorOutputNames.clear();
        for (int i=0; i<tokens.length(); i++) {
            anchorOutputNames.push_back(std::string(tokens[i].asChar()));
        }
    }
    else if (plug == affectedVertices_) {
        MDataHandle copyHandle(dh);
        MObject dataObj = copyHandle.data();
        if (dataObj.hasFn(MFn::kIntArrayData)) {
            MFnIntArrayData arrayData(dataObj);
            affectedVertices.resize(arrayData.length());
            for (uint i=0; i<arrayData.length(); i++) {
                affectedVertices[i] = arrayData[i];
            }
        }
    }
    else if (plug == numericInputMin_) {
        MDataHandle copyHandle(dh);
        MObject dataObj = copyHandle.data();
        if (dataObj.hasFn(MFn::kFloatArrayData)) {
            MFnFloatArrayData arrayData(dataObj);
            numericInputMin.resize(arrayData.length());
            for (uint i=0; i<arrayData.length(); i++) {
                numericInputMin[i] = arrayData[i];
            }
        }
    }
    else if (plug == numericInputMax_) {
        MDataHandle copyHandle(dh);
        MObject dataObj = copyHandle.data();
        if (dataObj.hasFn(MFn::kFloatArrayData)) {
            MFnFloatArrayData arrayData(dataObj);
            numericInputMax.resize(arrayData.length());
            for (uint i=0; i<arrayData.length(); i++) {
                numericInputMax[i] = arrayData[i];
            }
        }
    }
    else if (plug == laplacianMatrixValues_) {
        setSparseMatrix(dh, Laplacian, Value);
    }
    else if (plug == laplacianMatrixRows_) {
        setSparseMatrix(dh, Laplacian, Row);
    }
    else if (plug == laplacianMatrixColumns_) {
        setSparseMatrix(dh, Laplacian, Column);
    }
    else if (plug == choleskyMatrixValues_) {
        setSparseMatrix(dh, Cholesky, Value);
    }
    else if (plug == choleskyMatrixRows_) {
        setSparseMatrix(dh, Cholesky, Row);
    }
    else if (plug == choleskyMatrixColumns_) {
        setSparseMatrix(dh, Cholesky, Column);
    }
    else if (plug == reverseCuthillMckeeOrder_) {
        MDataHandle copyHandle(dh);
        MObject dataObj = copyHandle.data();
        if (dataObj.hasFn(MFn::kIntArrayData)) {
            std::vector<Triplet> triplets;
            std::vector<Triplet> inverseTriplets;
            MFnIntArrayData arrayData(dataObj);
            for (uint i=0; i<arrayData.length(); i++) {
                triplets.push_back(Triplet(i, arrayData[i], 1.0));
                inverseTriplets.push_back(Triplet(arrayData[i], i, 1.0));
            }
            rcmMatrix.resize(arrayData.length(), arrayData.length());
            rcmMatrix.setFromTriplets(triplets.begin(), triplets.end());
            rcmMatrixInverse.resize(arrayData.length(), arrayData.length());
            rcmMatrixInverse.setFromTriplets(inverseTriplets.begin(),
                                             inverseTriplets.end());
            if (laplacianMatrix.size() != 0 && valenceMatrix.size() != 0)
                mulMatrix = rcmMatrix * laplacianMatrix.transpose() * valenceMatrix;
        }
    }
    else if (plug == valences_) {
        MDataHandle copyHandle(dh);
        MObject dataObj = copyHandle.data();
        if (dataObj.hasFn(MFn::kFloatArrayData)) {
            std::vector<Triplet> triplets;
            MFnFloatArrayData arrayData(dataObj);
            for (uint i=0; i<arrayData.length(); i++) {
                triplets.push_back(Triplet(i, i, arrayData[i]));
            }
            valenceMatrix.resize(arrayData.length(), arrayData.length());
            valenceMatrix.setFromTriplets(triplets.begin(), triplets.end());
            if (laplacianMatrix.size() != 0 && rcmMatrix.size() != 0) {
                mulMatrix = rcmMatrix * laplacianMatrix.transpose() * valenceMatrix;
            }
        }
    }
    else if (plug == networkType_) {
        short enumValue = dh.asShort();
        if (enumValue == 0) // tensorflow
            networkType = Tensorflow;
        else
            networkType = Mxnet;
    }
    else if (plug == isDifferential_) {
        isDifferential = dh.asBool();
    }
    return MPxDeformerNode::setInternalValue(plug, dh);
}
// Initialize/Uninitialize Plugin

MStatus initializePlugin(MObject obj) {
    MStatus stat;
    MFnPlugin plugin(obj, "BlueSkyStudios", "1.0", "Any");
    stat = plugin.registerNode("bssNNDeform",
                               bssNNDeform::id,
                               bssNNDeform::creator,
                               bssNNDeform::initialize,
                               MPxNode::kDeformerNode);

    return stat;
}

MStatus uninitializePlugin( MObject obj ) {
    MStatus stat;
    MFnPlugin plugin(obj);
    stat = plugin.deregisterNode(bssNNDeform::id);
    return stat;
}
