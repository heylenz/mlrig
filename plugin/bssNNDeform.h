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

#ifndef __BSSNNDEFORM_H__
#define __BSSNNDEFORM_H__

#include <maya/MFnPlugin.h>

//	Include Maya API Proxy Headers
#include <maya/MPxNode.h>
#include <maya/MPxDeformerNode.h>

//	Include Maya API Function Set Headers
#include <maya/MFnDependencyNode.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnFloatArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFloatMatrix.h>
#include <maya/MGlobal.h>
//  Include BlueSky ProdType Node ID database
#include <mayaTypeIDs.h>

#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MPointArray.h>
#include <maya/MItGeometry.h>

#include <sys/stat.h>
#include <vector>
#include <string>

#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include "tensorflowModel.h"

typedef Eigen::SparseMatrix<float> SparseMatrix;
typedef Eigen::Triplet<float> Triplet;

enum TripletFlag {
    None = 0,
    HasValue = 1,
    HasRow = 2,
    HasColumn = 4,
    AllSet = 7
};
inline TripletFlag operator|(TripletFlag a, TripletFlag b) {
    return static_cast<TripletFlag>(static_cast<int>(a) | static_cast<int>(b));
}

class bssNNDeform : public MPxDeformerNode
{
	public:
		bssNNDeform();
		virtual ~bssNNDeform();
		static void *creator();
		
		// virtual member function overrides
		MStatus	deform(MDataBlock & data, 
					   MItGeometry & itr,
					   const MMatrix & m, 
					   unsigned int multiIndex) override;
		bool setInternalValue (const MPlug &, const MDataHandle &) override;	
		static MStatus initialize();
		
		MPxNode::SchedulingType schedulingType() const override;

		//	Input Attributes
		static MTypeId id;
		static MObject matrixInfluences_;
		static MObject numericInfluences_;
		static MObject jointInputMin_;
		static MObject jointInputMax_;
        static MObject numericInputMin_;
        static MObject numericInputMax_;
		static MObject normalizeInput_;
		static MObject differentialOutputMin_;
		static MObject differentialOutputMax_;
        static MObject differentialNormalizeOutput_;
        static MObject anchorOutputMin_;
        static MObject anchorOutputMax_;
        static MObject anchorNormalizeOutput_;
        static MObject networkInputSize_;
        static MObject differentialNetworkOutputSize_;
        static MObject anchorNetworkOutputSize_;
        static MObject affectedVertices_;
        static MObject differentialNetworkPath_;
        static MObject anchorNetworkPath_;
        static MObject differentialInputName_;
        static MObject anchorInputName_;
        static MObject differentialOutputName_;
        static MObject anchorOutputName_;
        static MObject networkType_;

        //Cholesky factorization is done in python, as it's only done once,
        //and Eigen doesn't have implementation for reverse Cuthill Mckee
        static MObject laplacianMatrixValues_;
        static MObject laplacianMatrixRows_;
        static MObject laplacianMatrixColumns_;
        static MObject choleskyMatrixValues_;
        static MObject choleskyMatrixRows_;
        static MObject choleskyMatrixColumns_;
        static MObject reverseCuthillMckeeOrder_;
        static MObject valences_;
        static MObject isDifferential_;

	private:
        const float TOL = 0.0001;
        MStatus fail(const char * err);

        enum MatrixType {Laplacian, Cholesky};
        enum MatrixValueType {Value, Row, Column};
        enum NetworkType {Tensorflow, Mxnet};
        void setSparseMatrix(const MDataHandle &dh,
                             MatrixType mType,
                             MatrixValueType vType);
        auto calCoord(std::vector<float> &diffCoord);
		std::vector<float> networkInput;
		std::vector<float> diffOutput;   //differential network output
		std::vector<float> anchorOutput;  //anchor network output
		std::vector<int> affectedVertices;
		std::vector<float> numericInputMin;
		std::vector<float> numericInputMax;

		TripletFlag laplacianFlag = None;
		TripletFlag choleskyFlag = None;
		std::vector<Triplet> laplacianMatrixTriplet;
		std::vector<Triplet> choleskyMatrixTriplet;
		std::vector<Triplet> rcmMatrixTriplet;

		int laplacianRowCt = 0;
		int laplacianColCt = 0;
		int choleskyRowCt = 0;
		int choleskyColCt = 0;
		int rcmRowCt = 0;
		int rcmColCt = 0;

		SparseMatrix laplacianMatrix;
		SparseMatrix choleskyMatrix;
		Eigen::MatrixXf choleskyMatrixDense;
		//reverse Cuthill Mckee ordering defined as Matrix
		SparseMatrix rcmMatrix;
		SparseMatrix rcmMatrixInverse;

		SparseMatrix valenceMatrix;

		SparseMatrix mulMatrix;
		NetworkType networkType;
		std::string diffInputName, diffOutputName;
		std::string anchorInputName, anchorOutputName;
		std::vector<std::string> anchorOutputNames;
		std::string diffNetworkPath, anchorNetworkPath;

		std::unique_ptr<Model> diffModel;
		std::unique_ptr<Model> anchorModel;

		bool isDifferential = true;
};

#endif    // __BSSNNDEFORM_H__
