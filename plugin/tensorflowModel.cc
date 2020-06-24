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

#include "tensorflowModel.h"

TensorflowModel::TensorflowModel(const std::string& networkPath) {
    std::ifstream file(networkPath, std::ios::binary | std::ios::ate);
    TF_Buffer* buffer = nullptr;
    if (file.is_open()) {
        auto size = file.tellg();
        std::unique_ptr<char[]> data = std::make_unique<char[]>(size);
        file.seekg(0, std::ios::beg);
        file.read(data.get(), size);
        file.close();

        buffer = TF_NewBufferFromString(data.get(), size);
    }
    if (buffer) {
        status = TF_NewStatus();
        graph = TF_NewGraph();

        TF_SessionOptions* options = TF_NewSessionOptions();

        uint core_num = std::thread::hardware_concurrency();
        uint8_t intra_op_parallelism_threads = (uint8_t)core_num;
        uint8_t inter_op_parallelism_threads = 2;

        //TODO(stevens): this is hacky, but tensorflow C doesn't have api to
        //serialize a ConfigProto object, so it needs to be serialized in
        //python and hard coded here
        uint8_t buf[] = {0x10,
                         intra_op_parallelism_threads,
                         0x28,
                         inter_op_parallelism_threads};
        TF_SetConfig(options, buf, sizeof(buf), status);

        session = TF_NewSession(graph, options, status);
        TF_DeleteSessionOptions(options);

        TF_ImportGraphDefOptions* graphOptions = TF_NewImportGraphDefOptions();
        TF_GraphImportGraphDef(graph, buffer, graphOptions, status);
        TF_DeleteImportGraphDefOptions(graphOptions);
        TF_DeleteBuffer(buffer);
    }
    else {
        status = nullptr;
        graph = nullptr;
        session = nullptr;
    }
}
TensorflowModel::~TensorflowModel() {
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

void TensorflowModel::inference(const std::vector<float>& inputs,
                                std::vector<float>& outputs,
                                const std::string& inputName,
                                const std::vector<std::string>& outputNames) {
    TF_Output inputTensorOp;
    inputTensorOp.oper = TF_GraphOperationByName(graph, inputName.c_str());
    inputTensorOp.index = 0;

    auto type = TF_OperationOutputType(inputTensorOp);

    int num_dims = TF_GraphGetTensorNumDims(graph, inputTensorOp, status);

    std::unique_ptr<int64_t[]> dims = std::make_unique<int64_t[]>(num_dims);
    TF_GraphGetTensorShape(graph, inputTensorOp, dims.get(), num_dims, status);

    for (int i=0; i<num_dims; i++) {
        //TODO(stevens): in the future, this should be calculated using inputs
        if (dims[i] < 0)
            dims[i] = 1;
    }
    int inputSize = inputs.size();

    auto deallocator = [](void*, size_t, void*){};

    TF_Tensor* inputTensors[1] = {TF_NewTensor(type,
                                               dims.get(),
                                               num_dims,
                                               (void*)inputs.data(),
                                               sizeof(float)*inputSize,
                                               deallocator,
                                               nullptr)};

    int outputSize = outputNames.size();
    std::unique_ptr<TF_Output[]> outputTensorOps = std::make_unique<TF_Output[]>(outputSize);

    TF_Output outputTensorOp;

    for (int i=0; i<outputSize; i++) {
        outputTensorOps[i].oper = TF_GraphOperationByName(graph, outputNames[i].c_str());
        outputTensorOps[i].index = 0;
    }

    std::unique_ptr<TF_Tensor*[]> outputTensors = std::make_unique<TF_Tensor*[]>(outputSize);
    TF_SessionRun(session,
                  nullptr,
                  &inputTensorOp,
                  inputTensors,
                  1,
                  outputTensorOps.get(),
                  outputTensors.get(),
                  outputSize,
                  nullptr,
                  0,
                  nullptr,
                  status);
    TF_DeleteTensor(inputTensors[0]);

    for (int i=0; i<outputSize; i++) {
        assert(outputTensors[i] != nullptr);
        auto rawData = TF_TensorData(outputTensors[i]);
        size_t size = TF_TensorByteSize(outputTensors[i])/
                      TF_DataTypeSize(TF_TensorType(outputTensors[i]));
        const auto doubleData = static_cast<float*>(rawData);

        if (outputs.size() != size*outputSize) {
            outputs.resize(size*outputSize);
        }
        for (size_t j=0; j<size; j++) {
            outputs[i*size + j] = doubleData[j];
        }
        TF_DeleteTensor(outputTensors[i]);
    }
}



