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


#ifndef BSSNNDEFORM_MODEL_H_
#define BSSNNDEFORM_MODEL_H_

#include <fstream>
#include <vector>

class Model
{
public:
    Model() {};
    virtual ~Model() {};

    virtual void inference(const std::vector<float>& inputs,
                           std::vector<float>& outputs,
                           const std::string& inputName,
                           const std::vector<std::string>& outputNames) = 0;
};

#endif /* BSSNNDEFORM_MODEL_H_ */
