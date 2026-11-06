/*
By downloading, copying, installing or using the software you agree to this license.
If you do not agree to this license, do not download, install,
copy or use the software.


                  License Agreement For libfacedetection
                     (3-clause BSD License)

Copyright (c) 2018-2021, Shiqi Yu, all rights reserved.
shiqi.yu@gmail.com

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission. 

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are disclaimed.
In no event shall copyright holders or contributors be liable for any direct,
indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/


#include "facedetectcnn.h"
#include "pointwise_packed4x4.hpp"

#include <algorithm>
#include <cstdint>


#if 0
#include <opencv2/opencv.hpp>
cv::TickMeter cvtm;
#define TIME_START cvtm.reset();cvtm.start();
#define TIME_END(FUNCNAME) cvtm.stop(); printf(FUNCNAME);printf("=%g\n", cvtm.getTimeMilli());
#else
#define TIME_START
#define TIME_END(FUNCNAME)
#endif


extern ConvInfoStruct param_pConvInfo[kFaceConvLayers];

namespace {

size_t blob_floats(int columns, int channels)
{
    return static_cast<size_t>(columns) *
           CDataBlob<float>::padded_channel_step(channels) / sizeof(float);
}

#if defined(FACEDETECTION_ENABLE_POINTWISE_PACKED4X4)
int packed_channels(const ConvInfoStruct& layer)
{
    return layer.is_pointwise && !layer.is_depthwise
        ? (layer.num_filters / 4) * layer.channels * 4 : 0;
}
#endif

struct LegacyFaceFilters
{
    Filters<float> filters[kFaceConvLayers];

    LegacyFaceFilters()
    {
        for (int i = 0; i < kFaceConvLayers; ++i) {
            filters[i] = param_pConvInfo[i];
            filters[i].profile_index = i;
#if defined(FACEDETECTION_ENABLE_POINTWISE_PACKED4X4)
            prepare_pointwise_packed4x4(filters[i]);
#endif
        }
    }
};

std::vector<FaceRect> detect_with_filters(
    const unsigned char* rgbImageData, int width, int height, int step,
    const Filters<float>* filters)
{
    TIME_START;
    auto fx = setDataFrom3x3S2P1to1x1S1P0FromImage(rgbImageData, width, height, 3, step);
    TIME_END("convert data");

    /***************CONV0*********************/
    TIME_START;
    fx = convolution(fx, filters[0]);
    TIME_END("conv_head");

    TIME_START;
    fx = convolutionDP(fx, filters[1], filters[2]);
    TIME_END("conv0");

    TIME_START;
    fx = maxpooling2x2S2(fx);
    TIME_END("pool0");

    /***************CONV1*********************/
    TIME_START;
    fx = convolution4layerUnit(fx, filters[3], filters[4], filters[5], filters[6]);
    TIME_END("conv1");

    /***************CONV2*********************/
    TIME_START;
    fx = convolution4layerUnit(fx, filters[7], filters[8], filters[9], filters[10]);
    TIME_END("conv2");

    /***************CONV3*********************/
    TIME_START;
    fx = maxpooling2x2S2(fx);
    TIME_END("pool3");

    TIME_START;
    auto fb1 = convolution4layerUnit(fx, filters[11], filters[12], filters[13], filters[14]);
    TIME_END("conv3");

    /***************CONV4*********************/
    TIME_START;
    fx = maxpooling2x2S2(fb1);
    TIME_END("pool4");

    TIME_START;
    auto fb2 = convolution4layerUnit(fx, filters[15], filters[16], filters[17], filters[18]);
    TIME_END("conv4");

    /***************CONV5*********************/
    TIME_START;
    fx = maxpooling2x2S2(fb2);
    TIME_END("pool5");

    TIME_START;
    auto fb3 = convolution4layerUnit(fx, filters[19], filters[20], filters[21], filters[22]);
    TIME_END("conv5");

    CDataBlob<float> pred_reg[3], pred_cls[3], pred_kps[3], pred_obj[3];
    /***************branch5*********************/
    TIME_START;
    fb3 = convolutionDP(fb3, filters[27], filters[28]);
    pred_cls[2] = convolutionDP(fb3, filters[33], filters[34], false);
    pred_reg[2] = convolutionDP(fb3, filters[39], filters[40], false);
    pred_kps[2] = convolutionDP(fb3, filters[51], filters[52], false);
    pred_obj[2] = convolutionDP(fb3, filters[45], filters[46], false);
    TIME_END("branch5");

    /*****************add5*********************/    
    TIME_START;
    fb2 = elementAdd(upsampleX2(fb3), fb2);
    TIME_END("add5");

    /*****************add6*********************/    
    TIME_START;
    fb2 = convolutionDP(fb2, filters[25], filters[26]);
    pred_cls[1] = convolutionDP(fb2, filters[31], filters[32], false);
    pred_reg[1] = convolutionDP(fb2, filters[37], filters[38], false);
    pred_kps[1] = convolutionDP(fb2, filters[49], filters[50], false);
    pred_obj[1] = convolutionDP(fb2, filters[43], filters[44], false);
    TIME_END("branch4");

    /*****************add4*********************/
    TIME_START;
    fb1 = elementAdd(upsampleX2(fb2), fb1);
    TIME_END("add4");

    /***************branch3*********************/
    TIME_START;
    fb1 = convolutionDP(fb1, filters[23], filters[24]);
    pred_cls[0] = convolutionDP(fb1, filters[29], filters[30], false);
    pred_reg[0] = convolutionDP(fb1, filters[35], filters[36], false);
    pred_kps[0] = convolutionDP(fb1, filters[47], filters[48], false);
    pred_obj[0] = convolutionDP(fb1, filters[41], filters[42], false);
    TIME_END("branch3");
    
    /***************PRIORBOX*********************/
    TIME_START;
    auto prior3 = meshgrid(fb1.cols, fb1.rows, 8);
    auto prior4 = meshgrid(fb2.cols, fb2.rows, 16);
    auto prior5 = meshgrid(fb3.cols, fb3.rows, 32);
    TIME_END("prior");
    /***************PRIORBOX*********************/

    TIME_START;
    bbox_decode(pred_reg[0], prior3, 8);
    bbox_decode(pred_reg[1], prior4, 16);
    bbox_decode(pred_reg[2], prior5, 32);

    kps_decode(pred_kps[0], prior3, 8);
    kps_decode(pred_kps[1], prior4, 16);
    kps_decode(pred_kps[2], prior5, 32);

    auto cls = concat3(blob2vector(pred_cls[0]), blob2vector(pred_cls[1]), blob2vector(pred_cls[2]));
    auto reg = concat3(blob2vector(pred_reg[0]), blob2vector(pred_reg[1]), blob2vector(pred_reg[2]));
    auto kps = concat3(blob2vector(pred_kps[0]), blob2vector(pred_kps[1]), blob2vector(pred_kps[2]));
    auto obj = concat3(blob2vector(pred_obj[0]), blob2vector(pred_obj[1]), blob2vector(pred_obj[2]));

    sigmoid(cls);
    sigmoid(obj);
    TIME_END("decode")

    TIME_START;
    std::vector<FaceRect> facesInfo = detection_output(cls, reg, kps, obj, 0.45f, 0.2f, 1000, 512);
    TIME_END("detection output")
    return facesInfo;
}

}  // namespace

size_t PreparedFaceWeights::required_storage_floats()
{
    size_t count = 0;
    for (const auto& layer : param_pConvInfo) {
        count += blob_floats(layer.is_depthwise ? 9 : layer.num_filters, layer.channels);
        count += blob_floats(1, layer.num_filters);
#if defined(FACEDETECTION_ENABLE_POINTWISE_PACKED4X4)
        count += blob_floats(1, packed_channels(layer));
#endif
    }
    return count;
}

bool PreparedFaceWeights::Initialize(float* storage, size_t storage_floats)
{
    const auto required = required_storage_floats();
    if (ready_ || !storage ||
        reinterpret_cast<std::uintptr_t>(storage) % storage_alignment() != 0 ||
        storage_floats < required) {
        return false;
    }
    std::fill_n(storage, required, 0.0f);

    for (int i = 0; i < kFaceConvLayers; ++i) {
        const auto& layer = param_pConvInfo[i];
        auto& filter = filters_[i];
        filter.set_metadata(layer);
        filter.profile_index = i;

        const int columns = layer.is_depthwise ? 9 : layer.num_filters;
        filter.weights.bind(1, columns, layer.channels, storage);
        storage += blob_floats(columns, layer.channels);
        filter.biases.bind(1, 1, layer.num_filters, storage);
        storage += blob_floats(1, layer.num_filters);
        filter.copy_parameters(layer);

#if defined(FACEDETECTION_ENABLE_POINTWISE_PACKED4X4)
        const int packed = packed_channels(layer);
        if (packed != 0) {
            filter.packed_pointwise_weights.bind(1, 1, packed, storage);
            storage += blob_floats(1, packed);
            pack_pointwise_4x4_weights(filter);
        }
#endif
    }
    ready_ = true;
    return true;
}

std::vector<FaceRect> objectdetect_cnn(
    const unsigned char* data, int width, int height, int step)
{
    static const LegacyFaceFilters weights;
    return detect_with_filters(data, width, height, step, weights.filters);
}

std::vector<FaceRect> objectdetect_cnn(
    unsigned char* data, int width, int height, int step)
{
    return objectdetect_cnn(static_cast<const unsigned char*>(data), width, height, step);
}

std::vector<FaceRect> objectdetect_cnn(
    const unsigned char* data, int width, int height, int step,
    const PreparedFaceWeights& weights)
{
    if (!weights.ready()) {
        return {};
    }
    return detect_with_filters(data, width, height, step, weights.filters_);
}

int* facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be FACEDETECTION_RESULT_BUFFER_SIZE Bytes!!
    unsigned char * rgb_image_data, int width, int height, int step) //input image, it must be BGR (three-channel) image!
{

    if (!result_buffer)
    {
        fprintf(stderr, "%s: null buffer memory.\n", __FUNCTION__);
        return NULL;
    }
    //clear memory
    result_buffer[0] = 0;
    result_buffer[1] = 0;
    result_buffer[2] = 0;
    result_buffer[3] = 0;

    std::vector<FaceRect> faces = objectdetect_cnn(rgb_image_data, width, height, step);

    int num_faces =(int)faces.size();
    num_faces = MIN(num_faces, FACEDETECTION_RESULT_MAX_FACES);

    int * pCount = (int *)result_buffer;
    pCount[0] = num_faces;

    for (int i = 0; i < num_faces; i++)
    {
        //copy data
        short * p = ((short*)(result_buffer + 4)) + FACEDETECTION_RESULT_STRIDE_SHORTS * size_t(i);
        p[0] = (short)(faces[i].score * 100);
        p[1] = (short)faces[i].x;
        p[2] = (short)faces[i].y;
        p[3] = (short)faces[i].w;
        p[4] = (short)faces[i].h;
        //copy landmarks
        for (int lmidx = 0; lmidx < 10; lmidx++)
        {
            p[5 + lmidx] = (short)faces[i].lm[lmidx];
        }
    }

    return pCount;
}
