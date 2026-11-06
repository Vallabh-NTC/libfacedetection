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

enum PredictionKind { kClass, kBox, kLandmarks, kObjectness, kPredictionKinds };
constexpr int kPredictionLayers[] = {29, 35, 47, 41};

struct Face192TensorLayout
{
    size_t first_slot;
    size_t scratch_slot;
    size_t feature_floats[3];
    size_t prediction_floats[3][kPredictionKinds];
};

Face192TensorLayout face192_tensor_layout()
{
    constexpr int stage_ends[] = {3, 11, 15, 19, 23};
    constexpr int head_side = Face192HeadWorkspace::input_size() / 2;
    Face192TensorLayout layout{};
    int side = head_side;
    int layer = 0;
    size_t retained = 0;
    int flattened_channels[kPredictionKinds]{};
    for (int stage = 0; stage < 5; ++stage) {
        const int pixels = side * side;
        for (; layer < stage_ends[stage]; ++layer) {
            layout.scratch_slot = std::max(layout.scratch_slot,
                blob_floats(pixels, param_pConvInfo[layer].num_filters));
        }
        if (stage >= 2) {
            const int level = stage - 2;
            auto& feature = layout.feature_floats[level];
            const int lateral = 23 + 2 * level;
            feature = blob_floats(pixels, std::max(param_pConvInfo[layer - 1].num_filters,
                param_pConvInfo[lateral + 1].num_filters));
            retained += feature;
            layout.scratch_slot = std::max(layout.scratch_slot, feature);
            layout.scratch_slot = std::max(layout.scratch_slot,
                blob_floats(pixels, param_pConvInfo[lateral].num_filters));
            layout.scratch_slot = std::max(layout.scratch_slot, blob_floats(pixels, 2));
            for (int kind = 0; kind < kPredictionKinds; ++kind) {
                const int pointwise = kPredictionLayers[kind] + 2 * level;
                const int channels = param_pConvInfo[pointwise + 1].num_filters;
                layout.scratch_slot = std::max(layout.scratch_slot,
                    blob_floats(pixels, param_pConvInfo[pointwise].num_filters));
                auto& prediction = layout.prediction_floats[level][kind];
                prediction = blob_floats(pixels, channels);
                retained += prediction;
                flattened_channels[kind] += pixels * channels;
            }
        }
        side /= 2;
    }
    size_t flattened = 0;
    for (int channels : flattened_channels) {
        flattened += blob_floats(1, channels);
    }
    layout.scratch_slot = std::max(layout.scratch_slot, flattened);
    const auto input = blob_floats(head_side * head_side, param_pConvInfo[0].channels);
    // The input tensor dies at the head convolution. Its slot can then hold
    // one scratch tensor, the three retained feature maps and the predictions.
    layout.first_slot = std::max(input, layout.scratch_slot + retained);
    return layout;
}

struct InferenceBuffers
{
    float* scratch0;
    float* scratch1;
    float* fb1;
    float* fb2;
    float* fb3;
    float* (*predictions)[kPredictionKinds];
};

CDataBlob<float> depthwise_pair(const CDataBlob<float>& input,
    const Filters<float>* filters, bool relu, float* intermediate, float* output)
{
    return convolution(convolution(input, filters[0], false, intermediate),
                       filters[1], relu, output);
}

CDataBlob<float> backbone_block(const CDataBlob<float>& input,
    const Filters<float>* filters, float* first, float* second, float* output)
{
    auto intermediate = convolution(input, filters[0], false, first);
    intermediate = convolution(intermediate, filters[1], true, second);
    intermediate = convolution(intermediate, filters[2], false, first);
    return convolution(intermediate, filters[3], true, output);
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
    const Filters<float>* filters, const InferenceBuffers& buffers = {})
{
    TIME_START;
    auto fx = setDataFrom3x3S2P1to1x1S1P0FromImage(
        rgbImageData, width, height, 3, step, 32, buffers.scratch0);
    TIME_END("convert data");

    /***************CONV0*********************/
    TIME_START;
    fx = convolution(fx, filters[0], true, buffers.scratch1);
    TIME_END("conv_head");

    TIME_START;
    fx = depthwise_pair(fx, filters + 1, true, buffers.scratch0, buffers.scratch1);
    TIME_END("conv0");

    TIME_START;
    fx = maxpooling2x2S2(fx, buffers.scratch0);
    TIME_END("pool0");

    /***************CONV1*********************/
    TIME_START;
    fx = backbone_block(fx, filters + 3, buffers.scratch1, buffers.scratch0, buffers.scratch0);
    TIME_END("conv1");

    /***************CONV2*********************/
    TIME_START;
    fx = backbone_block(fx, filters + 7, buffers.scratch1, buffers.scratch0, buffers.scratch0);
    TIME_END("conv2");

    /***************CONV3*********************/
    TIME_START;
    fx = maxpooling2x2S2(fx, buffers.scratch1);
    TIME_END("pool3");

    TIME_START;
    auto fb1 = backbone_block(fx, filters + 11, buffers.scratch0, buffers.scratch1, buffers.fb1);
    TIME_END("conv3");

    /***************CONV4*********************/
    TIME_START;
    fx = maxpooling2x2S2(fb1, buffers.scratch1);
    TIME_END("pool4");

    TIME_START;
    auto fb2 = backbone_block(fx, filters + 15, buffers.scratch0, buffers.scratch1, buffers.fb2);
    TIME_END("conv4");

    /***************CONV5*********************/
    TIME_START;
    fx = maxpooling2x2S2(fb2, buffers.scratch1);
    TIME_END("pool5");

    TIME_START;
    auto fb3 = backbone_block(fx, filters + 19, buffers.scratch0, buffers.scratch1, buffers.fb3);
    TIME_END("conv5");

    CDataBlob<float> predictions[kPredictionKinds][3];
    const auto predict = [&](const CDataBlob<float>& feature, int level) {
        for (int kind = 0; kind < kPredictionKinds; ++kind) {
            const int layer = kPredictionLayers[kind] + 2 * level;
            float* output = buffers.predictions ? buffers.predictions[level][kind] : nullptr;
            predictions[kind][level] = depthwise_pair(
                feature, filters + layer, false, buffers.scratch0, output);
        }
    };
    /***************branch5*********************/
    TIME_START;
    fb3 = depthwise_pair(fb3, filters + 27, true, buffers.scratch0, buffers.fb3);
    predict(fb3, 2);
    TIME_END("branch5");

    /*****************add5*********************/    
    TIME_START;
    fb2 = elementAdd(upsampleX2(fb3, buffers.scratch0), fb2, buffers.scratch1);
    TIME_END("add5");

    /*****************add6*********************/    
    TIME_START;
    fb2 = depthwise_pair(fb2, filters + 25, true, buffers.scratch0, buffers.fb2);
    predict(fb2, 1);
    TIME_END("branch4");

    /*****************add4*********************/
    TIME_START;
    fb1 = elementAdd(upsampleX2(fb2, buffers.scratch0), fb1, buffers.scratch1);
    TIME_END("add4");

    /***************branch3*********************/
    TIME_START;
    fb1 = depthwise_pair(fb1, filters + 23, true, buffers.scratch0, buffers.fb1);
    predict(fb1, 0);
    TIME_END("branch3");
    
    /***************PRIORBOX*********************/
    TIME_START;
    for (int level = 0; level < 3; ++level) {
        const int stride = 8 << level;
        auto& boxes = predictions[kBox][level];
        const auto priors = meshgrid(boxes.cols, boxes.rows, stride, 0.0f, buffers.scratch0);
        bbox_decode(boxes, priors, stride);
        kps_decode(predictions[kLandmarks][level], priors, stride);
    }
    float* next = buffers.scratch1;
    const auto flatten = [&](PredictionKind kind) {
        auto output = flatten_predictions(predictions[kind], next);
        if (next) {
            next += blob_floats(1, output.channels);
        }
        return output;
    };
    auto cls = flatten(kClass);
    auto reg = flatten(kBox);
    auto kps = flatten(kLandmarks);
    auto obj = flatten(kObjectness);

    sigmoid(cls);
    sigmoid(obj);
    TIME_END("decode")

    TIME_START;
    std::vector<FaceRect> facesInfo = detection_output(cls, reg, kps, obj, 0.45f, 0.2f, 1000, 512);
    TIME_END("detection output")
    return facesInfo;
}

}  // namespace

size_t Face192HeadWorkspace::required_storage_floats()
{
    const auto layout = face192_tensor_layout();
    return layout.first_slot + layout.scratch_slot;
}

bool Face192HeadWorkspace::Initialize(float* storage, size_t storage_floats)
{
    const auto layout = face192_tensor_layout();
    if (ready() || !storage ||
        reinterpret_cast<std::uintptr_t>(storage) % storage_alignment() != 0 ||
        storage_floats < layout.first_slot + layout.scratch_slot) {
        return false;
    }
    scratch_[0] = storage;
    scratch_[1] = storage + layout.first_slot;
    features_[0] = storage + layout.scratch_slot;
    features_[1] = features_[0] + layout.feature_floats[0];
    features_[2] = features_[1] + layout.feature_floats[1];
    float* next = features_[2] + layout.feature_floats[2];
    for (int level = 0; level < 3; ++level) {
        for (int kind = 0; kind < kPredictionKinds; ++kind) {
            predictions_[level][kind] = next;
            next += layout.prediction_floats[level][kind];
        }
    }
    return true;
}

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

std::vector<FaceRect> objectdetect_cnn_face192(
    const unsigned char* data, int step, const PreparedFaceWeights& weights,
    Face192HeadWorkspace& workspace)
{
    constexpr int size = Face192HeadWorkspace::input_size();
    if (!data || step < size * 3 || !weights.ready() || !workspace.ready()) {
        return {};
    }
    return detect_with_filters(data, size, size, step, weights.filters_,
        {workspace.scratch_[0], workspace.scratch_[1],
         workspace.features_[0], workspace.features_[1], workspace.features_[2],
         workspace.predictions_});
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
