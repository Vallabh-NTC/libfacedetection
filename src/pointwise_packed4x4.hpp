#pragma once

#include "facedetectcnn.h"

#if defined(FACEDETECTION_ENABLE_POINTWISE_PACKED4X4)

void prepare_pointwise_packed4x4(Filters<float>& filters);
void pack_pointwise_4x4_weights(Filters<float>& filters);

bool convolution_1x1pointwise_packed4x4(
    const CDataBlob<float>& inputData,
    const Filters<float>& filters,
    CDataBlob<float>& outputData);

#endif
