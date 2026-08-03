#include "pointwise_packed4x4.hpp"

#include <cstddef>

#if defined(_ENABLE_NEON)
#include <arm_neon.h>
#endif

#if defined(FACEDETECTION_ENABLE_POINTWISE_PACKED4X4)

bool convolution_1x1pointwise(
    const CDataBlob<float>& inputData,
    const Filters<float>& filters,
    CDataBlob<float>& outputData);

namespace {

inline float scalar_dot_product(
    const float* input,
    const float* weights,
    int channels)
{
    float sum = 0.0f;
    for (int channel = 0; channel < channels; ++channel)
        sum += input[channel] * weights[channel];
    return sum;
}

}  // namespace

void prepare_pointwise_packed4x4(Filters<float>& filters)
{
    if (filters.is_depthwise || !filters.is_pointwise) {
        filters.packed_pointwise_weights.setNULL();
        return;
    }

    const int output_blocks = filters.num_filters / 4;
    const std::size_t packed_count =
        static_cast<std::size_t>(output_blocks) *
        static_cast<std::size_t>(filters.channels) * 4U;

    if (packed_count == 0) {
        filters.packed_pointwise_weights.setNULL();
        return;
    }

    if (!filters.packed_pointwise_weights.create(
            1, 1, static_cast<int>(packed_count))) {
        return;
    }

    float* packed = filters.packed_pointwise_weights.data;

    for (int block = 0; block < output_blocks; ++block) {
        const int output_channel = block * 4;
        const float* weights0 = filters.weights.ptr(0, output_channel + 0);
        const float* weights1 = filters.weights.ptr(0, output_channel + 1);
        const float* weights2 = filters.weights.ptr(0, output_channel + 2);
        const float* weights3 = filters.weights.ptr(0, output_channel + 3);

        float* packed_block =
            packed + static_cast<std::size_t>(block) *
                         static_cast<std::size_t>(filters.channels) * 4U;

        for (int input_channel = 0;
             input_channel < filters.channels;
             ++input_channel) {
            float* packed_vector =
                packed_block +
                static_cast<std::size_t>(input_channel) * 4U;
            packed_vector[0] = weights0[input_channel];
            packed_vector[1] = weights1[input_channel];
            packed_vector[2] = weights2[input_channel];
            packed_vector[3] = weights3[input_channel];
        }
    }
}

bool convolution_1x1pointwise_packed4x4(
    const CDataBlob<float>& inputData,
    const Filters<float>& filters,
    CDataBlob<float>& outputData)
{
#if !defined(_ENABLE_NEON)
    return convolution_1x1pointwise(inputData, filters, outputData);
#else
    if (filters.packed_pointwise_weights.isEmpty())
        return convolution_1x1pointwise(inputData, filters, outputData);

    const float* packed_weights = filters.packed_pointwise_weights.data;
    const int output_blocks = outputData.channels / 4;
    const int column_blocks = outputData.cols / 4;

#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int row = 0; row < outputData.rows; ++row) {
        for (int column_block = 0;
             column_block < column_blocks;
             ++column_block) {
            const int col = column_block * 4;
            const float* input0 = inputData.ptr(row, col + 0);
            const float* input1 = inputData.ptr(row, col + 1);
            const float* input2 = inputData.ptr(row, col + 2);
            const float* input3 = inputData.ptr(row, col + 3);
            float* output0 = outputData.ptr(row, col + 0);
            float* output1 = outputData.ptr(row, col + 1);
            float* output2 = outputData.ptr(row, col + 2);
            float* output3 = outputData.ptr(row, col + 3);

            for (int block = 0; block < output_blocks; ++block) {
                const int output_channel = block * 4;
                const float* packed_block =
                    packed_weights +
                    static_cast<std::size_t>(block) *
                        static_cast<std::size_t>(inputData.channels) * 4U;
                const float32x4_t bias =
                    vld1q_f32(filters.biases.data + output_channel);
                float32x4_t accumulator0 = bias;
                float32x4_t accumulator1 = bias;
                float32x4_t accumulator2 = bias;
                float32x4_t accumulator3 = bias;

                for (int input_channel = 0;
                     input_channel < inputData.channels;
                     ++input_channel) {
                    const float32x4_t weights = vld1q_f32(
                        packed_block +
                        static_cast<std::size_t>(input_channel) * 4U);
                    accumulator0 = vmlaq_n_f32(
                        accumulator0, weights, input0[input_channel]);
                    accumulator1 = vmlaq_n_f32(
                        accumulator1, weights, input1[input_channel]);
                    accumulator2 = vmlaq_n_f32(
                        accumulator2, weights, input2[input_channel]);
                    accumulator3 = vmlaq_n_f32(
                        accumulator3, weights, input3[input_channel]);
                }

                vst1q_f32(output0 + output_channel, accumulator0);
                vst1q_f32(output1 + output_channel, accumulator1);
                vst1q_f32(output2 + output_channel, accumulator2);
                vst1q_f32(output3 + output_channel, accumulator3);
            }

            for (int output_channel = output_blocks * 4;
                 output_channel < outputData.channels;
                 ++output_channel) {
                const float* weights =
                    filters.weights.ptr(0, output_channel);
                const float bias = filters.biases.data[output_channel];
                output0[output_channel] =
                    scalar_dot_product(input0, weights, inputData.channels) + bias;
                output1[output_channel] =
                    scalar_dot_product(input1, weights, inputData.channels) + bias;
                output2[output_channel] =
                    scalar_dot_product(input2, weights, inputData.channels) + bias;
                output3[output_channel] =
                    scalar_dot_product(input3, weights, inputData.channels) + bias;
            }
        }

        for (int col = column_blocks * 4; col < outputData.cols; ++col) {
            const float* input = inputData.ptr(row, col);
            float* output = outputData.ptr(row, col);

            for (int block = 0; block < output_blocks; ++block) {
                const int output_channel = block * 4;
                const float* packed_block =
                    packed_weights +
                    static_cast<std::size_t>(block) *
                        static_cast<std::size_t>(inputData.channels) * 4U;
                float32x4_t accumulator =
                    vld1q_f32(filters.biases.data + output_channel);

                for (int input_channel = 0;
                     input_channel < inputData.channels;
                     ++input_channel) {
                    const float32x4_t weights = vld1q_f32(
                        packed_block +
                        static_cast<std::size_t>(input_channel) * 4U);
                    accumulator = vmlaq_n_f32(
                        accumulator, weights, input[input_channel]);
                }
                vst1q_f32(output + output_channel, accumulator);
            }

            for (int output_channel = output_blocks * 4;
                 output_channel < outputData.channels;
                 ++output_channel) {
                const float* weights =
                    filters.weights.ptr(0, output_channel);
                output[output_channel] =
                    scalar_dot_product(input, weights, inputData.channels) +
                    filters.biases.data[output_channel];
            }
        }
    }

    return true;
#endif
}

#endif
