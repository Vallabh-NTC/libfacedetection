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

#pragma once

#include "facedetection_export.h"

//#define _ENABLE_AVX512 //Please enable it if X64 CPU
//#define _ENABLE_AVX2 //Please enable it if X64 CPU
//#define _ENABLE_NEON //Please enable it if ARM CPU

#define FACEDETECTION_RESULT_BUFFER_SIZE 0x9000
#define FACEDETECTION_RESULT_MAX_FACES 1024
#define FACEDETECTION_RESULT_STRIDE_SHORTS 16

FACEDETECTION_EXPORT int * facedetect_cnn(unsigned char * result_buffer, //buffer memory for storing face detection results, !!its size must be FACEDETECTION_RESULT_BUFFER_SIZE Bytes!!
                    unsigned char * rgb_image_data, int width, int height, int step); //input image, it must be BGR (three channels) insteed of RGB image!

/*
DO NOT EDIT the following code if you don't really understand it.
*/
#if defined(_ENABLE_AVX512) || defined(_ENABLE_AVX2)
#include <immintrin.h>
#endif


#if defined(_ENABLE_NEON)
#include "arm_neon.h"
//NEON does not support UINT8*INT8 dot product
//to conver the input data to range [0, 127],
//and then use INT8*INT8 dot product
#define _MAX_UINT8_VALUE 127
#else
#define _MAX_UINT8_VALUE 255
#endif

#if defined(_ENABLE_AVX512) 
#define _MALLOC_ALIGN 512
#elif defined(_ENABLE_AVX2) 
#define _MALLOC_ALIGN 256
#else
#define _MALLOC_ALIGN 128
#endif

#if defined(_ENABLE_AVX512)&& defined(_ENABLE_NEON)
#error Cannot enable the two of AVX512 and NEON at the same time.
#endif
#if defined(_ENABLE_AVX2)&& defined(_ENABLE_NEON)
#error Cannot enable the two of AVX and NEON at the same time.
#endif
#if defined(_ENABLE_AVX512)&& defined(_ENABLE_AVX2)
#error Cannot enable the two of AVX512 and AVX2 at the same time.
#endif


#if defined(_OPENMP)
#include <omp.h>
#endif

#include <string.h>
#include <vector>
#include <iostream>
#include <typeinfo>

void* myAlloc(size_t size);
void myFree_(void* ptr);
#define myFree(ptr) (myFree_(*(ptr)), *(ptr)=0);

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

typedef struct FaceRect_
{
    float score;
    int x;
    int y;
    int w;
    int h;
    int lm[10];
}FaceRect;

typedef struct ConvInfoStruct_ {
    int channels;
    int num_filters;
    bool is_depthwise;
    bool is_pointwise;
    bool with_relu;
    float* pWeights;
    float* pBiases;
}ConvInfoStruct;



template <typename T>
class CDataBlob
{
    bool owns_data_ = true;

public:
	int rows;
	int cols;
	int channels; //in element
    int channelStep; //in byte
    T * data;

public:
	CDataBlob() {
		rows = 0;
		cols = 0;
        channels = 0;
        channelStep = 0;
        data = nullptr;
	}
	CDataBlob(int r, int c, int ch, T* storage = nullptr)
	{
        data = nullptr;
        if (storage)
            bind(r, c, ch, storage);
        else
            create(r, c, ch);
        setZero();
	}
	~CDataBlob()
	{
        setNULL();
	}

    CDataBlob(CDataBlob<T> &&other) {
        owns_data_ = other.owns_data_;
        data = other.data;
        other.data = nullptr;
        rows = other.rows;
        cols = other.cols;
        channels = other.channels;
        channelStep = other.channelStep;
    }

    CDataBlob<T> &operator=(CDataBlob<T> &&other) {
        this->~CDataBlob();
        new (this) CDataBlob<T>(std::move(other));
        return *this;
    }

    void setNULL()
    {
        if (data && owns_data_)
            myFree(&data);
        owns_data_ = true;
        rows = cols = channels = channelStep = 0;
        data = nullptr;
    }

    void setZero()
    {
        if(data)
            memset(data, 0, channelStep * rows * cols);
    }

    inline bool isEmpty() const
    {
        return (rows <= 0 || cols <= 0 || channels == 0 || data == nullptr);
    }

    static int padded_channel_step(int ch)
    {
        const int bytes = ch * sizeof(T);
        const int alignment = _MALLOC_ALIGN / 8;
        return (bytes + alignment - 1) / alignment * alignment;
    }

    // Caller supplies enough aligned storage for the padded layout.
    void bind(int r, int c, int ch, T* storage)
    {
        setNULL();
        rows = r;
        cols = c;
        channels = ch;
        channelStep = padded_channel_step(ch);
        data = storage;
        owns_data_ = false;
    }

	bool create(int r, int c, int ch)
	{
        setNULL();

		rows = r;
		cols = c;
        channels = ch;

        //alloc space for int8 array
        this->channelStep = padded_channel_step(channels);
        data = (T*)myAlloc(size_t(rows) * cols * this->channelStep);

        if (data == nullptr)
        {
            std::cerr << "Failed to alloc memeory for uint8 data blob: "
                << rows << "*"
                << cols << "*"
                << channels << std::endl;
            return false;
        }
        
        //memset(data, 0, width * height * channelStep);
        
        //the following code is faster than memset
        //but not only the padding bytes are set to zero.
        //BE CAREFUL!!!
//#if defined(_OPENMP)
//#pragma omp parallel for
//#endif
        // for (int r = 0; r < this->rows; r++)
        // {
        //     for (int c = 0; c < this->cols; c++)
        //     {
        //         int pixel_end = this->channelStep / sizeof(T);
        //         T * pI = this->ptr(r, c);
        //         for (int ch = this->channels; ch < pixel_end; ch++)
        //             pI[ch] = 0;
        //     }
        // }
        
        return true;
	}

    inline T * ptr(int r, int c)
    {
        if( r < 0 || r >= this->rows || c < 0 || c >= this->cols )
            return nullptr;

        return (this->data + (size_t(r) * this->cols + c) * this->channelStep /sizeof(T));
    }
    inline const T * ptr(int r, int c) const
    {
        if( r < 0 || r >= this->rows || c < 0 || c >= this->cols )
            return nullptr;

        return (this->data + (size_t(r) * this->cols + c) * this->channelStep /sizeof(T));
    }

    inline const T getElement(int r, int c, int ch) const
    {
        if (this->data)
        {
            if (r >= 0 && r < this->rows &&
                c >= 0 && c < this->cols &&
                ch >= 0 && ch < this->channels)
            {
                const T * p = this->ptr(r, c);
                return (p[ch]);
            }
        }
        
        return (T)(0);
    }

    friend std::ostream &operator<<(std::ostream &output, CDataBlob &dataBlob)
    {
        output << "DataBlob Size (channels, rows, cols) = ("
            << dataBlob.channels
            << ", " << dataBlob.rows
            << ", " << dataBlob.cols
            << ")" << std::endl;
        if( dataBlob.rows * dataBlob.cols * dataBlob.channels <= 16)
        { //print the elements only when the total number is less than 64
            for (int ch = 0; ch < dataBlob.channels; ch++)
            {
                output << "Channel " << ch << ": " << std::endl;

                for (int r = 0; r < dataBlob.rows; r++)
                {
                    output << "(";
                    for (int c = 0; c < dataBlob.cols; c++)
                    {
                        T * p = dataBlob.ptr(r, c);

                        if(sizeof(T)<4)
                            output << (int)(p[ch]);
                        else
                            output << p[ch];

                        if (c != dataBlob.cols - 1)
                            output << ", ";
                    }
                    output << ")" << std::endl;
                }
            }
        }
        else{
            output << "(" ;
            int idx = 0;
            bool outloop = false;
            for(int r = 0; r < dataBlob.rows && !outloop; ++r) {
                for(int c = 0; c < dataBlob.cols && !outloop; ++c) {
                    for(int ch = 0; ch < dataBlob.channels && !outloop; ++ch) {
                       output << dataBlob.getElement(r, c, ch) << ", ";
                       ++idx;
                       if(idx >= 16) {
                            outloop = true;
                       }
                    }
                }
            } 
            output << "..., " 
                    << dataBlob.getElement(dataBlob.rows-1, dataBlob.cols-1, dataBlob.channels-1) << ")"
                    << std::endl; 
            float max_it = -500.f;
            float min_it = 500.f;
            for(int r = 0; r < dataBlob.rows; ++r) {
                for(int c = 0; c < dataBlob.cols; ++c) {
                    for(int ch = 0; ch < dataBlob.channels; ++ch) {
                        max_it = std::max(max_it, dataBlob.getElement(r, c, ch));
                        min_it = std::min(min_it, dataBlob.getElement(r, c, ch));
                    }
                }
            }   
            output << "max_it: " << max_it << "    min_it: " << min_it << std::endl;        
        }
        return output;
    }
};

template <typename T>
class Filters{
  public:
    int channels;
    int num_filters;
    bool is_depthwise;
    bool is_pointwise;
    bool with_relu;
    int profile_index;
    CDataBlob<T> weights;
    CDataBlob<T> biases;
#if defined(FACEDETECTION_ENABLE_POINTWISE_PACKED4X4)
    CDataBlob<T> packed_pointwise_weights;
#endif

    Filters()
    {
        channels = 0;
        num_filters = 0;
        is_depthwise = false;
        is_pointwise = false;
        with_relu = true;
        profile_index = -1;
    }

    void set_metadata(const ConvInfoStruct& convinfo)
    {
        channels = convinfo.channels;
        num_filters = convinfo.num_filters;
        is_depthwise = convinfo.is_depthwise;
        is_pointwise = convinfo.is_pointwise;
        with_relu = convinfo.with_relu;
    }

    void copy_parameters(const ConvInfoStruct& convinfo)
    {
        for (int fidx = 0; fidx < weights.cols; ++fidx)
            memcpy(weights.ptr(0, fidx), convinfo.pWeights + channels * fidx,
                   channels * sizeof(T));
        memcpy(biases.ptr(0, 0), convinfo.pBiases, num_filters * sizeof(T));
    }

    Filters & operator=(ConvInfoStruct & convinfo)
    {
        if (typeid(float) != typeid(T))
        {
            std::cerr << "The data type must be float in this version." << std::endl;
            return *this;
        }
        if (typeid(float*) != typeid(convinfo.pWeights) ||
            typeid(float*) != typeid(convinfo.pBiases))
        {
            std::cerr << "The data type of the filter parameters must be float in this version." << std::endl;
            return *this;
        }

        set_metadata(convinfo);

        if(!this->is_depthwise && this->is_pointwise) //1x1 point wise
        {
            this->weights.create(1, num_filters, channels);
        }
        else if(this->is_depthwise && !this->is_pointwise) //3x3 depth wise
        {
            this->weights.create(1, 9, channels);
        }
        else 
        {
            std::cerr << "Unsupported filter type. Only 1x1 point-wise and 3x3 depth-wise are supported." << std::endl;
            return *this;
        }

        this->biases.create(1, 1, num_filters);

        copy_parameters(convinfo);

        return *this;
    }

};

constexpr int kFaceConvLayers = 53;

class PreparedFaceWeights;

class Face192HeadWorkspace
{
public:
    static constexpr int input_size() { return 192; }
    static constexpr size_t storage_alignment() { return _MALLOC_ALIGN / 8; }
    static size_t required_storage_floats();

    // Bind the inference tensor buffers once. Storage must outlive this workspace and
    // remain exclusive to its worker.
    bool Initialize(float* storage, size_t storage_floats);
    bool ready() const { return scratch_[0] != nullptr; }

    Face192HeadWorkspace() = default;
    Face192HeadWorkspace(const Face192HeadWorkspace&) = delete;
    Face192HeadWorkspace& operator=(const Face192HeadWorkspace&) = delete;

private:
    friend std::vector<FaceRect> objectdetect_cnn_face192(
        const unsigned char*, int, const PreparedFaceWeights&, Face192HeadWorkspace&);

    float* scratch_[2]{};
    float* features_[3]{};
    float* predictions_[3][4]{};
};

class PreparedFaceWeights
{
public:
    PreparedFaceWeights() = default;
    PreparedFaceWeights(const PreparedFaceWeights&) = delete;
    PreparedFaceWeights& operator=(const PreparedFaceWeights&) = delete;

    static constexpr size_t storage_alignment() { return _MALLOC_ALIGN / 8; }
    static size_t required_storage_floats();

    // Initialize once. Keep the aligned float storage alive and immutable
    // after initialization, for as long as this context or its users exist.
    bool Initialize(float* storage, size_t storage_floats);
    bool ready() const { return ready_; }

private:
    friend std::vector<FaceRect> objectdetect_cnn(
        const unsigned char*, int, int, int, const PreparedFaceWeights&);
    friend std::vector<FaceRect> objectdetect_cnn_face192(
        const unsigned char*, int, const PreparedFaceWeights&, Face192HeadWorkspace&);

    Filters<float> filters_[kFaceConvLayers];
    bool ready_ = false;
};

std::vector<FaceRect> objectdetect_cnn(const unsigned char* rgbImageData, int with, int height, int step);
std::vector<FaceRect> objectdetect_cnn(const unsigned char* rgbImageData,
    int width, int height, int step, const PreparedFaceWeights& weights);
std::vector<FaceRect> objectdetect_cnn_face192(const unsigned char* rgbImageData,
    int step, const PreparedFaceWeights& weights, Face192HeadWorkspace& workspace);

CDataBlob<float> setDataFrom3x3S2P1to1x1S1P0FromImage(const unsigned char* inputData, int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int padDivisor=32);
CDataBlob<float> setDataFrom3x3S2P1to1x1S1P0FromImage(const unsigned char* inputData,
    int imgWidth, int imgHeight, int imgChannels, int imgWidthStep, int padDivisor, float* output_storage);
CDataBlob<float> convolution(const CDataBlob<float>& inputData, const Filters<float>& filters, bool do_relu = true);
CDataBlob<float> convolution(const CDataBlob<float>& inputData,
    const Filters<float>& filters, bool do_relu, float* output_storage);
CDataBlob<float> convolutionDP(const CDataBlob<float>& inputData, 
                const Filters<float>& filtersP, const Filters<float>& filtersD, bool do_relu = true);
CDataBlob<float> convolution4layerUnit(const CDataBlob<float>& inputData, 
                const Filters<float>& filtersP1, const Filters<float>& filtersD1, 
                const Filters<float>& filtersP2, const Filters<float>& filtersD2, bool do_relu = true);
CDataBlob<float> maxpooling2x2S2(const CDataBlob<float>& inputData);
CDataBlob<float> maxpooling2x2S2(const CDataBlob<float>& inputData, float* output_storage);

CDataBlob<float> elementAdd(const CDataBlob<float>& inputData1, const CDataBlob<float>& inputData2);
CDataBlob<float> elementAdd(const CDataBlob<float>& inputData1,
    const CDataBlob<float>& inputData2, float* output_storage);
CDataBlob<float> upsampleX2(const CDataBlob<float>& inputData);
CDataBlob<float> upsampleX2(const CDataBlob<float>& inputData, float* output_storage);

CDataBlob<float> meshgrid(int feature_width, int feature_height, int stride, float offset=0.0f);
CDataBlob<float> meshgrid(int feature_width, int feature_height, int stride,
    float offset, float* output_storage);

// TODO implement in SIMD
void bbox_decode(CDataBlob<float>& bbox_pred, const CDataBlob<float>& priors, int stride);
void kps_decode(CDataBlob<float>& bbox_pred, const CDataBlob<float>& priors, int stride);

template<typename T>
CDataBlob<T> blob2vector(const CDataBlob<T> &inputData);

template<typename T>
CDataBlob<T> concat3(const CDataBlob<T>& inputData1, const CDataBlob<T>& inputData2, const CDataBlob<T>& inputData3);

CDataBlob<float> flatten_predictions(const CDataBlob<float> (&predictions)[3],
    float* output_storage);

// TODO implement in SIMD
void sigmoid(CDataBlob<float>& inputData);

std::vector<FaceRect> detection_output(const CDataBlob<float>& cls,
                const CDataBlob<float>& reg,
                const CDataBlob<float>& kps,
                const CDataBlob<float>& obj,
                float overlap_threshold, float confidence_threshold, int top_k, int keep_top_k);
