/**
 * YUV Preprocessing JNI Bridge
 *
 * High-performance YUV_420_888 → RGB24 conversion with:
 * - Stride normalization (handles all YUV_420_888 variants)
 * - YUV-plane rotation (0°/90°/180°/270°) via libyuv NEON
 * - Optional horizontal mirror (front camera)
 * - Letterbox scaling and padding into pre-allocated tensor buffer
 *
 * All operations use libyuv's NEON-optimized routines on ARM.
 */

#include <jni.h>
#include <android/log.h>
#include <libyuv.h>
#include <cstring>
#include <algorithm>
#include <mutex>

#define LOG_TAG "YuvPreprocessor"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN,  LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static constexpr int TARGET_SIZE = 640;
static constexpr int RGB_CHANNELS = 3;
static constexpr int TENSOR_BUF_SIZE = TARGET_SIZE * TARGET_SIZE * RGB_CHANNELS;

class YuvProcessor {
public:
    YuvProcessor() = default;

    ~YuvProcessor() {
        release();
    }

    /**
     * Ensure internal scratch buffers are large enough for the given source dimensions.
     * Buffers are allocated once and reused across frames (zero per-frame allocation).
     */
    void ensureBuffers(int srcW, int srcH) {
        int srcI420Size = srcW * srcH * 3 / 2;
        int maxDim = std::max(srcW, srcH);
        int rotatedI420Size = maxDim * maxDim * 3 / 2;

        std::lock_guard<std::mutex> lock(bufMutex_);
        if (!i420Src_ || srcI420Size > i420SrcSize_) {
            free(i420Src_);
            i420Src_ = static_cast<uint8_t*>(malloc(srcI420Size));
            i420SrcSize_ = srcI420Size;
            LOGI("Allocated i420_src buffer: %d bytes", srcI420Size);
        }
        if (!i420Rot_ || rotatedI420Size > i420RotSize_) {
            free(i420Rot_);
            i420Rot_ = static_cast<uint8_t*>(malloc(rotatedI420Size));
            i420RotSize_ = rotatedI420Size;
            LOGI("Allocated i420_rot buffer: %d bytes", rotatedI420Size);
        }
    }

    /**
     * Main processing pipeline:
     * YUV_420_888 → I420 normalize → rotate → mirror → scale → RGB24 letterbox
     *
     * @return true on success, false on error
     */
    bool process(
        const uint8_t* yPlane, const uint8_t* uPlane, const uint8_t* vPlane,
        int yRowStride, int uvRowStride, int uvPixelStride,
        int srcW, int srcH,
        int rotation, bool needMirror,
        uint8_t* outputBuffer, int targetSize)
    {
        // ---- Step 1: Normalize YUV_420_888 to standard I420 ----
        uint8_t* srcY = i420Src_;
        uint8_t* srcU = srcY + srcW * srcH;
        uint8_t* srcV = srcU + (srcW / 2) * (srcH / 2);

        int ret;
        if (uvPixelStride == 2) {
            // Semi-planar (NV12/NV21): U and V interleaved in the same buffer
            ret = libyuv::Android420ToI420(
                yPlane, yRowStride,
                uPlane, uvRowStride,
                vPlane, uvRowStride,
                uvPixelStride,
                srcY, srcW,
                srcU, srcW / 2,
                srcV, srcW / 2,
                srcW, srcH);
        } else if (uvPixelStride == 1) {
            // Fully planar: U and V in separate buffers
            ret = libyuv::I420Copy(
                yPlane, yRowStride,
                uPlane, uvRowStride,
                vPlane, uvRowStride,
                srcY, srcW,
                srcU, srcW / 2,
                srcV, srcW / 2,
                srcW, srcH);
        } else {
            LOGE("Unsupported uv_pixel_stride: %d", uvPixelStride);
            return false;
        }

        if (ret != 0) {
            LOGE("YUV to I420 conversion failed: %d", ret);
            return false;
        }

        // ---- Step 2: Rotate YUV planes ----
        libyuv::RotationMode rotMode;
        switch (rotation) {
            case 90:  rotMode = libyuv::kRotate90;  break;
            case 180: rotMode = libyuv::kRotate180; break;
            case 270: rotMode = libyuv::kRotate270; break;
            default:  rotMode = libyuv::kRotate0;   break;
        }

        int rotW = srcW, rotH = srcH;
        const uint8_t* curY = srcY;
        const uint8_t* curU = srcU;
        const uint8_t* curV = srcV;
        int curStrideY = srcW;
        int curStrideUV = srcW / 2;

        if (rotMode != libyuv::kRotate0) {
            uint8_t* dstY = i420Rot_;
            uint8_t* dstU, *dstV;

            if (rotMode == libyuv::kRotate90 || rotMode == libyuv::kRotate270) {
                rotW = srcH;
                rotH = srcW;
            }
            // For 180°, dimensions don't change

            dstU = dstY + rotW * rotH;
            dstV = dstU + (rotW / 2) * (rotH / 2);

            ret = libyuv::I420Rotate(
                srcY, srcW,
                srcU, srcW / 2,
                srcV, srcW / 2,
                dstY, rotW,
                dstU, rotW / 2,
                dstV, rotW / 2,
                srcW, srcH,
                rotMode);

            if (ret != 0) {
                LOGE("I420Rotate failed: %d", ret);
                return false;
            }

            curY = dstY;
            curU = dstU;
            curV = dstV;
            curStrideY = rotW;
            curStrideUV = rotW / 2;
        }

        // ---- Step 3: Mirror (front camera) ----
        if (needMirror) {
            // Mirror in-place into the rotation output buffer (or src buffer if no rotation)
            // We use i420Rot_ as scratch; if rotation didn't run, we need to copy first
            uint8_t* mirrY;
            uint8_t* mirrU;
            uint8_t* mirrV;

            if (rotMode == libyuv::kRotate0) {
                // No rotation was done, i420Rot_ is available as scratch
                mirrY = i420Rot_;
                mirrU = mirrY + rotW * rotH;
                mirrV = mirrU + (rotW / 2) * (rotH / 2);
            } else {
                // Rotation already wrote to i420Rot_, mirror in-place by using
                // a temporary swap. For simplicity, mirror into i420Src_ (which is
                // now free since we already read from it).
                mirrY = i420Src_;
                mirrU = mirrY + rotW * rotH;
                mirrV = mirrU + (rotW / 2) * (rotH / 2);
            }

            ret = libyuv::I420Mirror(
                curY, curStrideY,
                curU, curStrideUV,
                curV, curStrideUV,
                mirrY, rotW,
                mirrU, rotW / 2,
                mirrV, rotW / 2,
                rotW, rotH);

            if (ret != 0) {
                LOGE("I420Mirror failed: %d", ret);
                return false;
            }

            curY = mirrY;
            curU = mirrU;
            curV = mirrV;
            curStrideY = rotW;
            curStrideUV = rotW / 2;
        }

        // ---- Step 4: Compute letterbox geometry ----
        float scale = std::min(
            static_cast<float>(targetSize) / rotW,
            static_cast<float>(targetSize) / rotH);
        int newW = static_cast<int>(rotW * scale);
        int newH = static_cast<int>(rotH * scale);
        int padLeft = (targetSize - newW) / 2;
        int padTop  = (targetSize - newH) / 2;

        // ---- Step 5: Zero padding regions ----
        // Clear the full buffer once (simplest approach, fast for 1.2MB)
        memset(outputBuffer, 0, targetSize * targetSize * RGB_CHANNELS);

        // ---- Step 6: Convert I420 → RGB24 with letterbox placement ----
        uint8_t* outBase = outputBuffer;
        int outRowBytes = targetSize * RGB_CHANNELS;

        if (newW == rotW && newH == rotH) {
            // No scaling needed — convert directly to letterbox position
            uint8_t* dstPtr = outBase + padTop * outRowBytes + padLeft * RGB_CHANNELS;
            ret = libyuv::I420ToRGB24(
                curY, curStrideY,
                curU, curStrideUV,
                curV, curStrideUV,
                dstPtr, outRowBytes,
                rotW, rotH);
        } else {
            // Scale I420 first, then convert to RGB at letterbox position
            // Use the tail end of i420Rot_ for scaled I420 (safe: we're done with rotated data)
            int scaledI420Size = newW * newH * 3 / 2;
            // Allocate a temporary scaled buffer to avoid overwriting curY/U/V
            // (which may point into i420Rot_ or i420Src_)
            static thread_local uint8_t* scaledBuf = nullptr;
            static thread_local int scaledBufSize = 0;
            if (scaledBufSize < scaledI420Size) {
                free(scaledBuf);
                scaledBuf = static_cast<uint8_t*>(malloc(scaledI420Size));
                scaledBufSize = scaledI420Size;
            }

            uint8_t* sY = scaledBuf;
            uint8_t* sU = sY + newW * newH;
            uint8_t* sV = sU + (newW / 2) * (newH / 2);

            ret = libyuv::I420Scale(
                curY, curStrideY,
                curU, curStrideUV,
                curV, curStrideUV,
                rotW, rotH,
                sY, newW,
                sU, newW / 2,
                sV, newW / 2,
                newW, newH,
                libyuv::kFilterBilinear);

            if (ret != 0) {
                LOGE("I420Scale failed: %d", ret);
                return false;
            }

            // Convert scaled I420 to RGB24 at the letterbox position
            uint8_t* dstPtr = outBase + padTop * outRowBytes + padLeft * RGB_CHANNELS;
            ret = libyuv::I420ToRGB24(
                sY, newW,
                sU, newW / 2,
                sV, newW / 2,
                dstPtr, outRowBytes,
                newW, newH);
        }

        if (ret != 0) {
            LOGE("I420ToRGB24 failed: %d", ret);
            return false;
        }

        return true;
    }

    /**
     * Compute letterbox parameters and write them to the output int array.
     * Array layout: [scale*10000, padLeft, padTop, rotatedW, rotatedH, newW, newH]
     */
    void computeLetterboxInfo(int srcW, int srcH, int rotation, int targetSize,
                               int* outInfo) {
        int rotW = srcW, rotH = srcH;
        if (rotation == 90 || rotation == 270) {
            rotW = srcH;
            rotH = srcW;
        }
        float scale = std::min(
            static_cast<float>(targetSize) / rotW,
            static_cast<float>(targetSize) / rotH);
        int newW = static_cast<int>(rotW * scale);
        int newH = static_cast<int>(rotH * scale);
        int padLeft = (targetSize - newW) / 2;
        int padTop  = (targetSize - newH) / 2;

        outInfo[0] = static_cast<int>(scale * 10000); // Fixed-point scale * 10000
        outInfo[1] = padLeft;
        outInfo[2] = padTop;
        outInfo[3] = rotW;
        outInfo[4] = rotH;
        outInfo[5] = newW;
        outInfo[6] = newH;
    }

    void release() {
        std::lock_guard<std::mutex> lock(bufMutex_);
        free(i420Src_);
        free(i420Rot_);
        i420Src_ = nullptr;
        i420Rot_ = nullptr;
        i420SrcSize_ = 0;
        i420RotSize_ = 0;
    }

private:
    uint8_t* i420Src_ = nullptr;
    int      i420SrcSize_ = 0;
    uint8_t* i420Rot_ = nullptr;
    int      i420RotSize_ = 0;
    std::mutex bufMutex_;
};

// Global processor instance (one per process; the app has one Activity)
static YuvProcessor g_processor;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_xiaohaoo_yolo_util_YuvPreprocessor_nativeYuvToRgbLetterbox(
    JNIEnv* env, jobject thiz,
    jobject y_buffer, jobject u_buffer, jobject v_buffer,
    jint y_row_stride, jint uv_row_stride, jint uv_pixel_stride,
    jint src_width, jint src_height,
    jint rotation, jboolean need_mirror,
    jobject output_buffer, jint target_size)
{
    auto* yPtr = static_cast<const uint8_t*>(env->GetDirectBufferAddress(y_buffer));
    auto* uPtr = static_cast<const uint8_t*>(env->GetDirectBufferAddress(u_buffer));
    auto* vPtr = static_cast<const uint8_t*>(env->GetDirectBufferAddress(v_buffer));
    auto* outPtr = static_cast<uint8_t*>(env->GetDirectBufferAddress(output_buffer));

    if (!yPtr || !uPtr || !vPtr || !outPtr) {
        LOGE("Failed to get direct buffer address");
        return JNI_FALSE;
    }

    g_processor.ensureBuffers(src_width, src_height);

    bool ok = g_processor.process(
        yPtr, uPtr, vPtr,
        y_row_stride, uv_row_stride, uv_pixel_stride,
        src_width, src_height,
        rotation, need_mirror,
        outPtr, target_size);

    return ok ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jintArray JNICALL
Java_com_xiaohaoo_yolo_util_YuvPreprocessor_nativeComputeLetterboxInfo(
    JNIEnv* env, jobject thiz,
    jint src_width, jint src_height,
    jint rotation, jint target_size)
{
    int info[7];
    g_processor.computeLetterboxInfo(src_width, src_height, rotation, target_size, info);

    jintArray result = env->NewIntArray(7);
    if (result) {
        env->SetIntArrayRegion(result, 0, 7, info);
    }
    return result;
}

JNIEXPORT void JNICALL
Java_com_xiaohaoo_yolo_util_YuvPreprocessor_nativeRelease(
    JNIEnv* env, jobject thiz)
{
    g_processor.release();
    LOGI("Native resources released");
}

} // extern "C"
