package com.xiaohaoo.yolo.util

import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * High-performance YUV→RGB preprocessing via libyuv NDK.
 *
 * Handles YUV_420_888 stride normalization, YUV-plane rotation,
 * optional horizontal mirror (front camera), and letterbox scaling
 * into a pre-allocated 640×640×3 UINT8 tensor buffer.
 *
 * All operations are NEON-optimized on ARM via libyuv.
 */
class YuvPreprocessor {

    companion object {
        private const val TAG = "YuvPreprocessor"
        const val TARGET_SIZE = 640

        init {
            System.loadLibrary("yolo_jni")
        }
    }

    /**
     * Pre-allocated tensor input buffer: 640×640×3 bytes, UINT8.
     * This buffer is passed directly to TFLite Interpreter.run() — zero copy.
     */
    val tensorInputBuffer: ByteBuffer = ByteBuffer.allocateDirect(
        TARGET_SIZE * TARGET_SIZE * 3
    ).apply {
        order(ByteOrder.nativeOrder())
    }

    /**
     * Letterbox metadata updated after each [process] call.
     * Used by OverlayView for coordinate reverse-mapping.
     */
    data class LetterboxInfo(
        val scale: Float,
        val padLeft: Int,
        val padTop: Int,
        val rotatedWidth: Int,
        val rotatedHeight: Int,
        val scaledWidth: Int,
        val scaledHeight: Int
    )

    var letterboxInfo: LetterboxInfo = LetterboxInfo(1f, 0, 0, 640, 480, 640, 480)
        private set

    /**
     * Process a YUV_420_888 frame into the tensor input buffer.
     *
     * Pipeline: stride normalize → rotate → mirror → scale → RGB24 letterbox
     *
     * @param yBuffer  Y plane DirectByteBuffer from Image.Plane
     * @param uBuffer  U plane DirectByteBuffer from Image.Plane
     * @param vBuffer  V plane DirectByteBuffer from Image.Plane
     * @param yRowStride  Y plane row stride
     * @param uvRowStride  UV plane row stride
     * @param uvPixelStride  UV plane pixel stride (1=planar, 2=semi-planar)
     * @param width  Source frame width
     * @param height  Source frame height
     * @param rotation  Rotation angle: 0, 90, 180, or 270
     * @param needMirror  true for front camera (horizontal mirror)
     * @return true if processing succeeded
     */
    fun process(
        yBuffer: ByteBuffer,
        uBuffer: ByteBuffer,
        vBuffer: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        width: Int,
        height: Int,
        rotation: Int,
        needMirror: Boolean
    ): Boolean {
        tensorInputBuffer.clear()

        val ok = nativeYuvToRgbLetterbox(
            yBuffer, uBuffer, vBuffer,
            yRowStride, uvRowStride, uvPixelStride,
            width, height,
            rotation, needMirror,
            tensorInputBuffer, TARGET_SIZE
        )

        if (ok) {
            val info = nativeComputeLetterboxInfo(width, height, rotation, TARGET_SIZE)
            if (info != null && info.size == 7) {
                letterboxInfo = LetterboxInfo(
                    scale = info[0] / 10000f,
                    padLeft = info[1],
                    padTop = info[2],
                    rotatedWidth = info[3],
                    rotatedHeight = info[4],
                    scaledWidth = info[5],
                    scaledHeight = info[6]
                )
            }
        } else {
            Log.e(TAG, "nativeYuvToRgbLetterbox failed")
        }

        return ok
    }

    fun release() {
        nativeRelease()
    }

    // ---- Native methods ----

    private external fun nativeYuvToRgbLetterbox(
        yBuffer: ByteBuffer,
        uBuffer: ByteBuffer,
        vBuffer: ByteBuffer,
        yRowStride: Int,
        uvRowStride: Int,
        uvPixelStride: Int,
        srcWidth: Int,
        srcHeight: Int,
        rotation: Int,
        needMirror: Boolean,
        outputBuffer: ByteBuffer,
        targetSize: Int
    ): Boolean

    private external fun nativeComputeLetterboxInfo(
        srcWidth: Int,
        srcHeight: Int,
        rotation: Int,
        targetSize: Int
    ): IntArray?

    private external fun nativeRelease()
}
