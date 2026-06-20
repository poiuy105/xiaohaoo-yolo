package com.xiaohaoo.yolo.camera

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.util.Size
import android.view.Surface

/**
 * Camera2 lifecycle manager with precise control over ISP output streams.
 *
 * Provides two output streams:
 * - Preview: SurfaceTexture at high resolution for display
 * - Analysis: ImageReader at 640×480 YUV_420_888 for model inference
 *
 * Handles camera enumeration, device open/close, capture session creation,
 * auto-focus configuration, and front/rear camera switching.
 */
class Camera2Manager(private val context: Context) {

    companion object {
        private const val TAG = "Camera2Manager"
        const val ANALYSIS_WIDTH = 640
        const val ANALYSIS_HEIGHT = 480
    }

    private val cameraManager =
        context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var captureRequestBuilder: CaptureRequest.Builder? = null

    private val cameraThread = HandlerThread("CameraThread").apply {
        start()
    }
    val cameraHandler = Handler(cameraThread.looper)

    private val inferenceThread = HandlerThread("InferenceThread").apply {
        start()
    }
    val inferenceHandler = Handler(inferenceThread.looper)

    // ImageReader for analysis stream (640×480 YUV_420_888)
    var imageReader: ImageReader? = null
        private set

    // Current camera state
    var isFrontCamera: Boolean = false
        private set
    var sensorOrientation: Int = 0
        private set
    var analysisSize: Size = Size(ANALYSIS_WIDTH, ANALYSIS_HEIGHT)
        private set

    // Callbacks
    var onCameraOpened: (() -> Unit)? = null
    var onCameraError: ((String) -> Unit)? = null
    var onFrameAvailable: ((Image) -> Unit)? = null

    /**
     * Initialize the analysis ImageReader.
     * Call once before opening any camera.
     */
    fun initImageReader() {
        imageReader?.close()
        imageReader = ImageReader.newInstance(
            ANALYSIS_WIDTH, ANALYSIS_HEIGHT,
            ImageFormat.YUV_420_888,
            2 // double buffering: one being processed, one being filled
        ).apply {
            setOnImageAvailableListener({ reader ->
                val image = reader.acquireLatestImage() ?: return@setOnImageAvailableListener
                onFrameAvailable?.invoke(image)
                // Note: consumer MUST call image.close()
            }, inferenceHandler)
        }
        Log.d(TAG, "ImageReader initialized: ${ANALYSIS_WIDTH}x${ANALYSIS_HEIGHT} YUV_420_888")
    }

    /**
     * Open the camera with the specified facing.
     * @param facing CameraCharacteristics.LENS_FACING_BACK or LENS_FACING_FRONT
     */
    fun openCamera(facing: Int = CameraCharacteristics.LENS_FACING_BACK) {
        closeCamera()

        val cameraId = findCameraId(facing)
        if (cameraId == null) {
            val msg = "No camera found with facing: $facing"
            Log.e(TAG, msg)
            onCameraError?.invoke(msg)
            return
        }

        isFrontCamera = facing == CameraCharacteristics.LENS_FACING_FRONT

        // Read sensor orientation
        val characteristics = cameraManager.getCameraCharacteristics(cameraId)
        sensorOrientation = characteristics.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 0
        Log.d(TAG, "Opening camera $cameraId (facing=$facing, sensorOrientation=$sensorOrientation)")

        // Verify analysis size is supported
        val configMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val outputSizes = configMap?.getOutputSizes(ImageFormat.YUV_420_888)
        val supported = outputSizes?.any {
            it.width == ANALYSIS_WIDTH && it.height == ANALYSIS_HEIGHT
        } ?: false
        if (!supported) {
            Log.w(TAG, "640×480 YUV not in supported sizes: ${outputSizes?.toList()}")
            // Try to find the closest supported size
            val closest = outputSizes?.minByOrNull {
                Math.abs(it.width - ANALYSIS_WIDTH) + Math.abs(it.height - ANALYSIS_HEIGHT)
            }
            if (closest != null) {
                Log.w(TAG, "Using closest supported size: $closest")
                analysisSize = closest
                // Recreate ImageReader with the supported size
                imageReader?.close()
                imageReader = ImageReader.newInstance(
                    closest.width, closest.height,
                    ImageFormat.YUV_420_888, 2
                ).apply {
                    setOnImageAvailableListener({ reader ->
                        val image = reader.acquireLatestImage()
                            ?: return@setOnImageAvailableListener
                        onFrameAvailable?.invoke(image)
                    }, inferenceHandler)
                }
            }
        }

        try {
            cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
                override fun onOpened(camera: CameraDevice) {
                    Log.d(TAG, "Camera opened: ${camera.id}")
                    cameraDevice = camera
                    onCameraOpened?.invoke()
                }

                override fun onDisconnected(camera: CameraDevice) {
                    Log.w(TAG, "Camera disconnected: ${camera.id}")
                    camera.close()
                    cameraDevice = null
                }

                override fun onError(camera: CameraDevice, error: Int) {
                    val msg = "Camera error: ${camera.id}, error=$error"
                    Log.e(TAG, msg)
                    camera.close()
                    cameraDevice = null
                    onCameraError?.invoke(msg)
                }
            }, cameraHandler)
        } catch (e: SecurityException) {
            Log.e(TAG, "Camera permission not granted", e)
            onCameraError?.invoke("Camera permission not granted")
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to open camera", e)
            onCameraError?.invoke(e.message ?: "Camera access failed")
        }
    }

    /**
     * Create a capture session with the given preview and analysis surfaces.
     * Must be called after onCameraOpened callback fires.
     *
     * @param previewSurface Surface from TextureView for preview display
     */
    fun createCaptureSession(previewSurface: Surface) {
        val device = cameraDevice ?: run {
            Log.e(TAG, "Camera not opened yet")
            return
        }
        val analysisSurface = imageReader?.surface ?: run {
            Log.e(TAG, "ImageReader not initialized")
            return
        }

        // Close existing session
        captureSession?.close()

        try {
            captureRequestBuilder = device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addTarget(previewSurface)
                addTarget(analysisSurface)
                // Auto-focus continuous
                set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                // Auto-exposure
                set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON)
            }

            val sessionCallback = object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    Log.d(TAG, "Capture session configured")
                    captureSession = session
                    try {
                        session.setRepeatingRequest(
                            captureRequestBuilder!!.build(),
                            null,
                            cameraHandler
                        )
                        Log.d(TAG, "Repeating capture request started")
                    } catch (e: CameraAccessException) {
                        Log.e(TAG, "Failed to start repeating request", e)
                    }
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    Log.e(TAG, "Capture session configuration failed")
                    onCameraError?.invoke("Session configuration failed")
                }
            }

            device.createCaptureSession(
                listOf(previewSurface, analysisSurface),
                sessionCallback,
                cameraHandler
            )
        } catch (e: CameraAccessException) {
            Log.e(TAG, "Failed to create capture session", e)
            onCameraError?.invoke(e.message ?: "Session creation failed")
        }
    }

    /**
     * Switch to the specified camera facing.
     * Tears down current session and reopens with new camera.
     */
    fun switchCamera(facing: Int) {
        openCamera(facing)
    }

    /**
     * Close the capture session and camera device.
     * Safe to call multiple times.
     */
    fun closeCamera() {
        try {
            captureSession?.close()
            captureSession = null
            cameraDevice?.close()
            cameraDevice = null
            captureRequestBuilder = null
        } catch (e: Exception) {
            Log.w(TAG, "Error closing camera", e)
        }
    }

    /**
     * Release all resources including threads.
     * Call from Activity.onDestroy().
     */
    fun release() {
        closeCamera()
        imageReader?.close()
        imageReader = null
        cameraThread.quitSafely()
        inferenceThread.quitSafely()
        Log.d(TAG, "Camera2Manager released")
    }

    /**
     * Find camera ID matching the specified lens facing.
     */
    private fun findCameraId(facing: Int): String? {
        for (id in cameraManager.cameraIdList) {
            val characteristics = cameraManager.getCameraCharacteristics(id)
            val lensFacing = characteristics.get(CameraCharacteristics.LENS_FACING)
            if (lensFacing == facing) return id
        }
        return null
    }
}
