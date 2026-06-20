package com.xiaohaoo.yolo.ui.activity

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.Surface
import android.view.TextureView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.xiaohaoo.yolo.R
import com.xiaohaoo.yolo.camera.Camera2Manager
import com.xiaohaoo.yolo.databinding.ActivityMainBinding
import com.xiaohaoo.yolo.util.DetectorUtils
import com.xiaohaoo.yolo.util.YuvPreprocessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.FileInputStream
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private val binding: ActivityMainBinding by lazy {
        ActivityMainBinding.inflate(layoutInflater)
    }

    // ---- Core components ----
    private lateinit var camera2Manager: Camera2Manager
    private lateinit var yuvPreprocessor: YuvPreprocessor
    private var interpreter: Interpreter? = null
    private lateinit var outputBuffer: TensorBuffer
    private lateinit var floatInputBuffer: TensorBuffer
    private var needsFloatInput: Boolean = true

    // ---- State ----
    private var previewSurfaceTexture: SurfaceTexture? = null
    private var previewSurface: Surface? = null
    private var activeDelegate = DelegateType.CPU
    private var labels: List<String> = emptyList()

    // ---- Performance metrics (shared with OverlayView) ----
    var preprocessMs: Long = 0L
    var inferenceMs: Long = 0L
    var postprocessMs: Long = 0L

    enum class DelegateType(val displayName: String) {
        NNAPI("NNAPI/DSP"), GPU("GPU"), CPU("CPU")
    }

    // ---- Lifecycle ----

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(binding.root)
        WindowCompat.getInsetsController(window, window.decorView).also {
            it.isAppearanceLightStatusBars = false
        }

        // Permissions
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), REQUEST_CAMERA)
            return // will continue after permission granted
        }

        initializeApp()
    }

    private fun initializeApp() {
        // Initialize components
        yuvPreprocessor = YuvPreprocessor()
        camera2Manager = Camera2Manager(this)
        camera2Manager.initImageReader()

        // Pre-allocate inference buffers (reused every frame)
        outputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, 84, 8400), DataType.FLOAT32
        )
        floatInputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, 640, 640, 3), DataType.FLOAT32
        )

        // Load labels
        labels = loadLabels(this, "labels.txt")
        OverlayView.LABELS = labels

        // Load default model
        loadModel("yolov8n_int8.tflite")

        // Log tensor info
        interpreter?.let { interp ->
            for (i in 0 until interp.inputTensorCount) {
                val t = interp.getInputTensor(i)
                val qp = t.quantizationParams()
                Log.d(TAG, "Input[$i]: shape=${t.shape().contentToString()}, " +
                        "type=${t.dataType()}, scale=${qp.scale}, zp=${qp.zeroPoint}")
            }
            for (i in 0 until interp.outputTensorCount) {
                val t = interp.getOutputTensor(i)
                Log.d(TAG, "Output[$i]: shape=${t.shape().contentToString()}, type=${t.dataType()}")
            }
        }

        // Setup frame processing callback
        camera2Manager.onFrameAvailable = { image ->
            processFrame(image)
        }

        camera2Manager.onCameraOpened = {
            val surface = previewSurface
            if (surface != null && surface.isValid) {
                camera2Manager.createCaptureSession(surface)
            }
            // Reconfigure preview transform now that camera is open
            runOnUiThread {
                configureTransform(480, 640)
            }
        }

        // Setup TextureView
        binding.textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(
                surfaceTexture: SurfaceTexture, width: Int, height: Int
            ) {
                Log.d(TAG, "SurfaceTexture available: ${width}x${height}")
                previewSurfaceTexture = surfaceTexture
                startCamera()
            }

            override fun onSurfaceTextureSizeChanged(
                surfaceTexture: SurfaceTexture, width: Int, height: Int
            ) {
                configureTransform(480, 640)
            }

            override fun onSurfaceTextureDestroyed(surfaceTexture: SurfaceTexture): Boolean {
                Log.d(TAG, "SurfaceTexture destroyed")
                previewSurfaceTexture = null
                previewSurface?.release()
                previewSurface = null
                camera2Manager.closeCamera()
                return true
            }

            override fun onSurfaceTextureUpdated(surfaceTexture: SurfaceTexture) {}
        }

        setupSettings()
    }

    override fun onResume() {
        super.onResume()
        if (::camera2Manager.isInitialized && previewSurfaceTexture != null) {
            startCamera()
        }
    }

    override fun onPause() {
        super.onPause()
        if (::camera2Manager.isInitialized) {
            camera2Manager.closeCamera()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::camera2Manager.isInitialized) {
            camera2Manager.release()
        }
        if (::yuvPreprocessor.isInitialized) {
            yuvPreprocessor.release()
        }
        interpreter?.close()
        previewSurface?.release()
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            initializeApp()
        } else {
            finish()
        }
    }

    // ---- Camera ----

    private fun startCamera() {
        val st = previewSurfaceTexture ?: return

        // Set buffer to portrait dimensions (w<h).
        // Camera2 renders the sensor output upright to match this aspect ratio,
        // so no rotation transform is needed in the display matrix.
        val portraitW = 480
        val portraitH = 640
        st.setDefaultBufferSize(portraitW, portraitH)

        previewSurface?.release()
        previewSurface = Surface(st)

        val facing = if (camera2Manager.isFrontCamera) {
            CameraCharacteristics.LENS_FACING_FRONT
        } else {
            CameraCharacteristics.LENS_FACING_BACK
        }
        camera2Manager.openCamera(facing)

        // Configure transform now (no rotation needed, just fit-center scale)
        configureTransform(portraitW, portraitH)
    }

    /**
     * Configure TextureView transform matrix.
     * Since the buffer is set to portrait dimensions, Camera2 renders it upright.
     * We only need fit-center scaling and optional front-camera mirroring.
     *
     * @param bufWidth  buffer width (portrait: 480)
     * @param bufHeight buffer height (portrait: 640)
     */
    private fun configureTransform(bufWidth: Int, bufHeight: Int) {
        val textureView = binding.textureView
        val viewW = textureView.width
        val viewH = textureView.height
        if (viewW == 0 || viewH == 0) return

        val isFront = camera2Manager.isFrontCamera
        val bw = bufWidth.toFloat()
        val bh = bufHeight.toFloat()

        // Width-fill: preview always fills screen width, height keeps aspect ratio
        val scale = viewW / bw

        val matrix = Matrix()
        val cx = viewW / 2f
        val cy = viewH / 2f

        // 1. Move view center to origin
        matrix.postTranslate(-cx, -cy)
        // 2. Fit-center scale
        matrix.postScale(scale, scale)
        // 3. Front camera mirror (horizontal flip for natural selfie view)
        if (isFront) {
            matrix.postScale(-1f, 1f)
        }
        // 4. Translate back to view center
        matrix.postTranslate(cx, cy)

        textureView.setTransform(matrix)
    }

    // ---- Frame Processing ----

    private fun processFrame(image: android.media.Image) {
        val t0 = SystemClock.elapsedRealtime()

        // Extract YUV planes
        val yPlane = image.planes[0].buffer
        val uPlane = image.planes[1].buffer
        val vPlane = image.planes[2].buffer
        val yRowStride = image.planes[0].rowStride
        val uvRowStride = image.planes[1].rowStride
        val uvPixelStride = image.planes[1].pixelStride
        val width = image.width
        val height = image.height

        // Compute rotation angle
        val rotation = computeRotation()
        val needMirror = camera2Manager.isFrontCamera

        // JNI: rotate + mirror + YUV→RGB + letterbox → tensorInputBuffer
        val ok = yuvPreprocessor.process(
            yPlane, uPlane, vPlane,
            yRowStride, uvRowStride, uvPixelStride,
            width, height,
            rotation, needMirror
        )

        val t1 = SystemClock.elapsedRealtime()
        preprocessMs = t1 - t0

        // Return image to ImageReader pool ASAP
        image.close()

        if (!ok) return

        // Inference — normalize to float if model expects FLOAT32 input
        val inputBuffer = if (needsFloatInput) {
            val srcBuf = yuvPreprocessor.tensorInputBuffer
            val dstBuf = floatInputBuffer.buffer
            srcBuf.rewind()
            dstBuf.rewind()
            val floatArray = FloatArray(640 * 640 * 3)
            for (i in floatArray.indices) {
                floatArray[i] = (srcBuf.get(i).toInt() and 0xFF) / 255.0f
            }
            dstBuf.asFloatBuffer().put(floatArray)
            floatInputBuffer.buffer
        } else {
            yuvPreprocessor.tensorInputBuffer
        }
        interpreter?.run(inputBuffer, outputBuffer.buffer)

        val t2 = SystemClock.elapsedRealtime()
        inferenceMs = t2 - t1

        // Post-processing
        val boxes = DetectorUtils.boundingBox(outputBuffer.floatArray)

        val t3 = SystemClock.elapsedRealtime()
        postprocessMs = t3 - t2

        // Update UI
        val info = yuvPreprocessor.letterboxInfo
        runOnUiThread {
            binding.overlayView.updateDetections(
                boxes, info, rotation, needMirror,
                width, height, activeDelegate
            )
        }
    }

    /**
     * Compute the rotation angle needed for the analysis frame.
     * This is the angle to rotate the camera frame so it matches the display orientation.
     */
    private fun computeRotation(): Int {
        val sensorOrientation = camera2Manager.sensorOrientation
        val isFront = camera2Manager.isFrontCamera
        val displayRotation = display?.rotation ?: 0
        val displayDegrees = when (displayRotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return if (isFront) {
            (sensorOrientation + displayDegrees) % 360
        } else {
            (sensorOrientation - displayDegrees + 360) % 360
        }
    }

    // ---- Model Loading ----

    private fun loadModel(modelFileName: String) {
        val modelFile = loadModelFile(this, modelFileName)
        val isInt8 = modelFileName.contains("int8")

        // Strategy 1: NNAPI (best for INT8 on Qualcomm DSP)
        if (isInt8) {
            try {
                val nnOptions = NnApiDelegate.Options().apply {
                    executionPreference = NnApiDelegate.Options.EXECUTION_PREFERENCE_SUSTAINED_SPEED
                }
                val nnDelegate = NnApiDelegate(nnOptions)
                val options = Interpreter.Options().apply {
                    addDelegate(nnDelegate)
                    numThreads = 1
                }
                interpreter?.close()
                interpreter = Interpreter(modelFile, options)
                activeDelegate = DelegateType.NNAPI
                Log.d(TAG, "✓ NNAPI delegate active (Hexagon DSP) for $modelFileName")
                detectInputType()
                return
            } catch (e: Exception) {
                Log.w(TAG, "NNAPI failed: ${e.message}, trying GPU...")
            }
        }

        // Strategy 2: GPU delegate (good for Float16/Float32)
        try {
            val gpuDelegate = GpuDelegate()
            val options = Interpreter.Options().apply {
                addDelegate(gpuDelegate)
            }
            interpreter?.close()
            interpreter = Interpreter(modelFile, options)
            activeDelegate = DelegateType.GPU
            Log.d(TAG, "✓ GPU delegate active for $modelFileName")
            detectInputType()
            return
        } catch (e: Exception) {
            Log.w(TAG, "GPU delegate failed: ${e.message}, falling back to CPU")
        }

        // Strategy 3: CPU fallback
        val cpuOptions = Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors().coerceAtMost(4)
        }
        interpreter?.close()
        interpreter = Interpreter(modelFile, cpuOptions)
        activeDelegate = DelegateType.CPU
        Log.d(TAG, "✓ CPU fallback for $modelFileName")
        detectInputType()
    }

    /**
     * Detect model input tensor type. Some "int8" models still use FLOAT32 input.
     */
    private fun detectInputType() {
        val interp = interpreter ?: return
        val inputType = interp.getInputTensor(0).dataType()
        needsFloatInput = (inputType == DataType.FLOAT32)
        Log.d(TAG, "Model input type: $inputType, needsFloatInput=$needsFloatInput")
    }

    // ---- Settings UI ----

    private fun setupSettings() {
        binding.cameraToggleGroup.check(R.id.btnBackCamera)
        binding.modelToggleGroup.check(R.id.btnModelInt8)

        // Camera toggle
        binding.cameraToggleGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            val facing = when (checkedId) {
                R.id.btnFrontCamera -> CameraCharacteristics.LENS_FACING_FRONT
                else -> CameraCharacteristics.LENS_FACING_BACK
            }
            camera2Manager.closeCamera()
            camera2Manager.switchCamera(facing)
            Log.d(TAG, "Camera switched to: ${if (facing == CameraCharacteristics.LENS_FACING_FRONT) "front" else "back"}")
        }

        // Model toggle
        binding.modelToggleGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            val modelFileName = when (checkedId) {
                R.id.btnModelFloat16 -> "yolov8n_float16.tflite"
                R.id.btnModelFloat32 -> "yolov8n_float32.tflite"
                else -> "yolov8n_int8.tflite"
            }
            val dialog = MaterialAlertDialogBuilder(this)
                .setTitle("提示")
                .setMessage("模型加载中...")
                .setCancelable(false)
                .show()
            lifecycleScope.launch(Dispatchers.IO) {
                loadModel(modelFileName)
                runOnUiThread { dialog.dismiss() }
            }
            Log.d(TAG, "Model switched to: $modelFileName")
        }

        // Confidence slider
        binding.confidenceSlider.addOnChangeListener { _, value, _ ->
            DetectorUtils.CONFIDENCE_THRESHOLD = value
            binding.confidenceValue.text = String.format("%.2f", value)
        }

        // IoU slider
        binding.iouSlider.addOnChangeListener { _, value, _ ->
            DetectorUtils.IOU_THRESHOLD = value
            binding.iouValue.text = String.format("%.2f", value)
        }
    }

    // ---- Utility ----

    companion object {
        private const val TAG = "MainActivity"
        private const val REQUEST_CAMERA = 0

        /**
         * Load a TFLite model from assets into a memory-mapped buffer.
         */
        fun loadModelFile(context: Context, assetName: String): MappedByteBuffer {
            val fd = context.assets.openFd(assetName)
            val inputStream = FileInputStream(fd.fileDescriptor)
            val channel = inputStream.channel
            return channel.map(
                FileChannel.MapMode.READ_ONLY,
                fd.startOffset, fd.declaredLength
            )
        }

        /**
         * Load class labels from assets text file.
         */
        fun loadLabels(context: Context, assetName: String): List<String> {
            return context.assets.open(assetName).use { input ->
                BufferedReader(InputStreamReader(input)).readLines()
            }
        }
    }
}
