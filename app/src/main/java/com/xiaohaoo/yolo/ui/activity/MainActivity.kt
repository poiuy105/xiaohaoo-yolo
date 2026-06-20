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

        // Pre-allocate inference output buffer (reused every frame)
        outputBuffer = TensorBuffer.createFixedSize(
            intArrayOf(1, 84, 8400), DataType.FLOAT32
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
            // Reconfigure preview transform now that sensorOrientation is correct
            runOnUiThread {
                val viewW = binding.textureView.width
                val viewH = binding.textureView.height
                if (viewW > 0 && viewH > 0) {
                    configureTransform(viewW, viewH)
                }
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
                configureTransform(width, height)
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

        // Configure SurfaceTexture size to match display
        val displaySize = windowManager.currentWindowMetrics
        val viewW = displaySize.bounds.width()
        val viewH = displaySize.bounds.height()
        st.setDefaultBufferSize(viewW, viewH)

        previewSurface?.release()
        previewSurface = Surface(st)

        val facing = if (camera2Manager.isFrontCamera) {
            CameraCharacteristics.LENS_FACING_FRONT
        } else {
            CameraCharacteristics.LENS_FACING_BACK
        }
        camera2Manager.openCamera(facing)

        // Configure preview transform after camera opens (sensor orientation known)
        configureTransform(viewW, viewH)
    }

    /**
     * Configure TextureView transform matrix to properly display the camera preview.
     * Accounts for sensor orientation, display rotation, aspect ratio, and front camera mirror.
     */
    private fun configureTransform(viewWidth: Int, viewHeight: Int) {
        val textureView = binding.textureView
        if (viewWidth == 0 || viewHeight == 0) return

        val rotation = display?.rotation ?: 0
        val sensorOrientation = camera2Manager.sensorOrientation
        val isFront = camera2Manager.isFrontCamera

        // Camera buffer dimensions (landscape sensor output)
        val bufW = viewWidth.toFloat()
        val bufH = viewHeight.toFloat()

        // Compute display rotation
        val displayDegrees = when (rotation) {
            Surface.ROTATION_0 -> 0
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }

        // The preview image needs to be rotated to match the display orientation
        val rotationAngle = if (isFront) {
            (sensorOrientation + displayDegrees) % 360
        } else {
            (sensorOrientation - displayDegrees + 360) % 360
        }

        // After rotation, the effective dimensions swap for 90°/270° rotations
        val rotatedW: Float
        val rotatedH: Float
        if (rotationAngle == 90 || rotationAngle == 270) {
            rotatedW = bufH
            rotatedH = bufW
        } else {
            rotatedW = bufW
            rotatedH = bufH
        }

        val matrix = Matrix()
        val viewCenterX = viewWidth / 2f
        val viewCenterY = viewHeight / 2f

        // Step 1: Translate texture view center to origin
        matrix.postTranslate(-viewCenterX, -viewCenterY)

        // Step 2: Scale to match rotated camera image dimensions
        val scaleX = rotatedW / viewWidth
        val scaleY = rotatedH / viewHeight
        matrix.postScale(scaleX, scaleY)

        // Step 3: Rotate to compensate for sensor orientation vs display rotation
        matrix.postRotate((360 - rotationAngle).toFloat())

        // Step 4: Scale to fill the view while preserving aspect ratio
        val fillScale = maxOf(
            viewWidth / rotatedW,
            viewHeight / rotatedH
        )
        matrix.postScale(fillScale, fillScale)

        // Step 5: Front camera mirror
        if (isFront) {
            matrix.postScale(-1f, 1f)
        }

        // Step 6: Translate back to view center
        matrix.postTranslate(viewCenterX, viewCenterY)

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

        // Inference
        interpreter?.run(yuvPreprocessor.tensorInputBuffer, outputBuffer.buffer)

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
