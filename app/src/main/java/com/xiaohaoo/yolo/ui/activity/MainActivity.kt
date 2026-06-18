package com.xiaohaoo.yolo.ui.activity

import android.Manifest
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.PointF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.OrientationEventListener
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.xiaohaoo.yolo.R
import com.xiaohaoo.yolo.databinding.ActivityMainBinding
import com.xiaohaoo.yolo.util.DetectorUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.util.concurrent.Executors


class MainActivity : AppCompatActivity() {
    private val binding: ActivityMainBinding by lazy {
        ActivityMainBinding.inflate(layoutInflater)
    }

    private val imageAnalysisExecutor by lazy {
        Executors.newSingleThreadExecutor()
    }

    private var interpreter: Interpreter? = null

    private val inputSize = Size(640, 640)
    private var imageSize = Size(720, 1280)
    private lateinit var imageProcessor: ImageProcessor

    private lateinit var imageAnalysis: ImageAnalysis
    private lateinit var preview: Preview
    private var cameraProvider: ProcessCameraProvider? = null
    private var currentCameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

    private val orientationEventListener by lazy {
        object : OrientationEventListener(this) {
            override fun onOrientationChanged(orientation: Int) {
                if (orientation != ORIENTATION_UNKNOWN) {
                    imageAnalysis.targetRotation = binding.previewView.display.rotation
                    preview.targetRotation = binding.previewView.display.rotation
                }
            }
        }
    }

    private fun updateProcessParam() {
        val scaleFactor = Math.min(inputSize.width.toFloat() / imageSize.width.toFloat(), inputSize.height.toFloat() / imageSize.height.toFloat())
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp((imageSize.height * scaleFactor).toInt(), (imageSize.width * scaleFactor).toInt(), ResizeOp.ResizeMethod.BILINEAR))
            .add(object : ImageOperator {
                override fun apply(image: TensorImage): TensorImage {
                    Bitmap.createBitmap(inputSize.width, inputSize.height, Bitmap.Config.ARGB_8888).also {
                        Canvas(it).drawBitmap(image.bitmap, 0.0f, 0.0f, null)
                        image.load(it)
                    }
                    return image
                }

                override fun getOutputImageWidth(inputImageHeight: Int, inputImageWidth: Int): Int {
                    return inputSize.width
                }

                override fun getOutputImageHeight(inputImageHeight: Int, inputImageWidth: Int): Int {
                    return inputSize.height
                }

                override fun inverseTransform(point: PointF, inputImageHeight: Int, inputImageWidth: Int): PointF {
                    return PointF(point.x * inputImageWidth.toFloat() / inputSize.width, point.y * inputImageHeight.toFloat() / inputSize.height)
                }
            }).add(NormalizeOp(0.0f, 255.0f)).add(CastOp(DataType.FLOAT32)).build()
    }

    private fun loadModel(modelFileName: String) {
        val model = FileUtil.loadMappedFile(this, modelFileName)
        val options = Interpreter.Options().apply {
            numThreads = Runtime.getRuntime().availableProcessors()
        }
        try {
            options.addDelegate(GpuDelegate())
            interpreter = Interpreter(model, options)
            Log.d(TAG, "Interpreter created with GPU delegate for $modelFileName")
        } catch (e: Exception) {
            Log.w(TAG, "GPU delegate failed, falling back to CPU: ${e.message}")
            val cpuOptions = Interpreter.Options().apply {
                numThreads = Runtime.getRuntime().availableProcessors()
            }
            interpreter = Interpreter(model, cpuOptions)
            Log.d(TAG, "Interpreter created with CPU only for $modelFileName")
        }
    }

    private fun bindCamera() {
        val cp = cameraProvider ?: return
        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_16_9_FALLBACK_AUTO_STRATEGY)
            .setResolutionStrategy(ResolutionStrategy(inputSize, ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER)).build()

        preview = Preview.Builder()
            .setTargetRotation(binding.previewView.display.rotation)
            .setResolutionSelector(resolutionSelector).build()
            .also { it.setSurfaceProvider(binding.previewView.surfaceProvider) }

        imageAnalysis = ImageAnalysis.Builder()
            .setTargetRotation(binding.previewView.display.rotation)
            .setResolutionSelector(resolutionSelector)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
            .also {
                it.setAnalyzer(imageAnalysisExecutor) {
                    val matrix = Matrix().apply { postRotate(it.imageInfo.rotationDegrees.toFloat()) }
                    val bitmap = Bitmap.createBitmap(it.toBitmap(), 0, 0, it.width, it.height, matrix, true)
                    it.close()
                    if (bitmap.width != imageSize.width || bitmap.height != imageSize.height) {
                        this.imageSize = Size(bitmap.width, bitmap.height)
                        updateProcessParam()
                    }
                    Log.d(TAG, "Input Image: Size: ${bitmap.width} × ${bitmap.height} Format: ${bitmap.colorSpace}")
                    val input = imageProcessor.process(TensorImage.fromBitmap(bitmap))
                    val output = TensorBuffer.createFixedSize(intArrayOf(1, 84, 8400), DataType.FLOAT32)
                    interpreter?.run(input.buffer, output.buffer)
                    lifecycleScope.launch(Dispatchers.IO) {
                        val boundingBoxList = DetectorUtils.boundingBox(output.floatArray)
                        runOnUiThread {
                            binding.overlayView.event(inputSize, imageSize, boundingBoxList)
                        }
                    }
                }
            }
        val useCaseGroup = UseCaseGroup.Builder().addUseCase(preview).addUseCase(imageAnalysis).setViewPort(binding.previewView.viewPort!!).build()
        cp.unbindAll()
        cp.bindToLifecycle(this, currentCameraSelector, useCaseGroup)
    }

    private fun setupSettings() {
        // Default checked states
        binding.cameraToggleGroup.check(R.id.btnBackCamera)
        binding.modelToggleGroup.check(R.id.btnModelInt8)

        // Camera toggle
        binding.cameraToggleGroup.addOnButtonCheckedListener { _, checkedId, isChecked ->
            if (!isChecked) return@addOnButtonCheckedListener
            currentCameraSelector = when (checkedId) {
                R.id.btnFrontCamera -> CameraSelector.DEFAULT_FRONT_CAMERA
                else -> CameraSelector.DEFAULT_BACK_CAMERA
            }
            cameraProvider?.let {
                bindCamera()
            }
            Log.d(TAG, "Camera switched to: ${if (checkedId == R.id.btnFrontCamera) "front" else "back"}")
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
                interpreter?.close()
                loadModel(modelFileName)
                runOnUiThread {
                    dialog.hide()
                }
            }
            Log.d(TAG, "Model switched to: $modelFileName")
        }

        // Confidence threshold slider
        binding.confidenceSlider.addOnChangeListener { _, value, _ ->
            DetectorUtils.CONFIDENCE_THRESHOLD = value
            binding.confidenceValue.text = String.format("%.2f", value)
            Log.d(TAG, "Confidence threshold: $value")
        }

        // IoU threshold slider
        binding.iouSlider.addOnChangeListener { _, value, _ ->
            DetectorUtils.IOU_THRESHOLD = value
            binding.iouValue.text = String.format("%.2f", value)
            Log.d(TAG, "IoU threshold: $value")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(binding.root)
        WindowCompat.getInsetsController(window, window.decorView).also {
            it.isAppearanceLightStatusBars = false
        }
        val permissions = PERMISSIONS.filter { ContextCompat.checkSelfPermission(this, it) != android.content.pm.PackageManager.PERMISSION_GRANTED }
        if (permissions.isNotEmpty()) {
            requestPermissions(permissions.toTypedArray(), 0)
        }
        val dialog = MaterialAlertDialogBuilder(this@MainActivity).setTitle("提示").setMessage("模型加载中...").setCancelable(false).show()
        loadModel("yolov8n_int8.tflite")
        updateProcessParam()
        OverlayView.LABELS = FileUtil.loadLabels(this@MainActivity, "labels.txt")
        for (i in 0 until interpreter!!.inputTensorCount) {
            val tensor = interpreter!!.getInputTensor(i)
            Log.d(TAG, "Input Tensor: Name: ${tensor.name()}, Shape: ${tensor.shape().contentToString()}, DataType: ${tensor.dataType()}")
        }
        for (i in 0 until interpreter!!.outputTensorCount) {
            val tensor = interpreter!!.getOutputTensor(i)
            Log.d(TAG, "Output Tensor: Name: ${tensor.name()}, Shape: ${tensor.shape().contentToString()}, DataType: ${tensor.dataType()}")
        }
        dialog.hide()

        val manager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        val characteristics = manager.getCameraCharacteristics("0")
        val configMap = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        val cameraPreviewSize = configMap?.getOutputSizes(SurfaceTexture::class.java)
        Log.d(TAG, "Camera Sizes: ${cameraPreviewSize.contentToString()} OutputFormats: ${configMap?.outputFormats.contentToString()}")

        setupSettings()

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            bindCamera()
            orientationEventListener.enable()
        }, ContextCompat.getMainExecutor(this))
    }

    override fun onDestroy() {
        super.onDestroy()
        imageAnalysisExecutor.shutdown()
        interpreter?.close()
        orientationEventListener.disable()
    }

    companion object {
        private const val TAG = "MainActivity"
        private val PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}
