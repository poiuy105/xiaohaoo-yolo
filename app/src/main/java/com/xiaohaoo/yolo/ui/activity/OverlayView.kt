package com.xiaohaoo.yolo.ui.activity

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.os.SystemClock
import android.util.AttributeSet
import android.view.View
import com.xiaohaoo.yolo.util.DetectorUtils
import com.xiaohaoo.yolo.util.YuvPreprocessor

/**
 * Overlay view that draws detection bounding boxes on top of the camera preview.
 *
 * Coordinate mapping:
 * 1. Model output [0,1] → 640×640 pixel coordinates
 * 2. Remove letterbox padding → rotated image coordinates
 * 3. Scale to fill the view (matching TextureView's fill-center transform)
 * 4. Offset for centering
 */
class OverlayView(context: Context, attrs: AttributeSet?) : View(context, attrs) {

    // ---- Detection results (pre-computed screen coordinates) ----
    private data class ScreenBox(
        val left: Float, val top: Float,
        val right: Float, val bottom: Float,
        val label: String
    )

    private var screenBoxes: List<ScreenBox>? = null

    // ---- Paint objects ----
    private val boxPaint = Paint().apply {
        color = Color.GREEN
        strokeWidth = 5f
        style = Paint.Style.STROKE
        isAntiAlias = true
    }

    private val labelBgPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val labelPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        textSize = 40f
        isAntiAlias = true
    }

    private val infoPaint = Paint().apply {
        color = Color.YELLOW
        style = Paint.Style.FILL
        textSize = 36f
        isAntiAlias = true
    }

    private val textBounds = Rect()

    // ---- Performance metrics (read from MainActivity fields) ----
    private var delegateName: String = "CPU"
    private var preprocessMs: Long = 0
    private var inferenceMs: Long = 0
    private var postprocessMs: Long = 0

    // ---- FPS tracking ----
    private val frameTimestamps = LongArray(30)
    private var frameIndex = 0
    private var fps = 0f

    // ---- Draw ----

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        // FPS calculation
        val now = SystemClock.elapsedRealtime()
        frameTimestamps[frameIndex % frameTimestamps.size] = now
        frameIndex++
        if (frameIndex > frameTimestamps.size) {
            val oldest = frameTimestamps[frameIndex % frameTimestamps.size]
            val elapsed = now - oldest
            if (elapsed > 0) {
                fps = frameTimestamps.size * 1000f / elapsed
            }
        }

        // Performance info text
        val infoText = String.format(
            "FPS: %.1f | 预处理: %dms | 推理: %dms | 后处理: %dms | %s",
            fps, preprocessMs, inferenceMs, postprocessMs, delegateName
        )
        canvas.drawText(infoText, 40f, 100f, infoPaint)

        // Draw detection boxes
        screenBoxes?.forEach { box ->
            canvas.drawRect(box.left, box.top, box.right, box.bottom, boxPaint)

            // Label background
            labelPaint.getTextBounds(box.label, 0, box.label.length, textBounds)
            val labelH = textBounds.height() + TEXT_PADDING
            val labelW = textBounds.width() + TEXT_PADDING
            canvas.drawRect(box.left, box.top - labelH, box.left + labelW, box.top, labelBgPaint)
            canvas.drawText(box.label, box.left + TEXT_PADDING / 2, box.top - TEXT_PADDING / 2, labelPaint)
        }
    }

    /**
     * Update detections with coordinate mapping from model space to screen space.
     *
     * All coordinate computation happens here — draw() only renders pre-computed values.
     *
     * @param boxes Detection results in normalized [0,1] coordinates (640×640 model space)
     * @param letterbox Letterbox parameters from preprocessing
     * @param rotation Rotation angle applied to the frame (0/90/180/270)
     * @param needMirror Whether front camera mirror was applied
     * @param srcWidth Original camera frame width (before rotation)
     * @param srcHeight Original camera frame height (before rotation)
     * @param delegate Current inference delegate type
     */
    fun updateDetections(
        boxes: List<DetectorUtils.Companion.BoundingBox>?,
        letterbox: YuvPreprocessor.LetterboxInfo,
        rotation: Int,
        needMirror: Boolean,
        srcWidth: Int,
        srcHeight: Int,
        delegate: MainActivity.DelegateType
    ) {
        // Read performance metrics from MainActivity (parent)
        val activity = context as? MainActivity
        preprocessMs = activity?.preprocessMs ?: 0
        inferenceMs = activity?.inferenceMs ?: 0
        postprocessMs = activity?.postprocessMs ?: 0
        delegateName = delegate.displayName

        if (boxes == null || boxes.isEmpty()) {
            screenBoxes = null
            invalidate()
            return
        }

        val viewW = width.toFloat()
        val viewH = height.toFloat()
        if (viewW == 0f || viewH == 0f) return

        // Rotated image dimensions
        val rotatedW = letterbox.rotatedWidth.toFloat()
        val rotatedH = letterbox.rotatedHeight.toFloat()

        // Fill-center scale (same logic as TextureView configureTransform)
        val fillScale = maxOf(viewW / rotatedW, viewH / rotatedH)
        val scaledW = rotatedW * fillScale
        val scaledH = rotatedH * fillScale
        val offsetX = (viewW - scaledW) / 2f
        val offsetY = (viewH - scaledH) / 2f

        val mapped = ArrayList<ScreenBox>(boxes.size)

        for (box in boxes) {
            // Step 1: Denormalize [0,1] → 640×640 pixel coordinates
            val px1 = box.x1 * YuvPreprocessor.TARGET_SIZE
            val py1 = box.y1 * YuvPreprocessor.TARGET_SIZE
            val px2 = box.x2 * YuvPreprocessor.TARGET_SIZE
            val py2 = box.y2 * YuvPreprocessor.TARGET_SIZE

            // Step 2: Remove letterbox padding → rotated image coordinates
            val scale = letterbox.scale
            val padL = letterbox.padLeft.toFloat()
            val padT = letterbox.padTop.toFloat()

            val ix1 = (px1 - padL) / scale
            val iy1 = (py1 - padT) / scale
            val ix2 = (px2 - padL) / scale
            val iy2 = (py2 - padT) / scale

            // Note: Front camera mirror is NOT reversed here.
            // Both the model and the preview display see the mirrored frame,
            // so detection coordinates are already correct for the mirrored display.

            // Step 3: Map to screen coordinates (fill-center)
            val screenLeft = ix1 * fillScale + offsetX
            val screenTop = iy1 * fillScale + offsetY
            val screenRight = ix2 * fillScale + offsetX
            val screenBottom = iy2 * fillScale + offsetY

            // Step 4: Clip to view bounds
            val clippedLeft = screenLeft.coerceIn(0f, viewW)
            val clippedTop = screenTop.coerceIn(0f, viewH)
            val clippedRight = screenRight.coerceIn(0f, viewW)
            val clippedBottom = screenBottom.coerceIn(0f, viewH)

            // Skip degenerate boxes
            if (clippedRight - clippedLeft < 2f || clippedBottom - clippedTop < 2f) continue

            val label = if (box.cls in LABELS.indices) {
                "${LABELS[box.cls]} ${String.format("%.0f%%", box.cnf * 100)}"
            } else {
                "class_${box.cls}"
            }

            mapped.add(
                ScreenBox(
                    clippedLeft, clippedTop, clippedRight, clippedBottom, label
                )
            )
        }

        screenBoxes = mapped
        invalidate()
    }

    companion object {
        private const val TEXT_PADDING = 8
        private const val TAG = "OverlayView"
        lateinit var LABELS: List<String>
    }
}
