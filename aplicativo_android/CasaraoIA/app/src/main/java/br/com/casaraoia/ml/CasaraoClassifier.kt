package br.com.casaraoia.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class CasaraoClassifier(private val context: Context) : Closeable {
    companion object {
        private const val MODEL_FILE = "casarao_ia_mobilenetv3_float16.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val IMAGE_WIDTH = 224
        private const val IMAGE_HEIGHT = 224
        private const val CHANNELS = 3
        private const val BYTES_PER_FLOAT = 4
        private const val HIGH_CONFIDENCE_THRESHOLD = 0.95f
        private const val PROBABLE_RESULT_THRESHOLD = 0.90f
    }

    private val labels = loadLabels()
    private val interpreter = Interpreter(
        loadModelFile(),
        Interpreter.Options().apply { setNumThreads(4) }
    )

    init {
        require(labels.size == 5) { "labels.txt deve possuir 5 classes." }
        validateModelTensors()
    }

    fun classify(bitmap: Bitmap): ClassificationResult {
        val prepared = resizeForModel(bitmap, IMAGE_WIDTH, IMAGE_HEIGHT)
        val input = bitmapToFloatBuffer(prepared)
        val output = Array(1) { FloatArray(labels.size) }
        val start = System.nanoTime()
        interpreter.run(input, output)
        val elapsed = (System.nanoTime() - start) / 1_000_000
        val probabilities = output[0]
        val indexes = probabilities.indices.sortedByDescending { probabilities[it] }
        val best = indexes[0]
        val second = indexes[1]
        val confidence = probabilities[best]
        return ClassificationResult(
            classIndex = best,
            label = labels[best],
            displayName = formatDisplayName(labels[best]),
            confidence = confidence,
            secondLabel = labels[second],
            secondDisplayName = formatDisplayName(labels[second]),
            secondConfidence = probabilities[second],
            inferenceTimeMs = elapsed,
            status = confidenceStatus(confidence),
            probabilities = probabilities.toList()
        )
    }

    private fun resizeForModel(bitmap: Bitmap, width: Int, height: Int): Bitmap {
        require(bitmap.width > 0 && bitmap.height > 0) { "Imagem inválida." }
        return Bitmap.createScaledBitmap(bitmap, width, height, true)
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(
            IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS * BYTES_PER_FLOAT
        ).order(ByteOrder.nativeOrder())
        val pixels = IntArray(IMAGE_WIDTH * IMAGE_HEIGHT)
        bitmap.getPixels(pixels, 0, IMAGE_WIDTH, 0, 0, IMAGE_WIDTH, IMAGE_HEIGHT)
        for (pixel in pixels) {
            buffer.putFloat(((pixel shr 16) and 0xFF).toFloat())
            buffer.putFloat(((pixel shr 8) and 0xFF).toFloat())
            buffer.putFloat((pixel and 0xFF).toFloat())
        }
        buffer.rewind()
        return buffer
    }

    private fun confidenceStatus(value: Float) = when {
        value >= HIGH_CONFIDENCE_THRESHOLD -> ConfidenceStatus.HIGH_CONFIDENCE
        value >= PROBABLE_RESULT_THRESHOLD -> ConfidenceStatus.PROBABLE_RESULT
        else -> ConfidenceStatus.INCONCLUSIVE
    }

    private fun loadLabels() = context.assets.open(LABELS_FILE).bufferedReader().useLines {
        it.map(String::trim).filter(String::isNotEmpty).toList()
    }

    private fun loadModelFile(): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        FileInputStream(fd.fileDescriptor).use { stream ->
            return stream.channel.map(
                FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength
            )
        }
    }

    private fun validateModelTensors() {
        require(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1,224,224,3)))
        require(interpreter.getOutputTensor(0).shape().contentEquals(intArrayOf(1,5)))
    }

    private fun formatDisplayName(label: String) = when(label) {
        "museu_casa_de_nhozinho" -> "Museu Casa de Nhozinho"
        "palacio_cristo_rei" -> "Palácio Cristo Rei"
        "palacio_de_la_ravardiere" -> "Palácio de La Ravardière"
        "palacio_dos_leoes" -> "Palácio dos Leões"
        "teatro_arthur_azevedo" -> "Teatro Arthur Azevedo"
        else -> label.replace('_',' ')
    }

    override fun close() = interpreter.close()
}
