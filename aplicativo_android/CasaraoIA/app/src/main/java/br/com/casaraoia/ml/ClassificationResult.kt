package br.com.casaraoia.ml

data class ClassificationResult(
    val classIndex: Int,
    val label: String,
    val displayName: String,
    val confidence: Float,
    val secondLabel: String,
    val secondDisplayName: String,
    val secondConfidence: Float,
    val inferenceTimeMs: Long,
    val status: ConfidenceStatus,
    val probabilities: List<Float>
)

enum class ConfidenceStatus {
    HIGH_CONFIDENCE,
    PROBABLE_RESULT,
    INCONCLUSIVE
}
