package com.claw.assistant.service

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import java.util.Collections

/**
 * On-device wake word detection using openWakeWord ONNX models.
 *
 * The pipeline has three stages:
 * 1. melspectrogram.onnx - Converts raw 16kHz audio to mel-spectrogram features
 * 2. embedding_model.onnx - Converts mel-spectrograms to audio embeddings
 * 3. hey_claw.onnx - Classifies embeddings to detect the "Hey Claw" wake word
 *
 * If models are not present, the engine reports itself as unavailable and all
 * detection calls return false.
 */
class WakeWordEngine(private val context: Context) {

    private var ortEnvironment: OrtEnvironment? = null
    private var melSession: OrtSession? = null
    private var embeddingSession: OrtSession? = null
    private var classifierSession: OrtSession? = null

    private var isInitialized = false
    private var _modelsAvailable = false

    // Ring buffer for accumulating audio samples for the mel-spectrogram model.
    // openWakeWord's melspectrogram model expects 480 samples (30ms) per frame
    // but we accumulate 1280 samples (80ms) and process multiple frames.
    private val audioBuffer = FloatArray(AUDIO_BUFFER_SIZE)
    private var audioBufferPos = 0

    // Accumulator for mel-spectrogram frames before embedding
    private val melAccumulator = mutableListOf<FloatArray>()

    // Sliding window of embeddings for the classifier
    private val embeddingWindow = ArrayDeque<FloatArray>(EMBEDDING_WINDOW_SIZE)

    // Smoothing: rolling average of classifier scores
    private val scoreHistory = ArrayDeque<Float>(SCORE_HISTORY_SIZE)

    val modelsAvailable: Boolean get() = _modelsAvailable

    fun initialize(): Boolean {
        if (isInitialized) return _modelsAvailable

        try {
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Check which models are available
            val availableModels = checkAvailableModels()
            if (!availableModels.containsAll(REQUIRED_MODELS)) {
                val missing = REQUIRED_MODELS - availableModels
                Log.w(TAG, "Wake word models not found: $missing. Detection disabled.")
                _modelsAvailable = false
                isInitialized = true
                return false
            }

            val sessionOptions = OrtSession.SessionOptions().apply {
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                setIntraOpNumThreads(2)
                // NNAPI is attempted for battery efficiency but may not be available on all devices.
                try {
                    addNnapi()
                    Log.d(TAG, "NNAPI execution provider enabled")
                } catch (e: Exception) {
                    Log.d(TAG, "NNAPI not available, using CPU: ${e.message}")
                }
            }

            melSession = ortEnvironment!!.createSession(
                loadModelBytes("melspectrogram.onnx"),
                sessionOptions
            )
            embeddingSession = ortEnvironment!!.createSession(
                loadModelBytes("embedding_model.onnx"),
                sessionOptions
            )
            classifierSession = ortEnvironment!!.createSession(
                loadModelBytes("hey_claw.onnx"),
                sessionOptions
            )

            _modelsAvailable = true
            isInitialized = true
            Log.d(TAG, "Wake word engine initialized successfully")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize wake word engine", e)
            _modelsAvailable = false
            isInitialized = true
            release()
            return false
        }
    }

    private fun checkAvailableModels(): Set<String> {
        val available = mutableSetOf<String>()
        try {
            val assetFiles = context.assets.list("models") ?: emptyArray()
            for (file in assetFiles) {
                if (file.endsWith(".onnx")) {
                    available.add(file)
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Could not list model assets", e)
        }
        return available
    }

    private fun loadModelBytes(name: String): ByteArray {
        return context.assets.open("models/$name").use { it.readBytes() }
    }

    /**
     * Process a chunk of 16kHz int16 PCM audio samples.
     * Returns true if the wake word was detected.
     *
     * @param samples Audio samples as short array (int16 PCM, 16kHz, mono)
     */
    fun processAudio(samples: ShortArray): Boolean {
        if (!_modelsAvailable) return false

        // Convert int16 to float32 normalized to [-1, 1]
        for (sample in samples) {
            audioBuffer[audioBufferPos] = sample.toFloat() / 32768f
            audioBufferPos++

            // When we have enough samples for a mel frame, process it
            if (audioBufferPos >= MEL_FRAME_SAMPLES) {
                val frame = audioBuffer.copyOfRange(0, MEL_FRAME_SAMPLES)

                // Shift buffer
                val remaining = audioBufferPos - MEL_FRAME_SAMPLES
                System.arraycopy(audioBuffer, MEL_FRAME_SAMPLES, audioBuffer, 0, remaining)
                audioBufferPos = remaining

                val melFeatures = computeMelSpectrogram(frame)
                if (melFeatures != null) {
                    melAccumulator.add(melFeatures)
                }

                // When we have enough mel frames for an embedding, compute it
                if (melAccumulator.size >= MEL_FRAMES_PER_EMBEDDING) {
                    val melBatch = melAccumulator.toList()
                    melAccumulator.clear()

                    val embedding = computeEmbedding(melBatch)
                    if (embedding != null) {
                        embeddingWindow.addLast(embedding)
                        if (embeddingWindow.size > EMBEDDING_WINDOW_SIZE) {
                            embeddingWindow.removeFirst()
                        }

                        // Run classifier when we have enough embeddings
                        if (embeddingWindow.size >= MIN_EMBEDDINGS_FOR_CLASSIFICATION) {
                            val score = classify()
                            if (score != null) {
                                scoreHistory.addLast(score)
                                if (scoreHistory.size > SCORE_HISTORY_SIZE) {
                                    scoreHistory.removeFirst()
                                }

                                val avgScore = scoreHistory.average().toFloat()
                                if (avgScore > DETECTION_THRESHOLD) {
                                    Log.d(TAG, "Wake word detected! Score: $avgScore")
                                    reset()
                                    return true
                                }
                            }
                        }
                    }
                }
            }
        }
        return false
    }

    private fun computeMelSpectrogram(frame: FloatArray): FloatArray? {
        val env = ortEnvironment ?: return null
        val session = melSession ?: return null

        return try {
            val inputShape = longArrayOf(1, frame.size.toLong())
            val inputBuffer = FloatBuffer.wrap(frame)
            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)

            val results = session.run(Collections.singletonMap("input", inputTensor))
            val outputTensor = results[0] as OnnxTensor
            val output = outputTensor.floatBuffer
            val features = FloatArray(output.remaining())
            output.get(features)

            inputTensor.close()
            results.close()

            features
        } catch (e: Exception) {
            Log.e(TAG, "Mel spectrogram computation failed", e)
            null
        }
    }

    private fun computeEmbedding(melFrames: List<FloatArray>): FloatArray? {
        val env = ortEnvironment ?: return null
        val session = embeddingSession ?: return null

        return try {
            // Flatten mel frames into a single input array
            val totalSize = melFrames.sumOf { it.size }
            val flatInput = FloatArray(totalSize)
            var offset = 0
            for (frame in melFrames) {
                System.arraycopy(frame, 0, flatInput, offset, frame.size)
                offset += frame.size
            }

            val inputShape = longArrayOf(1, melFrames.size.toLong(), melFrames[0].size.toLong())
            val inputBuffer = FloatBuffer.wrap(flatInput)
            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)

            val results = session.run(Collections.singletonMap("input", inputTensor))
            val outputTensor = results[0] as OnnxTensor
            val output = outputTensor.floatBuffer
            val embedding = FloatArray(output.remaining())
            output.get(embedding)

            inputTensor.close()
            results.close()

            embedding
        } catch (e: Exception) {
            Log.e(TAG, "Embedding computation failed", e)
            null
        }
    }

    private fun classify(): Float? {
        val env = ortEnvironment ?: return null
        val session = classifierSession ?: return null

        return try {
            // Use the most recent embeddings
            val embeddings = embeddingWindow.toList()
            val embSize = embeddings[0].size
            val totalSize = embeddings.size * embSize
            val flatInput = FloatArray(totalSize)
            var offset = 0
            for (emb in embeddings) {
                System.arraycopy(emb, 0, flatInput, offset, emb.size)
                offset += emb.size
            }

            val inputShape = longArrayOf(1, embeddings.size.toLong(), embSize.toLong())
            val inputBuffer = FloatBuffer.wrap(flatInput)
            val inputTensor = OnnxTensor.createTensor(env, inputBuffer, inputShape)

            val results = session.run(Collections.singletonMap("input", inputTensor))
            val outputTensor = results[0] as OnnxTensor
            val output = outputTensor.floatBuffer
            val score = output.get(0)

            inputTensor.close()
            results.close()

            score
        } catch (e: Exception) {
            Log.e(TAG, "Classification failed", e)
            null
        }
    }

    /**
     * Reset all internal state. Called after a detection or when restarting.
     */
    fun reset() {
        audioBufferPos = 0
        audioBuffer.fill(0f)
        melAccumulator.clear()
        embeddingWindow.clear()
        scoreHistory.clear()
    }

    fun release() {
        try {
            melSession?.close()
            embeddingSession?.close()
            classifierSession?.close()
            ortEnvironment?.close()
        } catch (e: Exception) {
            Log.w(TAG, "Error releasing ONNX sessions", e)
        }
        melSession = null
        embeddingSession = null
        classifierSession = null
        ortEnvironment = null
        isInitialized = false
        _modelsAvailable = false
    }

    companion object {
        private const val TAG = "WakeWordEngine"

        // openWakeWord mel-spectrogram model expects 480 samples per frame (30ms at 16kHz)
        const val MEL_FRAME_SAMPLES = 480

        // Number of mel frames needed before computing an embedding
        const val MEL_FRAMES_PER_EMBEDDING = 76

        // Size of the embedding sliding window for the classifier
        const val EMBEDDING_WINDOW_SIZE = 16

        // Minimum embeddings needed before running the classifier
        const val MIN_EMBEDDINGS_FOR_CLASSIFICATION = 5

        // Audio buffer large enough to hold samples between processing
        const val AUDIO_BUFFER_SIZE = 4096

        // Rolling average window for detection smoothing
        const val SCORE_HISTORY_SIZE = 5

        // Detection threshold (0-1). Tuned for low false-positive rate.
        const val DETECTION_THRESHOLD = 0.5f

        val REQUIRED_MODELS = setOf(
            "melspectrogram.onnx",
            "embedding_model.onnx",
            "hey_claw.onnx"
        )
    }
}
