package com.claw.assistant.network

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.AudioTrack
import android.media.MediaRecorder
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class AudioStreamState {
    IDLE,
    LISTENING,
    STREAMING,
    PLAYING_TTS
}

class AudioStreamManager(
    private val apiClient: ClawApiClient
) {
    private var audioRecord: AudioRecord? = null
    private var audioTrack: AudioTrack? = null
    private var captureJob: Job? = null
    private var ttsBuffer = ByteArrayOutputStream()
    private var expectedTtsSize = 0

    private val _state = MutableStateFlow(AudioStreamState.IDLE)
    val state: StateFlow<AudioStreamState> = _state.asStateFlow()

    private val _audioChunks = MutableSharedFlow<ShortArray>(extraBufferCapacity = 64)
    val audioChunks = _audioChunks.asSharedFlow()

    private val scope = CoroutineScope(Dispatchers.IO + Job())

    private var isStreamingToServer = false

    val webSocketCallback = object : ClawApiClient.WebSocketCallback {
        override fun onConnected() {
            Log.d(TAG, "WebSocket connected, ready for audio")
        }

        override fun onDisconnected() {
            Log.d(TAG, "WebSocket disconnected")
            stopStreaming()
        }

        override fun onTranscription(text: String) {
            Log.d(TAG, "Transcription: $text")
            transcriptionCallback?.invoke(text)
        }

        override fun onResponse(text: String, music: MusicInfo?, claudeMode: Boolean) {
            Log.d(TAG, "Response: $text (claudeMode=$claudeMode)")
            responseCallback?.invoke(text, music)
        }

        override fun onTtsStart(size: Int) {
            Log.d(TAG, "TTS start, expected size: $size")
            _state.value = AudioStreamState.PLAYING_TTS
            ttsBuffer.reset()
            expectedTtsSize = size
        }

        override fun onTtsData(data: ByteArray) {
            ttsBuffer.write(data)
            // Don't play here — wait for onTtsEnd
        }

        override fun onTtsEnd() {
            Log.d(TAG, "TTS end")
            if (ttsBuffer.size() > 0) {
                playTtsAudio(ttsBuffer.toByteArray())
            }
            ttsBuffer.reset()
        }

        override fun onError(message: String) {
            Log.e(TAG, "WebSocket error: $message")
            errorCallback?.invoke(message)
        }
    }

    var transcriptionCallback: ((String) -> Unit)? = null
    var responseCallback: ((String, MusicInfo?) -> Unit)? = null
    var errorCallback: ((String) -> Unit)? = null
    var ttsCompleteCallback: (() -> Unit)? = null

    fun startCapture() {
        if (audioRecord != null) return

        val bufferSize = maxOf(
            AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            ),
            CHUNK_SIZE_BYTES * 4
        )

        try {
            audioRecord = AudioRecord(
                MediaRecorder.AudioSource.VOICE_RECOGNITION,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                bufferSize
            )

            if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                audioRecord?.release()
                audioRecord = null
                return
            }

            audioRecord?.startRecording()
            _state.value = AudioStreamState.LISTENING

            captureJob = scope.launch {
                val buffer = ShortArray(CHUNK_SIZE_SAMPLES)
                while (isActive) {
                    val read = audioRecord?.read(buffer, 0, CHUNK_SIZE_SAMPLES) ?: -1
                    if (read > 0) {
                        val chunk = buffer.copyOf(read)
                        _audioChunks.emit(chunk)

                        if (isStreamingToServer) {
                            val byteBuffer = ByteBuffer.allocate(read * 2)
                                .order(ByteOrder.LITTLE_ENDIAN)
                            for (i in 0 until read) {
                                byteBuffer.putShort(chunk[i])
                            }
                            apiClient.sendAudioData(byteBuffer.array())
                        }
                    }
                }
            }
        } catch (e: SecurityException) {
            Log.e(TAG, "Microphone permission not granted", e)
        }
    }

    fun stopCapture() {
        captureJob?.cancel()
        captureJob = null
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        isStreamingToServer = false
        _state.value = AudioStreamState.IDLE
    }

    fun startStreaming() {
        if (!apiClient.isWebSocketConnected()) {
            apiClient.connectWebSocket("client_wake", webSocketCallback)
        }
        isStreamingToServer = true
        _state.value = AudioStreamState.STREAMING
        apiClient.sendWebSocketControl("wake_word")
    }

    fun startPushToTalk() {
        if (!apiClient.isWebSocketConnected()) {
            apiClient.connectWebSocket("push_to_talk", webSocketCallback)
        }
        isStreamingToServer = true
        _state.value = AudioStreamState.STREAMING
        apiClient.sendWebSocketControl("start")
    }

    fun stopStreaming() {
        isStreamingToServer = false
        apiClient.sendWebSocketControl("stop")
        _state.value = if (audioRecord != null) AudioStreamState.LISTENING else AudioStreamState.IDLE
    }

    fun stopPushToTalk() {
        stopStreaming()
    }

    private fun playTtsAudio(wavData: ByteArray) {
        scope.launch {
            try {
                // Parse WAV header sample rate instead of hardcoding
                var ttsSampleRate = SAMPLE_RATE
                val pcmData = if (wavData.size > 44 &&
                    wavData[0] == 'R'.code.toByte() &&
                    wavData[1] == 'I'.code.toByte() &&
                    wavData[2] == 'F'.code.toByte() &&
                    wavData[3] == 'F'.code.toByte()
                ) {
                    // Parse sample rate from WAV header (bytes 24-27, little-endian)
                    ttsSampleRate = (wavData[24].toInt() and 0xFF) or
                        ((wavData[25].toInt() and 0xFF) shl 8) or
                        ((wavData[26].toInt() and 0xFF) shl 16) or
                        ((wavData[27].toInt() and 0xFF) shl 24)
                    wavData.copyOfRange(44, wavData.size)
                } else {
                    wavData
                }

                val track = AudioTrack.Builder()
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setAudioFormat(
                        AudioFormat.Builder()
                            .setSampleRate(ttsSampleRate)
                            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .build()
                    )
                    .setBufferSizeInBytes(pcmData.size)
                    .setTransferMode(AudioTrack.MODE_STATIC)
                    .build()

                track.write(pcmData, 0, pcmData.size)
                track.play()

                // Wait for playback to finish
                val durationMs = (pcmData.size.toLong() * 1000) / (ttsSampleRate * 2)
                delay(durationMs + 100)

                track.stop()
                track.release()

                _state.value = if (audioRecord != null) AudioStreamState.LISTENING else AudioStreamState.IDLE
                ttsCompleteCallback?.invoke()
            } catch (e: Exception) {
                Log.e(TAG, "Error playing TTS audio", e)
                _state.value = if (audioRecord != null) AudioStreamState.LISTENING else AudioStreamState.IDLE
            }
        }
    }

    fun release() {
        stopCapture()
        apiClient.disconnectWebSocket()
        audioTrack?.release()
        audioTrack = null
    }

    companion object {
        private const val TAG = "AudioStreamManager"
        const val SAMPLE_RATE = 16000
        const val CHUNK_SIZE_SAMPLES = 1280  // 80ms at 16kHz
        const val CHUNK_SIZE_BYTES = CHUNK_SIZE_SAMPLES * 2  // 2560 bytes (int16)
    }
}
