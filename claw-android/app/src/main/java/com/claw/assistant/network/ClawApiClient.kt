package com.claw.assistant.network

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

data class ChatResponse(
    val content: String,
    val toolsUsed: List<String>,
    val music: MusicInfo?,
    val claudeMode: Boolean = false
)

data class MusicInfo(
    val videoId: String,
    val title: String,
    val artist: String,
    val streamUrl: String
)

data class ServerStatus(
    val status: String,
    val uptime: String,
    val activeTools: List<String>
)

class ClawApiClient {

    private var serverUrl: String = ""
    private var apiKey: String = ""
    private var isConfigured: Boolean = false

    private val httpClient = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(60, TimeUnit.SECONDS)
        .writeTimeout(30, TimeUnit.SECONDS)
        .pingInterval(30, TimeUnit.SECONDS)
        .build()

    private var webSocket: WebSocket? = null
    private var webSocketListener: WebSocketCallback? = null

    fun configure(serverUrl: String, apiKey: String) {
        this.serverUrl = serverUrl.trimEnd('/')
        this.apiKey = apiKey.trim()
        this.isConfigured = true
    }

    fun isReady(): Boolean = isConfigured

    fun getStreamUrl(videoId: String): String =
        "$serverUrl/api/remote/stream/$videoId?key=$apiKey"

    private fun requireConfigured() {
        if (!isConfigured) throw IllegalStateException("API client not configured")
    }

    private fun buildRequest(path: String): Request.Builder {
        requireConfigured()
        return Request.Builder()
            .url("$serverUrl$path")
            .header("X-API-Key", apiKey)
    }

    suspend fun ping(): Boolean = withContext(Dispatchers.IO) {
        val request = buildRequest("/api/remote/ping").get().build()
        Log.d(TAG, "Ping request to: ${request.url}")
        val response = httpClient.newCall(request).await()
        response.use { it.isSuccessful }
    }

    suspend fun chat(message: String): ChatResponse = withContext(Dispatchers.IO) {
        val json = JSONObject().put("message", message)
        val body = json.toString().toRequestBody(JSON_MEDIA_TYPE)
        val request = buildRequest("/api/remote/chat").post(body).build()
        val response = httpClient.newCall(request).await()

        response.use { resp ->
            if (!resp.isSuccessful) {
                throw IOException("Chat request failed: ${resp.code} ${resp.message}")
            }

            val responseBody = resp.body?.string()
                ?: throw IOException("Empty response body")
            val responseJson = JSONObject(responseBody)

            val toolsUsed = mutableListOf<String>()
            val toolsArray = responseJson.optJSONArray("tools_used")
            if (toolsArray != null) {
                for (i in 0 until toolsArray.length()) {
                    toolsUsed.add(toolsArray.getString(i))
                }
            }

            val musicJson = responseJson.optJSONObject("music")
            val music = if (musicJson != null && musicJson.has("video_id")) {
                MusicInfo(
                    videoId = musicJson.getString("video_id"),
                    title = musicJson.optString("title", "Unknown"),
                    artist = musicJson.optString("artist", "Unknown"),
                    streamUrl = musicJson.optString("stream_url", "")
                )
            } else null

            val claudeMode = responseJson.optBoolean("claude_mode", false)

            ChatResponse(
                content = responseJson.optString("content", ""),
                toolsUsed = toolsUsed,
                music = music,
                claudeMode = claudeMode
            )
        }
    }

    suspend fun tts(text: String): ByteArray = withContext(Dispatchers.IO) {
        val json = JSONObject().put("text", text)
        val body = json.toString().toRequestBody(JSON_MEDIA_TYPE)
        val request = buildRequest("/api/remote/tts").post(body).build()
        val response = httpClient.newCall(request).await()

        response.use { resp ->
            if (!resp.isSuccessful) {
                throw IOException("TTS request failed: ${resp.code} ${resp.message}")
            }
            resp.body?.bytes() ?: throw IOException("Empty TTS response")
        }
    }

    suspend fun status(): ServerStatus = withContext(Dispatchers.IO) {
        val request = buildRequest("/api/remote/status").get().build()
        val response = httpClient.newCall(request).await()

        response.use { resp ->
            if (!resp.isSuccessful) {
                throw IOException("Status request failed: ${resp.code} ${resp.message}")
            }

            val responseBody = resp.body?.string()
                ?: throw IOException("Empty response body")
            val json = JSONObject(responseBody)

            val tools = mutableListOf<String>()
            val toolsArray = json.optJSONArray("active_tools")
            if (toolsArray != null) {
                for (i in 0 until toolsArray.length()) {
                    tools.add(toolsArray.getString(i))
                }
            }

            ServerStatus(
                status = json.optString("status", "unknown"),
                uptime = json.optString("uptime", "unknown"),
                activeTools = tools
            )
        }
    }

    fun connectWebSocket(mode: String, callback: WebSocketCallback) {
        requireConfigured()
        disconnectWebSocket()
        webSocketListener = callback

        val wsUrl = serverUrl
            .replace("http://", "ws://")
            .replace("https://", "wss://")

        val request = Request.Builder()
            .url("$wsUrl/api/remote/audio?key=$apiKey")
            .build()

        webSocket = httpClient.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "WebSocket connected")
                val config = JSONObject().put("mode", mode)
                webSocket.send(config.toString())
                callback.onConnected()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    when (json.optString("type")) {
                        "connected" -> {
                            Log.d(TAG, "WebSocket session established: ${json.optString("device")}")
                        }
                        "transcription" -> {
                            callback.onTranscription(json.optString("text", ""))
                        }
                        "response" -> {
                            val musicJson = json.optJSONObject("music")
                            val music = if (musicJson != null && musicJson.has("video_id")) {
                                MusicInfo(
                                    videoId = musicJson.getString("video_id"),
                                    title = musicJson.optString("title", "Unknown"),
                                    artist = musicJson.optString("artist", "Unknown"),
                                    streamUrl = musicJson.optString("stream_url", "")
                                )
                            } else null
                            val claudeMode = json.optBoolean("claude_mode", false)
                            callback.onResponse(json.optString("text", ""), music, claudeMode)
                        }
                        "tts_start" -> {
                            val size = json.optInt("size", 0)
                            callback.onTtsStart(size)
                        }
                        "tts_end" -> {
                            callback.onTtsEnd()
                        }
                        "error" -> {
                            callback.onError(json.optString("message", "Unknown error"))
                        }
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Error parsing WebSocket message", e)
                }
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                callback.onTtsData(bytes.toByteArray())
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket closing: $code $reason")
                webSocket.close(1000, null)
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket closed: $code $reason")
                callback.onDisconnected()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket failure", t)
                callback.onError(t.message ?: "WebSocket connection failed")
                callback.onDisconnected()
            }
        })
    }

    fun sendAudioData(data: ByteArray) {
        webSocket?.send(data.toByteString())
    }

    fun sendWebSocketControl(type: String, extras: Map<String, Any> = emptyMap()) {
        val json = JSONObject().put("type", type)
        extras.forEach { (key, value) -> json.put(key, value) }
        webSocket?.send(json.toString())
    }

    fun disconnectWebSocket() {
        webSocket?.close(1000, "Client disconnect")
        webSocket = null
        webSocketListener = null
    }

    suspend fun chatWithRetry(message: String, maxRetries: Int = 3): ChatResponse {
        var lastException: Exception? = null
        var delay = 1000L

        for (attempt in 1..maxRetries) {
            try {
                return chat(message)
            } catch (e: IOException) {
                lastException = e
                Log.w(TAG, "Chat attempt $attempt failed (IO): ${e.message}")
            } catch (e: Exception) {
                // Check if it's a retryable HTTP error
                val msg = e.message ?: ""
                if (msg.contains("502") || msg.contains("503") || msg.contains("504")) {
                    lastException = e
                    Log.w(TAG, "Chat attempt $attempt failed (HTTP): ${e.message}")
                } else {
                    throw e  // Non-retryable
                }
            }

            if (attempt < maxRetries) {
                kotlinx.coroutines.delay(delay)
                delay *= 2
            }
        }

        throw lastException ?: IOException("Chat failed after $maxRetries attempts")
    }

    fun isWebSocketConnected(): Boolean = webSocket != null

    fun shutdown() {
        disconnectWebSocket()
        httpClient.dispatcher.executorService.shutdown()
        httpClient.connectionPool.evictAll()
    }

    interface WebSocketCallback {
        fun onConnected()
        fun onDisconnected()
        fun onTranscription(text: String)
        fun onResponse(text: String, music: MusicInfo?, claudeMode: Boolean = false)
        fun onTtsStart(size: Int)
        fun onTtsData(data: ByteArray)
        fun onTtsEnd()
        fun onError(message: String)
    }

    companion object {
        private const val TAG = "ClawApiClient"
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }
}

/**
 * Extension to make OkHttp calls suspendable.
 */
private suspend fun okhttp3.Call.await(): Response = suspendCancellableCoroutine { cont ->
    enqueue(object : okhttp3.Callback {
        override fun onResponse(call: okhttp3.Call, response: Response) {
            cont.resume(response)
        }

        override fun onFailure(call: okhttp3.Call, e: IOException) {
            if (cont.isCancelled) return
            cont.resumeWithException(e)
        }
    })

    cont.invokeOnCancellation {
        try {
            cancel()
        } catch (_: Exception) {
        }
    }
}
