package com.claw.assistant.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.claw.assistant.ClawApplication
import com.claw.assistant.MainActivity
import com.claw.assistant.R
import com.claw.assistant.network.AudioStreamManager
import com.claw.assistant.network.AudioStreamState
import com.claw.assistant.network.MusicInfo
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class WakeWordService : Service() {

    private lateinit var wakeWordEngine: WakeWordEngine
    private lateinit var audioStreamManager: AudioStreamManager

    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var wakeWordJob: Job? = null
    private var isRunning = false

    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "WakeWordService created")

        val app = application as ClawApplication
        wakeWordEngine = WakeWordEngine(this)
        audioStreamManager = AudioStreamManager(app.apiClient)

        audioStreamManager.responseCallback = { text, music ->
            Log.d(TAG, "Got response: $text")
            if (music != null) {
                handleMusicResponse(music)
            }
        }

        audioStreamManager.ttsCompleteCallback = {
            Log.d(TAG, "TTS playback complete, resuming wake word listening")
            wakeWordEngine.reset()
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startListening()
            ACTION_STOP -> stopListening()
            ACTION_PUSH_TO_TALK_START -> startPushToTalk()
            ACTION_PUSH_TO_TALK_STOP -> stopPushToTalk()
            else -> startListening()
        }
        return START_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startListening() {
        if (isRunning) return
        isRunning = true

        val notification = createNotification(
            getString(R.string.notification_text)
        )

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                ClawApplication.WAKE_WORD_NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(ClawApplication.WAKE_WORD_NOTIFICATION_ID, notification)
        }

        // Initialize wake word engine
        val modelsReady = wakeWordEngine.initialize()
        if (!modelsReady) {
            Log.w(TAG, "Wake word models not available, running in push-to-talk only mode")
            updateNotification(getString(R.string.wake_word_no_models))
        }

        // Start audio capture
        audioStreamManager.startCapture()

        // Start wake word detection loop
        if (modelsReady) {
            wakeWordJob = serviceScope.launch {
                audioStreamManager.audioChunks.collectLatest { chunk ->
                    // Only process for wake word when in LISTENING state
                    if (audioStreamManager.state.value == AudioStreamState.LISTENING) {
                        val detected = wakeWordEngine.processAudio(chunk)
                        if (detected) {
                            onWakeWordDetected()
                        }
                    }
                }
            }
        }

        Log.d(TAG, "Wake word service started, models available: $modelsReady")
    }

    private fun stopListening() {
        isRunning = false
        wakeWordJob?.cancel()
        wakeWordJob = null
        audioStreamManager.release()
        wakeWordEngine.release()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
        Log.d(TAG, "Wake word service stopped")
    }

    private fun onWakeWordDetected() {
        Log.d(TAG, "Wake word detected!")
        updateNotification(getString(R.string.notification_text_processing))

        // Start streaming audio to server
        audioStreamManager.startStreaming()

        // Monitor state to update notification when done
        serviceScope.launch {
            audioStreamManager.state.collectLatest { state ->
                when (state) {
                    AudioStreamState.LISTENING -> {
                        updateNotification(getString(R.string.notification_text))
                    }
                    AudioStreamState.STREAMING -> {
                        updateNotification(getString(R.string.recording))
                    }
                    AudioStreamState.PLAYING_TTS -> {
                        updateNotification(getString(R.string.processing))
                    }
                    AudioStreamState.IDLE -> { /* Service not capturing */ }
                }
            }
        }

        // Auto-stop streaming after silence timeout (server handles VAD)
        // The server will send back a response which triggers stopStreaming via callback
    }

    private fun startPushToTalk() {
        if (!isRunning) {
            startListening()
        }
        updateNotification(getString(R.string.recording))
        audioStreamManager.startPushToTalk()
    }

    private fun stopPushToTalk() {
        audioStreamManager.stopPushToTalk()
        updateNotification(getString(R.string.notification_text))
    }

    private fun handleMusicResponse(music: MusicInfo) {
        val app = application as ClawApplication
        app.musicPlayerManager.playFromServer(music)
    }

    private fun createNotification(contentText: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, ClawApplication.CHANNEL_WAKE_WORD)
            .setContentTitle(getString(R.string.notification_title))
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setSilent(true)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }

    private fun updateNotification(contentText: String) {
        val notification = createNotification(contentText)
        val manager = getSystemService(android.app.NotificationManager::class.java)
        manager.notify(ClawApplication.WAKE_WORD_NOTIFICATION_ID, notification)
    }

    override fun onDestroy() {
        isRunning = false
        wakeWordJob?.cancel()
        audioStreamManager.release()
        wakeWordEngine.release()
        serviceScope.cancel()
        super.onDestroy()
        Log.d(TAG, "WakeWordService destroyed")
    }

    companion object {
        private const val TAG = "WakeWordService"
        const val ACTION_START = "com.claw.assistant.action.START_LISTENING"
        const val ACTION_STOP = "com.claw.assistant.action.STOP_LISTENING"
        const val ACTION_PUSH_TO_TALK_START = "com.claw.assistant.action.PUSH_TO_TALK_START"
        const val ACTION_PUSH_TO_TALK_STOP = "com.claw.assistant.action.PUSH_TO_TALK_STOP"

        fun isServiceRunning(context: android.content.Context): Boolean {
            val manager = context.getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
            @Suppress("DEPRECATION")
            for (service in manager.getRunningServices(Int.MAX_VALUE)) {
                if (WakeWordService::class.java.name == service.service.className) {
                    return true
                }
            }
            return false
        }
    }
}
