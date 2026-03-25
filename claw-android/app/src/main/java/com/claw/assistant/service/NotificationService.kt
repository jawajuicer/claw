package com.claw.assistant.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.Log
import androidx.core.app.NotificationCompat
import com.claw.assistant.ClawApplication
import com.claw.assistant.MainActivity
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.sse.EventSource
import okhttp3.sse.EventSourceListener
import okhttp3.sse.EventSources
import org.json.JSONObject
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

class NotificationService : Service() {

    private var eventSource: EventSource? = null
    private val client = OkHttpClient.Builder()
        .readTimeout(0, TimeUnit.SECONDS)  // SSE needs unlimited read timeout
        .build()
    private val handler = Handler(Looper.getMainLooper())
    private var reconnectDelay = INITIAL_RECONNECT_DELAY
    private val alertNotificationId = AtomicInteger(ALERT_NOTIFICATION_BASE)

    override fun onCreate() {
        super.onCreate()
        createChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // MUST call startForeground immediately to avoid ForegroundServiceDidNotStartInTimeException
        val notification = buildForegroundNotification("Connecting...")
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        when (intent?.action) {
            ACTION_START -> startListening()
            ACTION_STOP -> {
                stopListening()
                stopSelf()
            }
            null -> {
                // START_STICKY restart after process death
                startListening()
            }
        }
        return START_STICKY
    }

    private fun startListening() {
        // Cancel any pending reconnect
        handler.removeCallbacksAndMessages(null)
        // Cancel existing connection
        eventSource?.cancel()

        val app = application as? ClawApplication
        val credentials = app?.preferencesManager?.getCredentials()
        if (credentials == null) {
            updateForegroundNotification("Not configured")
            stopSelf()
            return
        }

        val serverUrl = credentials.first.trimEnd('/')
        val apiKey = credentials.second

        updateForegroundNotification("Listening for events...")

        val request = Request.Builder()
            .url("$serverUrl/api/remote/events")
            .header("X-API-Key", apiKey)
            .build()

        val factory = EventSources.createFactory(client)
        eventSource = factory.newEventSource(request, object : EventSourceListener() {
            override fun onOpen(eventSource: EventSource, response: Response) {
                Log.d(TAG, "SSE connected")
                reconnectDelay = INITIAL_RECONNECT_DELAY  // Reset backoff on success
                updateForegroundNotification("Connected")
            }

            override fun onEvent(eventSource: EventSource, id: String?, type: String?, data: String) {
                try {
                    val json = JSONObject(data)
                    val event = json.optString("event", type ?: "")

                    when (event) {
                        "response" -> {
                            val text = json.optString("last_response", "")
                            if (text.isNotBlank()) {
                                showNotification("Claw responded", text)
                            }
                        }
                        "status" -> {
                            val state = json.optString("state", "")
                            updateForegroundNotification("Status: $state")
                        }
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Failed to parse SSE event", e)
                }
            }

            override fun onFailure(eventSource: EventSource, t: Throwable?, response: Response?) {
                Log.w(TAG, "SSE connection failed: ${t?.message}")
                updateForegroundNotification("Disconnected — retrying in ${reconnectDelay / 1000}s...")
                // OkHttp SSE does NOT auto-reconnect — schedule manual reconnect with backoff
                handler.postDelayed({ startListening() }, reconnectDelay)
                reconnectDelay = (reconnectDelay * 2).coerceAtMost(MAX_RECONNECT_DELAY)
            }

            override fun onClosed(eventSource: EventSource) {
                Log.d(TAG, "SSE closed")
                updateForegroundNotification("Disconnected")
                // Reconnect after server-initiated close
                handler.postDelayed({ startListening() }, INITIAL_RECONNECT_DELAY)
            }
        })
    }

    private fun stopListening() {
        handler.removeCallbacksAndMessages(null)
        eventSource?.cancel()
        eventSource = null
    }

    private fun createChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Claw Events",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Real-time events from The Claw"
            setShowBadge(false)
        }
        val alertChannel = NotificationChannel(
            ALERT_CHANNEL_ID,
            "Claw Alerts",
            NotificationManager.IMPORTANCE_DEFAULT
        ).apply {
            description = "Alerts and responses from The Claw"
        }
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(channel)
        manager.createNotificationChannel(alertChannel)
    }

    private fun buildForegroundNotification(text: String): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("The Claw")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateForegroundNotification(text: String) {
        val notification = buildForegroundNotification(text)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }

    private fun showNotification(title: String, text: String) {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        val notification = NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_dialog_info)
            .setContentIntent(pendingIntent)
            .setAutoCancel(true)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .build()

        // Use cycling ID to avoid unbounded notification stack (wraps after 10)
        val id = ALERT_NOTIFICATION_BASE + (alertNotificationId.getAndIncrement() % 10)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(id, notification)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        stopListening()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "NotificationService"
        const val ACTION_START = "com.claw.assistant.action.START_NOTIFICATIONS"
        const val ACTION_STOP = "com.claw.assistant.action.STOP_NOTIFICATIONS"
        const val CHANNEL_ID = "claw_events_channel"
        const val ALERT_CHANNEL_ID = "claw_alerts_channel"
        const val NOTIFICATION_ID = 3
        private const val ALERT_NOTIFICATION_BASE = 100
        private const val INITIAL_RECONNECT_DELAY = 5_000L
        private const val MAX_RECONNECT_DELAY = 60_000L

        fun start(context: Context) {
            val intent = Intent(context, NotificationService::class.java).apply {
                action = ACTION_START
            }
            context.startForegroundService(intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, NotificationService::class.java).apply {
                action = ACTION_STOP
            }
            context.startService(intent)
        }
    }
}
