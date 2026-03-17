package com.claw.assistant

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.os.Build
import com.claw.assistant.data.PreferencesManager
import com.claw.assistant.data.local.ClawDatabase
import com.claw.assistant.media.MusicPlayerManager
import com.claw.assistant.network.ClawApiClient
import com.claw.assistant.network.TunnelManager

class ClawApplication : Application() {

    lateinit var preferencesManager: PreferencesManager
        private set

    lateinit var apiClient: ClawApiClient
        private set

    lateinit var musicPlayerManager: MusicPlayerManager
        private set

    lateinit var tunnelManager: TunnelManager
        private set

    lateinit var database: ClawDatabase
        private set

    override fun onCreate() {
        super.onCreate()
        instance = this

        preferencesManager = PreferencesManager(this)
        apiClient = ClawApiClient()
        musicPlayerManager = MusicPlayerManager(this)
        tunnelManager = TunnelManager(this)
        database = ClawDatabase.getInstance(this)

        createNotificationChannels()

        // Configure API client if credentials already saved
        val credentials = preferencesManager.getCredentials()
        if (credentials != null) {
            apiClient.configure(credentials.first, credentials.second)
        }
    }

    private fun createNotificationChannels() {
        val wakeWordChannel = NotificationChannel(
            CHANNEL_WAKE_WORD,
            getString(R.string.notification_channel_name),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = getString(R.string.notification_channel_description)
            setShowBadge(false)
        }

        val musicChannel = NotificationChannel(
            CHANNEL_MUSIC,
            "Music Playback",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Controls for music playback"
            setShowBadge(false)
        }

        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(wakeWordChannel)
        manager.createNotificationChannel(musicChannel)
    }

    override fun onTerminate() {
        tunnelManager.disconnect()
        musicPlayerManager.release()
        apiClient.shutdown()
        super.onTerminate()
    }

    companion object {
        const val CHANNEL_WAKE_WORD = "wake_word_channel"
        const val CHANNEL_MUSIC = "music_channel"
        const val WAKE_WORD_NOTIFICATION_ID = 1
        const val MUSIC_NOTIFICATION_ID = 2

        @Volatile
        private var instance: ClawApplication? = null

        fun getInstance(): ClawApplication =
            instance ?: throw IllegalStateException("ClawApplication not initialized")
    }
}
