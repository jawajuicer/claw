package com.claw.assistant.service

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.os.Build
import android.util.Log
import com.claw.assistant.ClawApplication

class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != Intent.ACTION_BOOT_COMPLETED) return

        val app = context.applicationContext as ClawApplication
        if (!app.preferencesManager.isAutoStartService()) {
            Log.d(TAG, "Auto-start disabled, not starting wake word service")
            return
        }

        if (!app.preferencesManager.isConfigured()) {
            Log.d(TAG, "Not configured, not starting wake word service")
            return
        }

        Log.d(TAG, "Boot completed, starting wake word service")
        val serviceIntent = Intent(context, WakeWordService::class.java).apply {
            action = WakeWordService.ACTION_START
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            context.startForegroundService(serviceIntent)
        } else {
            context.startService(serviceIntent)
        }
    }

    companion object {
        private const val TAG = "BootReceiver"
    }
}
