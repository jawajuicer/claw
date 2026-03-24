package com.claw.assistant.service

import android.os.Bundle
import android.service.voice.VoiceInteractionService
import android.service.voice.VoiceInteractionSession
import android.util.Log

class ClawVoiceInteractionService : VoiceInteractionService() {
    companion object {
        private const val TAG = "ClawVIS"
    }

    override fun onReady() {
        super.onReady()
        Log.d(TAG, "Claw VoiceInteractionService ready")
    }

    override fun onLaunchVoiceAssistFromKeyguard() {
        showSession(null, VoiceInteractionSession.SHOW_WITH_ASSIST)
    }

    override fun onShutdown() {
        Log.d(TAG, "Claw VoiceInteractionService shutdown")
        super.onShutdown()
    }
}
