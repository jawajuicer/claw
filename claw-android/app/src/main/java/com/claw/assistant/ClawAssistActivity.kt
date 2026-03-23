package com.claw.assistant

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.claw.assistant.network.AudioStreamManager
import com.claw.assistant.network.AudioStreamState
import com.claw.assistant.network.ClawApiClient
import com.claw.assistant.network.MusicInfo
import com.claw.assistant.ui.theme.ClawAccent
import com.claw.assistant.ui.theme.ClawTextPrimary
import com.claw.assistant.ui.theme.ClawTextSecondary
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

/**
 * Transparent overlay activity that handles ACTION_ASSIST and ACTION_VOICE_COMMAND.
 * When the user selects Claw as the default assistant, pressing the assistant button
 * (home long-press, swipe gesture, or steering wheel button in Android Auto) launches this.
 *
 * Flow: launch → listen → transcribe → respond with TTS → auto-dismiss or continue.
 */
class ClawAssistActivity : ComponentActivity() {

    private var audioStreamManager: AudioStreamManager? = null
    private val scope = CoroutineScope(Dispatchers.Main + Job())
    private var autoDismissJob: Job? = null

    // UI state
    private val transcription = mutableStateOf("")
    private val responseText = mutableStateOf("")
    private val currentState = mutableStateOf(AssistState.INITIALIZING)
    private val isClaudeMode = mutableStateOf(false)

    private enum class AssistState {
        INITIALIZING, LISTENING, PROCESSING, RESPONDING, ERROR
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val app = application as ClawApplication

        // Check if configured
        if (!app.apiClient.isReady()) {
            val creds = app.preferencesManager.getCredentials()
            if (creds != null) {
                app.apiClient.configure(creds.first, creds.second)
            } else {
                currentState.value = AssistState.ERROR
                responseText.value = "Claw is not set up yet. Open the app to connect."
                setContent { AssistOverlay() }
                scope.launch {
                    delay(3000)
                    finish()
                }
                return
            }
        }

        // Check microphone permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            currentState.value = AssistState.ERROR
            responseText.value = "Microphone permission required. Open the Claw app to grant it."
            setContent { AssistOverlay() }
            scope.launch {
                delay(3000)
                finish()
            }
            return
        }

        // Initialize audio and start listening
        audioStreamManager = AudioStreamManager(app.apiClient).apply {
            transcriptionCallback = { text ->
                transcription.value = text
                currentState.value = AssistState.PROCESSING
            }
            responseCallback = { text, music ->
                responseText.value = text
                currentState.value = AssistState.RESPONDING
                if (music != null) {
                    app.musicPlayerManager.playFromServer(music)
                }
                scheduleAutoDismiss()
            }
            ttsCompleteCallback = {
                scheduleAutoDismiss()
            }
            errorCallback = { error ->
                Log.e(TAG, "Assist error: $error")
                if (currentState.value != AssistState.RESPONDING) {
                    responseText.value = "Something went wrong. Try again."
                    currentState.value = AssistState.ERROR
                    scheduleAutoDismiss()
                }
            }
        }

        setContent { AssistOverlay() }

        // Start voice capture immediately
        startListening()
    }

    private fun startListening() {
        val manager = audioStreamManager ?: return
        currentState.value = AssistState.LISTENING
        transcription.value = ""
        responseText.value = ""
        autoDismissJob?.cancel()

        manager.startCapture()
        manager.startPushToTalk()
    }

    private fun stopListening() {
        audioStreamManager?.stopPushToTalk()
    }

    private fun scheduleAutoDismiss() {
        autoDismissJob?.cancel()
        autoDismissJob = scope.launch {
            delay(5000)
            finish()
        }
    }

    @Composable
    private fun AssistOverlay() {
        val state by currentState
        val transcriptionText by transcription
        val response by responseText
        val claudeMode by isClaudeMode

        DisposableEffect(Unit) {
            onDispose {
                audioStreamManager?.release()
            }
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(Color.Black.copy(alpha = 0.85f))
                .clickable(
                    indication = null,
                    interactionSource = remember { MutableInteractionSource() }
                ) {
                    // Tap outside to dismiss
                    finish()
                },
            contentAlignment = Alignment.Center
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth(0.85f)
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                // Pulsing mic icon when listening
                if (state == AssistState.LISTENING) {
                    PulsingMicIcon()
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Listening...",
                        style = MaterialTheme.typography.titleMedium,
                        color = ClawAccent,
                        textAlign = TextAlign.Center
                    )
                } else if (state == AssistState.PROCESSING) {
                    Icon(
                        imageVector = Icons.Default.Mic,
                        contentDescription = null,
                        tint = ClawTextSecondary,
                        modifier = Modifier.size(48.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Processing...",
                        style = MaterialTheme.typography.titleMedium,
                        color = ClawTextSecondary,
                        textAlign = TextAlign.Center
                    )
                } else if (state == AssistState.ERROR) {
                    Icon(
                        imageVector = Icons.Default.MicOff,
                        contentDescription = null,
                        tint = Color(0xFFF85149),
                        modifier = Modifier.size(48.dp)
                    )
                }

                // Transcription text
                AnimatedVisibility(
                    visible = transcriptionText.isNotBlank(),
                    enter = fadeIn(),
                    exit = fadeOut()
                ) {
                    Text(
                        text = "\"$transcriptionText\"",
                        style = MaterialTheme.typography.bodyLarge,
                        color = ClawTextPrimary,
                        textAlign = TextAlign.Center,
                        modifier = Modifier.padding(top = 12.dp)
                    )
                }

                Spacer(modifier = Modifier.height(16.dp))

                // Response
                AnimatedVisibility(
                    visible = response.isNotBlank(),
                    enter = fadeIn(),
                    exit = fadeOut()
                ) {
                    Surface(
                        shape = RoundedCornerShape(16.dp),
                        color = if (claudeMode) Color(0xFF1E1E3F) else Color(0xFF161B22),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(
                            modifier = Modifier
                                .padding(16.dp)
                                .verticalScroll(rememberScrollState())
                        ) {
                            if (claudeMode) {
                                Text(
                                    text = "Claude Code",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = Color(0xFFB388FF),
                                    modifier = Modifier.padding(bottom = 4.dp)
                                )
                            } else {
                                Text(
                                    text = "The Claw",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ClawTextSecondary,
                                    modifier = Modifier.padding(bottom = 4.dp)
                                )
                            }
                            Text(
                                text = response,
                                style = MaterialTheme.typography.bodyMedium,
                                color = ClawTextPrimary,
                                fontFamily = if (response.contains("```")) FontFamily.Monospace
                                    else FontFamily.SansSerif
                            )
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun PulsingMicIcon() {
        val infiniteTransition = rememberInfiniteTransition(label = "pulse")
        val scale by infiniteTransition.animateFloat(
            initialValue = 1f,
            targetValue = 1.2f,
            animationSpec = infiniteRepeatable(
                animation = tween(600),
                repeatMode = RepeatMode.Reverse
            ),
            label = "micPulse"
        )

        Box(
            contentAlignment = Alignment.Center,
            modifier = Modifier.size(80.dp)
        ) {
            // Glow ring
            Box(
                modifier = Modifier
                    .size(80.dp)
                    .scale(scale)
                    .background(
                        ClawAccent.copy(alpha = 0.2f),
                        CircleShape
                    )
            )
            // Mic icon
            Icon(
                imageVector = Icons.Default.Mic,
                contentDescription = "Listening",
                tint = ClawAccent,
                modifier = Modifier.size(40.dp)
            )
        }
    }

    override fun onDestroy() {
        autoDismissJob?.cancel()
        audioStreamManager?.release()
        audioStreamManager = null
        super.onDestroy()
    }

    companion object {
        private const val TAG = "ClawAssistActivity"
    }
}
