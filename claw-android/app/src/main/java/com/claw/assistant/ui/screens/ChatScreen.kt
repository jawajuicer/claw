package com.claw.assistant.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Logout
import androidx.compose.material.icons.automirrored.filled.Send
import androidx.compose.material.icons.filled.Circle
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Code
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material.icons.filled.SystemUpdate
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.platform.LocalContext
import com.claw.assistant.R
import com.claw.assistant.data.local.ClawDatabase
import com.claw.assistant.data.local.MessageEntity
import com.claw.assistant.data.local.MessageQueueWorker
import com.claw.assistant.media.MusicPlayerManager
import com.claw.assistant.network.AppUpdate
import com.claw.assistant.network.AudioStreamState
import com.claw.assistant.network.ClawApiClient
import com.claw.assistant.ui.components.ChatMessage
import com.claw.assistant.ui.components.MessageBubble
import com.claw.assistant.ui.components.MiniPlayer
import com.claw.assistant.ui.components.TypingIndicator
import com.claw.assistant.ui.components.VoiceButton
import com.claw.assistant.ui.theme.ClawAccent
import com.claw.assistant.ui.theme.ClawBackground
import com.claw.assistant.ui.theme.ClawBorder
import com.claw.assistant.ui.theme.ClawError
import com.claw.assistant.ui.theme.ClawSuccess
import com.claw.assistant.ui.theme.ClawSurface
import com.claw.assistant.ui.theme.ClawSurfaceVariant
import com.claw.assistant.ui.theme.ClawTextPrimary
import com.claw.assistant.ui.theme.ClawTextSecondary
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import java.util.UUID

private const val TOOLS_DELIMITER = "|||"

fun ChatMessage.toEntity() = MessageEntity(
    id = id,
    content = content,
    isUser = isUser,
    toolsUsed = toolsUsed.joinToString(TOOLS_DELIMITER),
    timestamp = timestamp,
    status = status,
    isClaudeCode = isClaudeCode
)

fun MessageEntity.toChatMessage() = ChatMessage(
    id = id,
    content = content,
    isUser = isUser,
    toolsUsed = if (toolsUsed.isBlank()) emptyList()
        else toolsUsed.split(TOOLS_DELIMITER).filter { it.isNotBlank() },
    timestamp = timestamp,
    status = status,
    isClaudeCode = isClaudeCode
)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(
    apiClient: ClawApiClient,
    database: ClawDatabase? = null,
    musicPlayerManager: MusicPlayerManager,
    audioStreamState: StateFlow<AudioStreamState>,
    wakeWordAvailable: Boolean,
    isServiceRunning: Boolean,
    updateAvailable: AppUpdate? = null,
    onUpdateClick: () -> Unit = {},
    onToggleService: (Boolean) -> Unit,
    onPushToTalkStart: () -> Unit,
    onPushToTalkStop: () -> Unit,
    onDisconnect: () -> Unit,
    modifier: Modifier = Modifier
) {
    val messages = remember { mutableStateListOf<ChatMessage>() }
    var inputText by remember { mutableStateOf("") }
    var isLoading by remember { mutableStateOf(false) }
    var showSettings by remember { mutableStateOf(false) }
    var isClaudeMode by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()
    val focusManager = LocalFocusManager.current
    val context = LocalContext.current

    val streamState by audioStreamState.collectAsState()
    val nowPlaying by musicPlayerManager.nowPlaying.collectAsState()
    val isPlaying by musicPlayerManager.isPlaying.collectAsState()

    // Load messages from Room DB on first composition
    LaunchedEffect(Unit) {
        database?.messageDao()?.getAll()?.map { it.toChatMessage() }?.let { saved ->
            messages.addAll(saved)
        }
    }

    // Scroll to bottom when new messages arrive
    LaunchedEffect(messages.size) {
        if (messages.isNotEmpty()) {
            listState.animateScrollToItem(messages.size - 1)
        }
    }

    fun sendMessage(retryId: String? = null) {
        // Block concurrent sends (applies to both new messages and retries)
        if (isLoading) return

        val text = if (retryId != null) {
            // Find the original user message to retry
            messages.find { it.id == retryId }?.content ?: return
        } else {
            inputText.trim().also { if (it.isBlank()) return }
        }

        if (retryId == null) {
            inputText = ""
            focusManager.clearFocus()
            val userMessage = ChatMessage(
                id = UUID.randomUUID().toString(),
                content = text,
                isUser = true
            )
            messages.add(userMessage)
            // Save user message to DB
            scope.launch {
                database?.messageDao()?.insert(userMessage.toEntity())
            }
        }

        // Remove the error message associated with this retry (the one right after the user msg)
        if (retryId != null) {
            val userIdx = messages.indexOfFirst { it.id == retryId }
            val errorIdx = if (userIdx >= 0 && userIdx + 1 < messages.size) userIdx + 1 else -1
            if (errorIdx >= 0 && messages[errorIdx].status == "failed") {
                messages.removeAt(errorIdx)
            }
        }

        isLoading = true

        scope.launch {
            try {
                val response = apiClient.chatWithRetry(text)
                isClaudeMode = response.claudeMode
                val assistantMessage = ChatMessage(
                    id = UUID.randomUUID().toString(),
                    content = response.content,
                    isUser = false,
                    toolsUsed = response.toolsUsed,
                    isClaudeCode = response.claudeMode
                )
                messages.add(assistantMessage)
                // Save assistant message to DB
                database?.messageDao()?.insert(assistantMessage.toEntity())

                // Mark the user message as sent (clears "pending" status after manual retry)
                if (retryId != null) {
                    database?.messageDao()?.updateStatus(retryId, "sent")
                    val userIdx = messages.indexOfFirst { it.id == retryId }
                    if (userIdx >= 0) {
                        messages[userIdx] = messages[userIdx].copy(status = "sent")
                    }
                }

                // Handle music response
                if (response.music != null) {
                    musicPlayerManager.playFromServer(response.music)
                }
            } catch (e: Exception) {
                // Save user message as pending for offline retry
                val pendingUserMsg = messages.lastOrNull { it.isUser && it.content == text }
                if (pendingUserMsg != null) {
                    val pendingEntity = pendingUserMsg.copy(status = "pending").toEntity()
                    database?.messageDao()?.insert(pendingEntity)
                    // Update in-memory status
                    val idx = messages.indexOf(pendingUserMsg)
                    if (idx >= 0) {
                        messages[idx] = pendingUserMsg.copy(status = "pending")
                    }
                    MessageQueueWorker.enqueue(context)
                }

                val errorMessage = ChatMessage(
                    id = UUID.randomUUID().toString(),
                    content = "Failed to send: ${e.message ?: "Connection error"}",
                    isUser = false,
                    status = "failed"
                )
                messages.add(errorMessage)
            } finally {
                isLoading = false
            }
        }
    }

    // Settings dialog
    if (showSettings) {
        SettingsDialog(
            isServiceRunning = isServiceRunning,
            wakeWordAvailable = wakeWordAvailable,
            onToggleService = onToggleService,
            onDisconnect = {
                showSettings = false
                onDisconnect()
            },
            onDismiss = { showSettings = false }
        )
    }

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(ClawBackground)
    ) {
        // Top bar
        TopAppBar(
            title = {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    if (isClaudeMode) {
                        Icon(
                            imageVector = Icons.Default.Code,
                            contentDescription = null,
                            modifier = Modifier.size(20.dp),
                            tint = ClawAccent
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                    }
                    Text(
                        text = if (isClaudeMode) "Claude Code" else "The Claw",
                        style = MaterialTheme.typography.titleLarge,
                        color = ClawTextPrimary
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    // Status indicator
                    Icon(
                        imageVector = Icons.Default.Circle,
                        contentDescription = if (isServiceRunning) "Service running" else "Service stopped",
                        modifier = Modifier.size(10.dp),
                        tint = if (isServiceRunning) ClawSuccess else ClawTextSecondary
                    )
                }
            },
            actions = {
                IconButton(onClick = { showSettings = true }) {
                    Icon(
                        imageVector = Icons.Default.Settings,
                        contentDescription = stringResource(R.string.settings_label),
                        tint = ClawTextSecondary
                    )
                }
            },
            colors = TopAppBarDefaults.topAppBarColors(
                containerColor = ClawSurface
            )
        )

        // Update banner
        if (updateAvailable != null) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(ClawAccent.copy(alpha = 0.15f))
                    .clickable { onUpdateClick() }
                    .padding(12.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    Icons.Default.SystemUpdate,
                    contentDescription = null,
                    tint = ClawAccent,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Update available: v${updateAvailable.versionName}",
                        style = MaterialTheme.typography.bodyMedium,
                        color = ClawTextPrimary
                    )
                    if (updateAvailable.changelog.isNotBlank()) {
                        Text(
                            text = updateAvailable.changelog,
                            style = MaterialTheme.typography.bodySmall,
                            color = ClawTextSecondary
                        )
                    }
                }
                Text(
                    text = "Install",
                    color = ClawAccent,
                    style = MaterialTheme.typography.labelLarge
                )
            }
        }

        // Claude Code mode banner
        if (isClaudeMode) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(ClawAccent.copy(alpha = 0.10f))
                    .padding(horizontal = 12.dp, vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Icon(
                    imageVector = Icons.Default.Code,
                    contentDescription = null,
                    tint = ClawAccent,
                    modifier = Modifier.size(18.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Claude Code Mode",
                    style = MaterialTheme.typography.bodyMedium,
                    color = ClawTextPrimary,
                    modifier = Modifier.weight(1f)
                )
                TextButton(
                    onClick = {
                        inputText = "exit claude"
                        sendMessage()
                    }
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Exit Claude Code mode",
                        tint = ClawTextSecondary,
                        modifier = Modifier.size(16.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "Exit",
                        color = ClawTextSecondary,
                        style = MaterialTheme.typography.labelMedium
                    )
                }
            }
        }

        // Messages list
        LazyColumn(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth(),
            state = listState,
            contentPadding = PaddingValues(vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(0.dp)
        ) {
            items(messages, key = { it.id }) { message ->
                // Find the user message immediately before a failed assistant message for retry
                val retryUserMsgId = if (message.status == "failed" && !message.isUser) {
                    val idx = messages.indexOf(message)
                    if (idx > 0) messages[idx - 1].takeIf { it.isUser }?.id else null
                } else null

                MessageBubble(
                    message = message,
                    onRetry = retryUserMsgId?.let { id -> { sendMessage(retryId = id) } }
                )
            }

            if (isLoading) {
                item(key = "typing") {
                    TypingIndicator()
                }
            }
        }

        // Mini player
        MiniPlayer(
            nowPlaying = nowPlaying,
            onPlayPause = { musicPlayerManager.togglePlayPause() },
            onStop = { musicPlayerManager.stop() },
            onGetPosition = {
                Pair(musicPlayerManager.getPositionMs(), musicPlayerManager.getDurationMs())
            }
        )

        // Input area
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .background(ClawSurface)
                .padding(horizontal = 12.dp, vertical = 8.dp)
                .imePadding(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Text input
            OutlinedTextField(
                value = inputText,
                onValueChange = { inputText = it },
                placeholder = {
                    Text(
                        stringResource(R.string.chat_hint),
                        color = ClawTextSecondary
                    )
                },
                singleLine = false,
                maxLines = 4,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = { sendMessage() }),
                colors = OutlinedTextFieldDefaults.colors(
                    focusedBorderColor = ClawAccent,
                    unfocusedBorderColor = ClawBorder,
                    cursorColor = ClawAccent,
                    focusedTextColor = ClawTextPrimary,
                    unfocusedTextColor = ClawTextPrimary,
                ),
                shape = RoundedCornerShape(24.dp),
                modifier = Modifier.weight(1f)
            )

            Spacer(modifier = Modifier.width(8.dp))

            // Voice button
            VoiceButton(
                state = streamState,
                wakeWordAvailable = wakeWordAvailable,
                onPushToTalkStart = onPushToTalkStart,
                onPushToTalkStop = onPushToTalkStop,
                onTap = {
                    if (inputText.isNotBlank()) {
                        sendMessage()
                    }
                },
                modifier = Modifier.size(56.dp)
            )

            // Send button (visible when there's text)
            AnimatedVisibility(
                visible = inputText.isNotBlank(),
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                IconButton(
                    onClick = { sendMessage() },
                    modifier = Modifier
                        .padding(start = 4.dp)
                        .size(44.dp)
                        .clip(CircleShape)
                        .background(ClawAccent)
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Send,
                        contentDescription = stringResource(R.string.send_button),
                        tint = ClawBackground,
                        modifier = Modifier.size(22.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun SettingsDialog(
    isServiceRunning: Boolean,
    wakeWordAvailable: Boolean,
    onToggleService: (Boolean) -> Unit,
    onDisconnect: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = ClawSurface,
        titleContentColor = ClawTextPrimary,
        textContentColor = ClawTextPrimary,
        title = {
            Text(
                text = stringResource(R.string.settings_label),
                style = MaterialTheme.typography.titleLarge
            )
        },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(16.dp)) {
                // Wake word service toggle
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Wake Word Service",
                            style = MaterialTheme.typography.bodyLarge,
                            color = ClawTextPrimary
                        )
                        Text(
                            text = if (!wakeWordAvailable) {
                                stringResource(R.string.wake_word_no_models)
                            } else if (isServiceRunning) {
                                stringResource(R.string.wake_word_active)
                            } else {
                                stringResource(R.string.wake_word_disabled)
                            },
                            style = MaterialTheme.typography.bodySmall,
                            color = ClawTextSecondary
                        )
                    }
                    Switch(
                        checked = isServiceRunning,
                        onCheckedChange = onToggleService,
                        colors = SwitchDefaults.colors(
                            checkedThumbColor = ClawAccent,
                            checkedTrackColor = ClawSurfaceVariant,
                            uncheckedThumbColor = ClawTextSecondary,
                            uncheckedTrackColor = ClawSurfaceVariant,
                        )
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))

                // Disconnect button
                TextButton(
                    onClick = onDisconnect,
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Logout,
                        contentDescription = null,
                        tint = ClawError,
                        modifier = Modifier.size(18.dp)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = stringResource(R.string.disconnect_label),
                        color = ClawError
                    )
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) {
                Text("Done", color = ClawAccent)
            }
        }
    )
}
