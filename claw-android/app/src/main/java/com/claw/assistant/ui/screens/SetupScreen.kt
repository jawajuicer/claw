package com.claw.assistant.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.QrCodeScanner
import androidx.compose.material.icons.filled.ContentPaste
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.Visibility
import androidx.compose.material.icons.filled.VisibilityOff
import androidx.compose.material.icons.filled.VpnKey
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.claw.assistant.ui.components.QrScanner
import com.claw.assistant.ui.theme.ClawAccent
import com.claw.assistant.ui.theme.ClawBackground
import com.claw.assistant.ui.theme.ClawBorder
import com.claw.assistant.ui.theme.ClawError
import com.claw.assistant.ui.theme.ClawSuccess
import com.claw.assistant.ui.theme.ClawTextPrimary
import com.claw.assistant.ui.theme.ClawTextSecondary
import kotlinx.coroutines.launch

@Composable
fun SetupScreen(
    onConnect: suspend (provisioningCode: String) -> Boolean,
    modifier: Modifier = Modifier
) {
    var provisioningCode by remember { mutableStateOf("") }
    var showCode by remember { mutableStateOf(false) }
    var showScanner by remember { mutableStateOf(false) }
    var isConnecting by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf<String?>(null) }
    var isError by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val focusManager = LocalFocusManager.current
    val clipboardManager = LocalClipboardManager.current

    val textFieldColors = OutlinedTextFieldDefaults.colors(
        focusedBorderColor = ClawAccent,
        unfocusedBorderColor = ClawBorder,
        focusedLabelColor = ClawAccent,
        unfocusedLabelColor = ClawTextSecondary,
        cursorColor = ClawAccent,
        focusedTextColor = ClawTextPrimary,
        unfocusedTextColor = ClawTextPrimary,
        focusedLeadingIconColor = ClawAccent,
        unfocusedLeadingIconColor = ClawTextSecondary,
    )

    fun doConnect() {
        if (provisioningCode.isBlank()) {
            statusMessage = "Paste your provisioning code from the admin panel"
            isError = true
            return
        }
        isConnecting = true
        statusMessage = null
        isError = false

        scope.launch {
            try {
                statusMessage = "Establishing encrypted tunnel..."
                isError = false
                val success = onConnect(provisioningCode.trim())
                if (!success) {
                    statusMessage = "Connection failed. Check your provisioning code."
                    isError = true
                }
            } catch (e: Exception) {
                statusMessage = e.message ?: "Connection failed"
                isError = true
            } finally {
                isConnecting = false
            }
        }
    }

    if (showScanner) {
        QrScanner(
            onResult = { code ->
                showScanner = false
                provisioningCode = code
                // Auto-connect after scan
                isConnecting = true
                statusMessage = "Establishing encrypted tunnel..."
                isError = false
                scope.launch {
                    try {
                        val success = onConnect(code.trim())
                        if (!success) {
                            statusMessage = "Connection failed. Invalid QR code?"
                            isError = true
                        }
                    } catch (e: Exception) {
                        statusMessage = e.message ?: "Connection failed"
                        isError = true
                    } finally {
                        isConnecting = false
                    }
                }
            },
            onDismiss = { showScanner = false }
        )
        return
    }

    Box(
        modifier = modifier
            .fillMaxSize()
            .padding(24.dp)
            .imePadding(),
        contentAlignment = Alignment.Center
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            // App title
            Text(
                text = "The Claw",
                style = MaterialTheme.typography.headlineLarge,
                color = ClawTextPrimary,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = "Private voice AI assistant",
                style = MaterialTheme.typography.bodyLarge,
                color = ClawTextSecondary,
                textAlign = TextAlign.Center
            )

            Spacer(modifier = Modifier.height(48.dp))

            // Lock icon + encryption info
            Row(
                verticalAlignment = Alignment.CenterVertically,
                modifier = Modifier.padding(bottom = 16.dp)
            ) {
                Icon(
                    Icons.Default.Lock,
                    contentDescription = null,
                    tint = ClawAccent,
                    modifier = Modifier.size(20.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "End-to-end encrypted via WireGuard",
                    style = MaterialTheme.typography.bodyMedium,
                    color = ClawTextSecondary
                )
            }

            // QR Code scan button
            Button(
                onClick = { showScanner = true },
                colors = ButtonDefaults.buttonColors(
                    containerColor = ClawAccent.copy(alpha = 0.15f),
                    contentColor = ClawAccent,
                ),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp)
            ) {
                Icon(
                    Icons.Default.QrCodeScanner,
                    contentDescription = null,
                    modifier = Modifier.size(24.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Scan QR Code",
                    style = MaterialTheme.typography.titleMedium
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Text(
                text = "or enter code manually",
                style = MaterialTheme.typography.bodySmall,
                color = ClawTextSecondary,
                textAlign = TextAlign.Center,
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Provisioning code field
            OutlinedTextField(
                value = provisioningCode,
                onValueChange = {
                    provisioningCode = it
                    statusMessage = null
                    isError = false
                },
                label = { Text("Provisioning Code") },
                placeholder = { Text("Paste code from admin panel") },
                leadingIcon = {
                    Icon(Icons.Default.VpnKey, contentDescription = null)
                },
                trailingIcon = {
                    Row {
                        IconButton(onClick = {
                            clipboardManager.getText()?.let {
                                provisioningCode = it.text
                            }
                        }) {
                            Icon(
                                Icons.Default.ContentPaste,
                                contentDescription = "Paste from clipboard",
                                tint = ClawTextSecondary
                            )
                        }
                        IconButton(onClick = { showCode = !showCode }) {
                            Icon(
                                imageVector = if (showCode) Icons.Default.VisibilityOff
                                    else Icons.Default.Visibility,
                                contentDescription = if (showCode) "Hide" else "Show",
                                tint = ClawTextSecondary
                            )
                        }
                    }
                },
                singleLine = true,
                visualTransformation = if (showCode) VisualTransformation.None
                    else PasswordVisualTransformation(),
                keyboardOptions = KeyboardOptions(
                    keyboardType = KeyboardType.Password,
                    imeAction = ImeAction.Done
                ),
                keyboardActions = KeyboardActions(
                    onDone = {
                        focusManager.clearFocus()
                        doConnect()
                    }
                ),
                colors = textFieldColors,
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Status message
            AnimatedVisibility(
                visible = statusMessage != null,
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                Text(
                    text = statusMessage ?: "",
                    style = MaterialTheme.typography.bodySmall,
                    color = if (isError) ClawError else ClawAccent,
                    textAlign = TextAlign.Center,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 8.dp)
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            // Connect button
            Button(
                onClick = { doConnect() },
                enabled = !isConnecting && provisioningCode.isNotBlank(),
                colors = ButtonDefaults.buttonColors(
                    containerColor = ClawAccent,
                    contentColor = ClawBackground,
                    disabledContainerColor = ClawBorder,
                    disabledContentColor = ClawTextSecondary,
                ),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .height(52.dp)
            ) {
                if (isConnecting) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(24.dp),
                        color = ClawBackground,
                        strokeWidth = 2.dp
                    )
                } else {
                    Text(
                        text = "Connect Securely",
                        style = MaterialTheme.typography.titleMedium
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            Text(
                text = "Create a device in the admin panel to get your provisioning code. " +
                    "It contains your API key and VPN credentials — one code, fully encrypted.",
                style = MaterialTheme.typography.bodySmall,
                color = ClawTextSecondary,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(horizontal = 16.dp)
            )
        }
    }
}
