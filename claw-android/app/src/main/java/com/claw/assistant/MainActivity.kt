package com.claw.assistant

import android.Manifest
import android.app.Activity
import android.app.role.RoleManager
import android.content.Intent
import android.content.pm.PackageManager
import android.net.VpnService
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.Crossfade
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Chat
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.claw.assistant.network.AppUpdate
import com.claw.assistant.network.AudioStreamManager
import com.claw.assistant.network.AudioStreamState
import com.claw.assistant.network.TunnelManager
import com.claw.assistant.network.TunnelState
import com.claw.assistant.network.UpdateManager
import com.claw.assistant.service.WakeWordEngine
import com.claw.assistant.service.NotificationService
import com.claw.assistant.service.WakeWordService
import com.claw.assistant.ui.screens.AdminScreen
import com.claw.assistant.ui.screens.ChatScreen
import com.claw.assistant.ui.screens.SetupScreen
import com.claw.assistant.ui.theme.ClawAccent
import com.claw.assistant.ui.theme.ClawBackground
import com.claw.assistant.ui.theme.ClawSurface
import com.claw.assistant.ui.theme.ClawTextPrimary
import com.claw.assistant.ui.theme.ClawTextSecondary
import com.claw.assistant.ui.theme.ClawTheme
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var app: ClawApplication
    private var audioStreamManager: AudioStreamManager? = null
    private val audioState = MutableStateFlow(AudioStreamState.IDLE)
    private var wakeWordAvailable = mutableStateOf(false)
    private var isServiceRunning = mutableStateOf(false)
    private var pendingUpdate = mutableStateOf<AppUpdate?>(null)

    private val requiredPermissions = buildList {
        add(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            add(Manifest.permission.POST_NOTIFICATIONS)
        }
    }.toTypedArray()

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val micGranted = permissions[Manifest.permission.RECORD_AUDIO] == true
        if (micGranted) {
            Log.d(TAG, "Microphone permission granted")
            checkWakeWordModels()
        } else {
            Log.w(TAG, "Microphone permission denied")
        }
    }

    private var onVpnPermissionResult: ((Boolean) -> Unit)? = null

    private val vpnPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        val granted = result.resultCode == Activity.RESULT_OK
        if (granted) {
            Log.d(TAG, "VPN permission granted")
        } else {
            Log.w(TAG, "VPN permission denied")
        }
        onVpnPermissionResult?.invoke(granted)
        onVpnPermissionResult = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        app = application as ClawApplication

        // Check permissions
        requestPermissionsIfNeeded()

        // Check wake word model availability
        checkWakeWordModels()

        // Check if service is already running
        isServiceRunning.value = WakeWordService.isServiceRunning(this)

        // Restore session if credentials are saved
        val credentials = app.preferencesManager.getCredentials()
        if (credentials != null) {
            // Always reconfigure the API client from saved credentials
            app.apiClient.configure(credentials.first, credentials.second)
            Log.d(TAG, "API client restored from saved credentials")

            // Auto-reconnect WireGuard tunnel
            val savedWgConfig = app.preferencesManager.getWgConfig()
            if (savedWgConfig != null) {
                kotlinx.coroutines.MainScope().launch {
                    try {
                        val vpnIntent = TunnelManager.prepareVpn(this@MainActivity)
                        if (vpnIntent == null) {
                            app.tunnelManager.connect(savedWgConfig)
                            Log.d(TAG, "Auto-reconnected WireGuard tunnel")
                        } else {
                            Log.w(TAG, "VPN permission revoked — tunnel not reconnected, requesting permission")
                            vpnPermissionLauncher.launch(vpnIntent)
                        }
                    } catch (e: Exception) {
                        Log.w(TAG, "Auto-reconnect failed", e)
                    }
                }
            }

            // Check for app updates
            kotlinx.coroutines.MainScope().launch {
                try {
                    val update = UpdateManager(this@MainActivity)
                        .checkForUpdate(credentials.first, credentials.second)
                    if (update != null) {
                        pendingUpdate.value = update
                        Log.d(TAG, "Update available: ${update.versionName}")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Update check failed", e)
                }
            }

            // Start SSE notification listener
            NotificationService.start(this)

            // Request default assistant role
            requestAssistantRole()
        }

        setContent {
            ClawTheme {
                ClawApp()
            }
        }
    }

    override fun onResume() {
        super.onResume()
        isServiceRunning.value = WakeWordService.isServiceRunning(this)
    }

    @Composable
    private fun ClawApp() {
        var isConfigured by remember {
            mutableStateOf(app.preferencesManager.isConfigured())
        }
        val serviceRunning by isServiceRunning
        val wakeWordReady by wakeWordAvailable

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(ClawBackground)
        ) {
            Crossfade(
                targetState = isConfigured,
                animationSpec = tween(400),
                label = "screenTransition"
            ) { configured ->
                if (!configured) {
                    SetupScreen(
                        onConnect = { provisioningCode ->
                            connectWithProvisioning(provisioningCode).also { success ->
                                if (success) isConfigured = true
                            }
                        }
                    )
                } else {
                    // Initialize AudioStreamManager for this session
                    val streamManager = remember {
                        AudioStreamManager(app.apiClient).also { manager ->
                            audioStreamManager = manager
                        }
                    }

                    val credentials = remember { app.preferencesManager.getCredentials() }
                    var selectedTab by rememberSaveable { mutableIntStateOf(0) }

                    Scaffold(
                        containerColor = ClawBackground,
                        bottomBar = {
                            NavigationBar(
                                containerColor = ClawSurface,
                                contentColor = ClawTextPrimary
                            ) {
                                NavigationBarItem(
                                    selected = selectedTab == 0,
                                    onClick = { selectedTab = 0 },
                                    icon = {
                                        Icon(
                                            imageVector = Icons.AutoMirrored.Filled.Chat,
                                            contentDescription = "Chat"
                                        )
                                    },
                                    label = { Text("Chat") },
                                    colors = NavigationBarItemDefaults.colors(
                                        selectedIconColor = ClawAccent,
                                        selectedTextColor = ClawAccent,
                                        unselectedIconColor = ClawTextSecondary,
                                        unselectedTextColor = ClawTextSecondary,
                                        indicatorColor = ClawSurface
                                    )
                                )
                                NavigationBarItem(
                                    selected = selectedTab == 1,
                                    onClick = { selectedTab = 1 },
                                    icon = {
                                        Icon(
                                            imageVector = Icons.Default.Settings,
                                            contentDescription = "Admin"
                                        )
                                    },
                                    label = { Text("Admin") },
                                    colors = NavigationBarItemDefaults.colors(
                                        selectedIconColor = ClawAccent,
                                        selectedTextColor = ClawAccent,
                                        unselectedIconColor = ClawTextSecondary,
                                        unselectedTextColor = ClawTextSecondary,
                                        indicatorColor = ClawSurface
                                    )
                                )
                            }
                        }
                    ) { innerPadding ->
                        when (selectedTab) {
                            0 -> ChatScreen(
                                apiClient = app.apiClient,
                                database = app.database,
                                musicPlayerManager = app.musicPlayerManager,
                                audioStreamState = streamManager.state,
                                wakeWordAvailable = wakeWordReady,
                                isServiceRunning = serviceRunning,
                                updateAvailable = pendingUpdate.value,
                                onUpdateClick = {
                                    val creds = app.preferencesManager.getCredentials()
                                    if (creds != null) {
                                        UpdateManager(this@MainActivity).downloadAndInstall(creds.first, creds.second)
                                    }
                                },
                                onToggleService = { enabled ->
                                    toggleWakeWordService(enabled)
                                },
                                onPushToTalkStart = {
                                    handlePushToTalkStart()
                                },
                                onPushToTalkStop = {
                                    handlePushToTalkStop()
                                },
                                onDisconnect = {
                                    disconnect()
                                    isConfigured = false
                                },
                                modifier = Modifier.padding(bottom = innerPadding.calculateBottomPadding())
                            )
                            1 -> {
                                if (credentials != null) {
                                    AdminScreen(
                                        serverUrl = credentials.first,
                                        apiKey = credentials.second,
                                        modifier = Modifier.padding(bottom = innerPadding.calculateBottomPadding())
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private suspend fun connectWithProvisioning(provisioningCode: String): Boolean {
        try {
            // Parse provisioning code
            val data = TunnelManager.parseProvisioningCode(provisioningCode)

            // Check VPN permission — suspend until granted
            val vpnIntent = TunnelManager.prepareVpn(this@MainActivity)
            if (vpnIntent != null) {
                val granted = kotlinx.coroutines.suspendCancellableCoroutine { cont ->
                    onVpnPermissionResult = { result ->
                        if (cont.isActive) cont.resumeWith(Result.success(result))
                    }
                    cont.invokeOnCancellation { onVpnPermissionResult = null }
                    vpnPermissionLauncher.launch(vpnIntent)
                }
                if (!granted) {
                    throw Exception("VPN permission denied — required for secure connection")
                }
            }

            // Build WireGuard config and connect tunnel
            val wgConfig = TunnelManager.buildWgConfig(data)
            Log.d(TAG, "Connecting tunnel to endpoint: ${data.wgEndpoint}")
            app.tunnelManager.connect(wgConfig)

            // Give the tunnel a moment to establish the handshake
            Log.d(TAG, "Tunnel interface up, waiting for handshake...")
            kotlinx.coroutines.delay(2000)

            // Configure API client with tunnel server URL
            Log.d(TAG, "Pinging server at: ${data.serverUrl}")
            app.apiClient.configure(data.serverUrl, data.apiKey)

            // Verify connection through tunnel with retries
            var success = false
            var lastError: Exception? = null
            for (attempt in 1..3) {
                try {
                    success = app.apiClient.ping()
                    if (success) break
                } catch (e: Exception) {
                    lastError = e
                    Log.w(TAG, "Ping attempt $attempt failed: ${e.message}")
                }
                if (!success && attempt < 3) kotlinx.coroutines.delay(2000)
            }

            if (success) {
                app.preferencesManager.saveCredentials(data.serverUrl, data.apiKey)
                app.preferencesManager.saveWgConfig(wgConfig)
                app.preferencesManager.saveProvisioningCode(provisioningCode)
                Log.d(TAG, "Connected via WireGuard tunnel to ${data.serverUrl}")
                // Check for updates now that we're connected
                try {
                    val update = UpdateManager(this@MainActivity)
                        .checkForUpdate(data.serverUrl, data.apiKey)
                    if (update != null) {
                        pendingUpdate.value = update
                        Log.d(TAG, "Update available: ${update.versionName}")
                    }
                } catch (e: Exception) {
                    Log.w(TAG, "Post-connect update check failed", e)
                }
            } else {
                app.tunnelManager.disconnect()
                val detail = lastError?.message ?: "Server unreachable"
                throw Exception("Tunnel up but server not reachable: $detail")
            }
            return success
        } catch (e: Exception) {
            Log.e(TAG, "Provisioning connection failed", e)
            try { app.tunnelManager.disconnect() } catch (_: Exception) {}
            throw e
        }
    }

    private fun disconnect() {
        toggleWakeWordService(false)
        NotificationService.stop(this)
        audioStreamManager?.release()
        audioStreamManager = null
        app.musicPlayerManager.stop()
        kotlinx.coroutines.MainScope().launch {
            try { app.tunnelManager.disconnect() } catch (_: Exception) {}
        }
        app.preferencesManager.clearCredentials()
        Log.d(TAG, "Disconnected from server")
    }

    private fun handlePushToTalkStart() {
        if (!hasMicPermission()) {
            requestPermissionsIfNeeded()
            return
        }

        val manager = audioStreamManager ?: return

        // If service is running, use the service for push-to-talk
        if (isServiceRunning.value) {
            val intent = Intent(this, WakeWordService::class.java).apply {
                action = WakeWordService.ACTION_PUSH_TO_TALK_START
            }
            startService(intent)
        } else {
            // Use the local audio stream manager
            if (manager.state.value == AudioStreamState.IDLE) {
                manager.startCapture()
            }
            manager.startPushToTalk()
        }
    }

    private fun handlePushToTalkStop() {
        if (isServiceRunning.value) {
            val intent = Intent(this, WakeWordService::class.java).apply {
                action = WakeWordService.ACTION_PUSH_TO_TALK_STOP
            }
            startService(intent)
        } else {
            audioStreamManager?.stopPushToTalk()
        }
    }

    private fun toggleWakeWordService(enabled: Boolean) {
        val intent = Intent(this, WakeWordService::class.java)
        if (enabled) {
            if (!hasMicPermission()) {
                requestPermissionsIfNeeded()
                return
            }
            intent.action = WakeWordService.ACTION_START
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                startForegroundService(intent)
            } else {
                startService(intent)
            }
            isServiceRunning.value = true
        } else {
            intent.action = WakeWordService.ACTION_STOP
            startService(intent)
            isServiceRunning.value = false
        }
        app.preferencesManager.setWakeWordEnabled(enabled)
    }

    private fun checkWakeWordModels() {
        val engine = WakeWordEngine(this)
        val available = engine.initialize()
        engine.release()
        wakeWordAvailable.value = available
    }

    private fun requestPermissionsIfNeeded() {
        val needed = requiredPermissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needed.isNotEmpty()) {
            permissionLauncher.launch(needed.toTypedArray())
        }
    }

    private fun requestAssistantRole() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val roleManager = getSystemService(RoleManager::class.java)
            if (roleManager != null && !roleManager.isRoleHeld(RoleManager.ROLE_ASSISTANT)) {
                val intent = roleManager.createRequestRoleIntent(RoleManager.ROLE_ASSISTANT)
                startActivityForResult(intent, 1001)
            }
        }
    }

    private fun hasMicPermission(): Boolean =
        ContextCompat.checkSelfPermission(
            this, Manifest.permission.RECORD_AUDIO
        ) == PackageManager.PERMISSION_GRANTED

    override fun onDestroy() {
        audioStreamManager?.release()
        super.onDestroy()
    }

    companion object {
        private const val TAG = "MainActivity"
    }
}
