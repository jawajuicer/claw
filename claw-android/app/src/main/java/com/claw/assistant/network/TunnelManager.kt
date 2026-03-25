package com.claw.assistant.network

import android.content.Context
import android.content.Intent
import android.net.VpnService
import android.util.Base64
import android.util.Log
import com.wireguard.android.backend.GoBackend
import com.wireguard.android.backend.Tunnel
import com.wireguard.config.Config
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.ByteArrayInputStream

enum class TunnelState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    ERROR
}

data class ProvisioningData(
    val apiKey: String,
    val wgPrivateKey: String,
    val wgAddress: String,
    val wgServerPubKey: String,
    val wgPsk: String,
    val wgEndpoint: String,
    val wgAllowedIps: String,
    val serverUrl: String,
)

class TunnelManager(context: Context) {

    private val appContext = context.applicationContext
    private val backend = GoBackend(appContext)
    private var tunnel: ClawTunnel? = null

    private val _state = MutableStateFlow(TunnelState.DISCONNECTED)
    val state: StateFlow<TunnelState> = _state

    suspend fun connect(wgConfig: String) = withContext(Dispatchers.IO) {
        try {
            _state.value = TunnelState.CONNECTING

            val config = Config.parse(ByteArrayInputStream(wgConfig.toByteArray()))
            val clawTunnel = ClawTunnel("claw")

            backend.setState(clawTunnel, Tunnel.State.UP, config)
            tunnel = clawTunnel
            _state.value = TunnelState.CONNECTED

            Log.d(TAG, "WireGuard tunnel connected")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect tunnel", e)
            _state.value = TunnelState.ERROR
            throw e
        }
    }

    suspend fun disconnect() = withContext(Dispatchers.IO) {
        try {
            tunnel?.let {
                backend.setState(it, Tunnel.State.DOWN, null)
            }
            tunnel = null
            _state.value = TunnelState.DISCONNECTED
            Log.d(TAG, "WireGuard tunnel disconnected")
        } catch (e: Exception) {
            Log.e(TAG, "Error disconnecting tunnel", e)
            _state.value = TunnelState.ERROR
        }
    }

    fun isConnected(): Boolean = _state.value == TunnelState.CONNECTED

    companion object {
        private const val TAG = "TunnelManager"

        /**
         * Decode a base64 provisioning code from the admin panel into
         * all the connection details the app needs.
         */
        fun parseProvisioningCode(code: String): ProvisioningData {
            val decoded: String
            try {
                decoded = String(Base64.decode(code.trim(), Base64.URL_SAFE or Base64.NO_WRAP))
            } catch (e: IllegalArgumentException) {
                throw IllegalArgumentException("Invalid provisioning code format — scan a QR code from the Claw admin panel")
            }
            val json = try {
                JSONObject(decoded)
            } catch (e: org.json.JSONException) {
                throw IllegalArgumentException("Invalid provisioning code — not a valid Claw configuration")
            }
            val wg = json.optJSONObject("wg")
                ?: throw IllegalArgumentException("Invalid provisioning code — missing connection data")
            return ProvisioningData(
                apiKey = json.getString("api_key"),
                wgPrivateKey = wg.getString("private_key"),
                wgAddress = wg.getString("address"),
                wgServerPubKey = wg.getString("server_public_key"),
                wgPsk = wg.getString("psk"),
                wgEndpoint = wg.getString("endpoint"),
                wgAllowedIps = wg.optString("allowed_ips", "10.10.0.0/24"),
                serverUrl = json.getString("server_url"),
            )
        }

        /**
         * Build a WireGuard INI config string from provisioning data.
         */
        fun buildWgConfig(data: ProvisioningData): String {
            return "[Interface]\n" +
                "PrivateKey = ${data.wgPrivateKey}\n" +
                "Address = ${data.wgAddress}/32\n" +
                "\n" +
                "[Peer]\n" +
                "PublicKey = ${data.wgServerPubKey}\n" +
                "PresharedKey = ${data.wgPsk}\n" +
                "Endpoint = ${data.wgEndpoint}\n" +
                "AllowedIPs = ${data.wgAllowedIps}\n" +
                "PersistentKeepalive = 25\n"
        }

        /**
         * Check if VPN permission is granted.
         * Returns null if already granted, or an Intent to launch for permission.
         */
        fun prepareVpn(context: Context): Intent? {
            return VpnService.prepare(context)
        }
    }

    private class ClawTunnel(private val tunnelName: String) : Tunnel {
        override fun getName(): String = tunnelName
        override fun onStateChange(newState: Tunnel.State) {
            Log.d("ClawTunnel", "State changed to: $newState")
        }
    }
}
