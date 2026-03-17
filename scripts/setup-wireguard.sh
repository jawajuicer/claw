#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# WireGuard VPN Setup for The Claw
#
# Sets up a WireGuard VPN server on the Claw host machine so you can
# access The Claw from anywhere — phone, laptop, car — with full E2E
# encryption (ChaCha20-Poly1305, Curve25519).
#
# Usage:
#   sudo bash scripts/setup-wireguard.sh
#
# After running:
#   1. Forward UDP port 51820 on your router to this machine's LAN IP
#   2. Import the phone config via the WireGuard Android app (QR code)
#   3. Import the desktop config via the WireGuard desktop app
#   4. Enable remote: set remote.enabled=true in config.yaml
#   5. Register devices: POST /api/devices with admin auth
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
err()   { echo -e "${RED}[✗]${NC} $*" >&2; }
header() { echo -e "\n${CYAN}── $* ──${NC}"; }

# ── Checks ──
if [[ $EUID -ne 0 ]]; then
    err "This script must be run as root (sudo)"
    exit 1
fi

# ── Configuration ──
WG_IFACE="wg0"
WG_PORT=51820
WG_SUBNET="10.10.0"
WG_DIR="/etc/wireguard"
CLIENT_DIR="${WG_DIR}/clients"

SERVER_IP="${WG_SUBNET}.1/24"
PHONE_IP="${WG_SUBNET}.2/32"
DESKTOP_IP="${WG_SUBNET}.3/32"

# Detect LAN IP
LAN_IP=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'src \K[0-9.]+' || hostname -I | awk '{print $1}')
# Detect WAN interface
WAN_IFACE=$(ip -4 route get 1.1.1.1 2>/dev/null | grep -oP 'dev \K\S+' || echo "eth0")

header "Detecting public IP"
PUBLIC_IP=$(curl -4 -s --max-time 5 https://ifconfig.me || curl -4 -s --max-time 5 https://api.ipify.org || echo "YOUR_PUBLIC_IP")
info "Public IP: ${PUBLIC_IP}"
info "LAN IP:    ${LAN_IP}"
info "WAN iface: ${WAN_IFACE}"

# ── Install WireGuard ──
header "Installing WireGuard"
if command -v wg &>/dev/null; then
    info "WireGuard already installed"
else
    apt-get update -qq
    apt-get install -y -qq wireguard wireguard-tools qrencode
    info "WireGuard installed"
fi

# Ensure qrencode is available
if ! command -v qrencode &>/dev/null; then
    apt-get install -y -qq qrencode
fi

# ── Enable IP forwarding ──
header "Enabling IP forwarding"
if grep -q '^net.ipv4.ip_forward=1' /etc/sysctl.conf; then
    info "IP forwarding already enabled"
else
    echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.conf
    sysctl -w net.ipv4.ip_forward=1 >/dev/null
    info "IP forwarding enabled"
fi

# ── Generate keys ──
header "Generating cryptographic keys"
mkdir -p "${WG_DIR}" "${CLIENT_DIR}"
chmod 700 "${WG_DIR}"

generate_keypair() {
    local name="$1"
    local dir="$2"
    if [[ -f "${dir}/${name}_private" ]]; then
        warn "Keys for '${name}' already exist, skipping"
        return
    fi
    umask 077
    wg genkey | tee "${dir}/${name}_private" | wg pubkey > "${dir}/${name}_public"
    # Generate preshared key for extra security
    wg genpsk > "${dir}/${name}_psk"
    info "Generated keys for ${name}"
}

generate_keypair "server" "${WG_DIR}"
generate_keypair "phone" "${CLIENT_DIR}"
generate_keypair "desktop" "${CLIENT_DIR}"

# Read keys
SERVER_PRIV=$(cat "${WG_DIR}/server_private")
SERVER_PUB=$(cat "${WG_DIR}/server_public")
PHONE_PRIV=$(cat "${CLIENT_DIR}/phone_private")
PHONE_PUB=$(cat "${CLIENT_DIR}/phone_public")
PHONE_PSK=$(cat "${CLIENT_DIR}/phone_psk")
DESKTOP_PRIV=$(cat "${CLIENT_DIR}/desktop_private")
DESKTOP_PUB=$(cat "${CLIENT_DIR}/desktop_public")
DESKTOP_PSK=$(cat "${CLIENT_DIR}/desktop_psk")

# ── Server config ──
header "Creating server configuration"

if [[ -f "${WG_DIR}/${WG_IFACE}.conf" ]]; then
    warn "Server config already exists, backing up"
    cp "${WG_DIR}/${WG_IFACE}.conf" "${WG_DIR}/${WG_IFACE}.conf.bak.$(date +%s)"
fi

cat > "${WG_DIR}/${WG_IFACE}.conf" << EOF
# The Claw — WireGuard VPN Server
[Interface]
Address = ${SERVER_IP}
ListenPort = ${WG_PORT}
PrivateKey = ${SERVER_PRIV}

# NAT for clients to reach LAN services (Claw on port 8080, llama-swap on 8081)
PostUp = iptables -t nat -I POSTROUTING -o ${WAN_IFACE} -j MASQUERADE
PostUp = iptables -A FORWARD -i ${WG_IFACE} -j ACCEPT
PostUp = iptables -A FORWARD -o ${WG_IFACE} -j ACCEPT
PostDown = iptables -t nat -D POSTROUTING -o ${WAN_IFACE} -j MASQUERADE
PostDown = iptables -D FORWARD -i ${WG_IFACE} -j ACCEPT
PostDown = iptables -D FORWARD -o ${WG_IFACE} -j ACCEPT

# Android phone
[Peer]
PublicKey = ${PHONE_PUB}
PresharedKey = ${PHONE_PSK}
AllowedIPs = ${PHONE_IP}

# Desktop / Laptop
[Peer]
PublicKey = ${DESKTOP_PUB}
PresharedKey = ${DESKTOP_PSK}
AllowedIPs = ${DESKTOP_IP}
EOF

chmod 600 "${WG_DIR}/${WG_IFACE}.conf"
info "Server config written to ${WG_DIR}/${WG_IFACE}.conf"

# ── Client configs ──
header "Creating client configurations"

# Phone config (split tunnel — only VPN subnet through tunnel, saves battery)
cat > "${CLIENT_DIR}/phone.conf" << EOF
# The Claw — Phone VPN Config
[Interface]
Address = ${PHONE_IP}
PrivateKey = ${PHONE_PRIV}
DNS = 1.1.1.1, 9.9.9.9

[Peer]
PublicKey = ${SERVER_PUB}
PresharedKey = ${PHONE_PSK}
Endpoint = ${PUBLIC_IP}:${WG_PORT}
AllowedIPs = ${WG_SUBNET}.0/24
PersistentKeepalive = 25
EOF

# Desktop config (split tunnel)
cat > "${CLIENT_DIR}/desktop.conf" << EOF
# The Claw — Desktop VPN Config
[Interface]
Address = ${DESKTOP_IP}
PrivateKey = ${DESKTOP_PRIV}
DNS = 1.1.1.1, 9.9.9.9

[Peer]
PublicKey = ${SERVER_PUB}
PresharedKey = ${DESKTOP_PSK}
Endpoint = ${PUBLIC_IP}:${WG_PORT}
AllowedIPs = ${WG_SUBNET}.0/24
PersistentKeepalive = 25
EOF

chmod 600 "${CLIENT_DIR}"/*.conf
info "Client configs written to ${CLIENT_DIR}/"

# ── Generate QR code for phone ──
header "Generating QR code for phone"
qrencode -t ansiutf8 < "${CLIENT_DIR}/phone.conf"
echo ""
info "Scan this QR code with the WireGuard Android app"
echo ""

# Also save QR as PNG for later use
qrencode -t png -o "${CLIENT_DIR}/phone_qr.png" < "${CLIENT_DIR}/phone.conf"
info "QR code also saved to ${CLIENT_DIR}/phone_qr.png"

# ── Firewall ──
header "Configuring firewall"
if command -v ufw &>/dev/null; then
    ufw allow ${WG_PORT}/udp >/dev/null 2>&1 || true
    info "UFW rule added for UDP ${WG_PORT}"
else
    info "No UFW detected, ensure UDP ${WG_PORT} is open in your firewall"
fi

# ── Start WireGuard ──
header "Starting WireGuard"

# Bring down if already running
wg-quick down ${WG_IFACE} 2>/dev/null || true

wg-quick up ${WG_IFACE}
systemctl enable wg-quick@${WG_IFACE}
info "WireGuard interface ${WG_IFACE} is UP"

# Show status
echo ""
wg show ${WG_IFACE}

# ── Summary ──
header "Setup Complete"
echo ""
echo -e "  ${GREEN}VPN Server:${NC}     ${WG_SUBNET}.1  (this machine)"
echo -e "  ${GREEN}Phone:${NC}          ${WG_SUBNET}.2  (Android)"
echo -e "  ${GREEN}Desktop:${NC}        ${WG_SUBNET}.3  (laptop/PC)"
echo -e "  ${GREEN}WireGuard Port:${NC} UDP ${WG_PORT}"
echo -e "  ${GREEN}Public IP:${NC}      ${PUBLIC_IP}"
echo ""
echo -e "  ${CYAN}Client configs:${NC} ${CLIENT_DIR}/"
echo -e "  ${CYAN}Phone QR:${NC}       ${CLIENT_DIR}/phone_qr.png"
echo ""
echo -e "  ${YELLOW}REQUIRED: Router Port Forwarding${NC}"
echo -e "  Forward UDP port ${WG_PORT} → ${LAN_IP}:${WG_PORT}"
echo ""
echo -e "  ${YELLOW}REQUIRED: Enable Remote Access in Claw${NC}"
echo -e "  1. Set ${CYAN}remote.enabled: true${NC} in config.yaml"
echo -e "  2. Register devices via admin panel: POST /api/devices"
echo ""
echo -e "  ${YELLOW}Phone Setup:${NC}"
echo -e "  1. Install WireGuard app from Play Store"
echo -e "  2. Tap '+' → 'Scan from QR code' → scan the QR above"
echo -e "  3. Enable the tunnel"
echo -e "  4. Access Claw at: http://${WG_SUBNET}.1:8080"
echo ""
echo -e "  ${YELLOW}Desktop Setup:${NC}"
echo -e "  1. Install WireGuard: sudo apt install wireguard"
echo -e "  2. Copy ${CLIENT_DIR}/desktop.conf to your desktop"
echo -e "  3. Import: sudo cp desktop.conf /etc/wireguard/claw.conf"
echo -e "  4. Connect: sudo wg-quick up claw"
echo -e "  5. Access Claw at: http://${WG_SUBNET}.1:8080"
echo ""
