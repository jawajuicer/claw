"""Channel profile resolution for bridge messaging."""

from __future__ import annotations

import logging

from claw.config import ChannelProfile, get_settings

log = logging.getLogger(__name__)


def resolve_profile(platform: str, channel_id: str) -> ChannelProfile:
    """Resolve the effective channel profile for a given platform and channel.

    Resolution order:
    1. bridges.<platform>.channel_profiles[channel_id]
    2. bridges.<platform>.profile
    3. channel_profiles.default_profile
    4. ChannelProfile() defaults
    """
    settings = get_settings()
    cp_cfg = settings.channel_profiles
    profiles = cp_cfg.profiles

    # 1. Check per-channel override on the bridge
    bridge_cfg = getattr(settings.bridges, platform, None)
    if bridge_cfg is not None:
        bridge_channel_profiles = getattr(bridge_cfg, "channel_profiles", {})
        if channel_id in bridge_channel_profiles:
            profile_name = bridge_channel_profiles[channel_id]
            if profile_name in profiles:
                log.debug("Profile for %s/%s: %s (channel override)", platform, channel_id, profile_name)
                return profiles[profile_name]

        # 2. Check bridge-level default profile
        bridge_profile = getattr(bridge_cfg, "profile", "")
        if bridge_profile and bridge_profile in profiles:
            log.debug("Profile for %s/%s: %s (bridge default)", platform, channel_id, bridge_profile)
            return profiles[bridge_profile]

    # 3. Global default profile
    if cp_cfg.default_profile in profiles:
        return profiles[cp_cfg.default_profile]

    # 4. Fallback
    return ChannelProfile()
