//! Multi-channel audio selection and routing.
//!
//! Channels are 1-based everywhere users see them (CLI flags, config file,
//! error messages) and converted to 0-based indices only after validating
//! against a device's channel count.

use std::fmt;
use std::str::FromStr;

use anyhow::{Result, anyhow, bail, ensure};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

/// Which 1-based audio channel(s) to use on a device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelSelection {
    /// Every channel on the device (`"all"`).
    All,
    /// Specific 1-based channel numbers, sorted ascending and deduplicated.
    /// Non-empty; every value is `>= 1`.
    Channels(Vec<u16>),
}

impl ChannelSelection {
    /// Resolve to 0-based channel indices for a device with `device_channels`
    /// channels. Errors if any requested channel is out of range.
    pub fn resolve(&self, device_channels: u16) -> Result<Vec<usize>> {
        ensure!(device_channels > 0, "device reports 0 audio channels");
        match self {
            Self::All => Ok((0..device_channels as usize).collect()),
            Self::Channels(list) => {
                for &ch in list {
                    if ch == 0 || ch > device_channels {
                        bail!(
                            "channel {ch} is out of range for device with {device_channels} channel(s) (valid: 1..={device_channels})"
                        );
                    }
                }
                Ok(list.iter().map(|&ch| (ch - 1) as usize).collect())
            }
        }
    }

    /// Encode as a `toml_edit::Value` for persisting into `config.toml`:
    /// `"all"` as a string, or specific channels as an integer array `[3, 4]`.
    pub fn to_toml_value(&self) -> toml_edit::Value {
        match self {
            Self::All => toml_edit::Value::from("all"),
            Self::Channels(list) => {
                let mut arr = toml_edit::Array::new();
                for &ch in list {
                    arr.push(i64::from(ch));
                }
                toml_edit::Value::Array(arr)
            }
        }
    }
}

impl fmt::Display for ChannelSelection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => f.write_str("all"),
            Self::Channels(list) => {
                for (i, ch) in list.iter().enumerate() {
                    if i > 0 {
                        f.write_str(",")?;
                    }
                    write!(f, "{ch}")?;
                }
                Ok(())
            }
        }
    }
}

impl FromStr for ChannelSelection {
    type Err = anyhow::Error;

    /// Parse a 1-based channel specification:
    /// - `"all"` (case-insensitive)
    /// - single channel: `"3"`
    /// - comma-separated list: `"3,4"`
    /// - inclusive range: `"3-4"`
    /// - combined: `"1,3-4"`
    fn from_str(s: &str) -> Result<Self> {
        let trimmed = s.trim();
        ensure!(!trimmed.is_empty(), "empty channel specification");
        if trimmed.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }

        let mut channels = Vec::new();
        for part in trimmed.split(',') {
            let part = part.trim();
            ensure!(
                !part.is_empty(),
                "empty entry in channel specification {s:?}"
            );
            if let Some((start_s, end_s)) = part.split_once('-') {
                ensure!(
                    !end_s.contains('-'),
                    "invalid channel range {part:?} in {s:?}"
                );
                let start = parse_channel(start_s.trim(), s)?;
                let end = parse_channel(end_s.trim(), s)?;
                ensure!(
                    start <= end,
                    "channel range {start}-{end} is backwards in {s:?}"
                );
                channels.extend(start..=end);
            } else {
                channels.push(parse_channel(part, s)?);
            }
        }

        channels.sort_unstable();
        channels.dedup();
        Ok(Self::Channels(channels))
    }
}

fn parse_channel(token: &str, whole: &str) -> Result<u16> {
    let n: u16 = token
        .parse()
        .map_err(|_| anyhow!("invalid channel {token:?} in {whole:?} (expected 1-based number)"))?;
    ensure!(n >= 1, "channels are 1-based; got 0 in {whole:?}");
    Ok(n)
}

impl Serialize for ChannelSelection {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::All => serializer.serialize_str("all"),
            Self::Channels(list) => list.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for ChannelSelection {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;

        impl<'de> de::Visitor<'de> for Visitor {
            type Value = ChannelSelection;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a channel spec string (e.g. \"3-4\", \"all\") or an array of 1-based channel numbers (e.g. [3, 4])")
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                v.parse().map_err(E::custom)
            }

            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut channels = Vec::new();
                while let Some(n) = seq.next_element::<u16>()? {
                    if n == 0 {
                        return Err(de::Error::custom("channels are 1-based; got 0"));
                    }
                    channels.push(n);
                }
                if channels.is_empty() {
                    return Err(de::Error::custom("channel list cannot be empty"));
                }
                channels.sort_unstable();
                channels.dedup();
                Ok(ChannelSelection::Channels(channels))
            }
        }

        deserializer.deserialize_any(Visitor)
    }
}

/// Expand mono samples into an interleaved `device_channels`-channel buffer,
/// placing each sample onto every 0-based index in `target_channels` and `0.0`
/// onto all other channels.
pub fn route_mono_to_channels(
    mono: &[f32],
    device_channels: u16,
    target_channels: &[usize],
) -> Vec<f32> {
    let ch = device_channels as usize;
    let mut out = vec![0.0f32; mono.len() * ch];
    for (frame_idx, &sample) in mono.iter().enumerate() {
        let base = frame_idx * ch;
        for &c in target_channels {
            out[base + c] = sample;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_list_range_combined_and_all() {
        assert_eq!(
            "3".parse::<ChannelSelection>().unwrap(),
            ChannelSelection::Channels(vec![3])
        );
        assert_eq!(
            "3,4".parse::<ChannelSelection>().unwrap(),
            ChannelSelection::Channels(vec![3, 4])
        );
        assert_eq!(
            "3-4".parse::<ChannelSelection>().unwrap(),
            ChannelSelection::Channels(vec![3, 4])
        );
        assert_eq!(
            "1, 3-5, 3".parse::<ChannelSelection>().unwrap(),
            ChannelSelection::Channels(vec![1, 3, 4, 5])
        );
        assert_eq!(
            "4,2".parse::<ChannelSelection>().unwrap(),
            ChannelSelection::Channels(vec![2, 4])
        );
        assert_eq!(
            "all".parse::<ChannelSelection>().unwrap(),
            ChannelSelection::All
        );
        assert_eq!(
            "ALL".parse::<ChannelSelection>().unwrap(),
            ChannelSelection::All
        );
    }

    #[test]
    fn rejects_invalid_specs() {
        for bad in [
            "", " ", "0", "0-2", "3-2", "1,", ",1", "1-2-3", "abc", "1,foo",
        ] {
            assert!(
                bad.parse::<ChannelSelection>().is_err(),
                "expected {bad:?} to fail"
            );
        }
    }

    #[test]
    fn resolves_and_bounds_checks() {
        let sel: ChannelSelection = "3,4".parse().unwrap();
        assert_eq!(sel.resolve(4).unwrap(), vec![2, 3]);
        assert_eq!(sel.resolve(16).unwrap(), vec![2, 3]);
        assert!(sel.resolve(2).is_err());
        assert!(sel.resolve(3).is_err());

        assert_eq!(ChannelSelection::All.resolve(4).unwrap(), vec![0, 1, 2, 3]);
        assert!(ChannelSelection::All.resolve(0).is_err());
    }

    #[test]
    fn toml_deserializes_array_and_string() {
        #[derive(Deserialize)]
        struct Wrap {
            ch: ChannelSelection,
        }
        let from_arr: Wrap = toml::from_str("ch = [3, 4]").unwrap();
        assert_eq!(from_arr.ch, ChannelSelection::Channels(vec![3, 4]));

        let from_str: Wrap = toml::from_str("ch = \"3-4\"").unwrap();
        assert_eq!(from_str.ch, ChannelSelection::Channels(vec![3, 4]));

        let from_all: Wrap = toml::from_str("ch = \"all\"").unwrap();
        assert_eq!(from_all.ch, ChannelSelection::All);

        assert!(toml::from_str::<Wrap>("ch = []").is_err());
        assert!(toml::from_str::<Wrap>("ch = [0, 1]").is_err());
    }

    #[test]
    fn routes_mono_to_selected_channels_only() {
        let out = route_mono_to_channels(&[0.5, -0.25], 4, &[2, 3]);
        assert_eq!(out, vec![0.0, 0.0, 0.5, 0.5, 0.0, 0.0, -0.25, -0.25]);
    }
}
