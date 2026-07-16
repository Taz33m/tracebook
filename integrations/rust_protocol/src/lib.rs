use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use sha2::{Digest, Sha256};
use std::io::{self, Write};

mod server;

pub use server::{EngineAdapter, EngineIdentity, run};

pub const PROTOCOL_NAME: &str = "tracebook.conformance";
pub const PROTOCOL_VERSION: u64 = 1;

#[derive(Debug, Deserialize)]
pub struct FrameType {
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Deserialize)]
pub struct HelloFrame {
    #[serde(rename = "type")]
    pub kind: String,
    pub protocol: String,
    pub protocol_version: u64,
    #[serde(default)]
    pub config: ConfigWire,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConfigWire {
    #[serde(default = "default_matching_algorithm")]
    pub matching_algorithm: String,
    #[serde(default = "default_tick_size")]
    pub tick_size: String,
    #[serde(default = "default_self_trade_policy")]
    pub self_trade_policy: String,
    #[serde(default = "default_quantity_decimal_places")]
    pub quantity_decimal_places: u32,
}

impl Default for ConfigWire {
    fn default() -> Self {
        Self {
            matching_algorithm: default_matching_algorithm(),
            tick_size: default_tick_size(),
            self_trade_policy: default_self_trade_policy(),
            quantity_decimal_places: default_quantity_decimal_places(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EventFrame {
    pub index: u64,
    pub event: MarketEvent,
}

#[derive(Debug, Deserialize)]
pub struct SnapshotRequest {
    pub index: u64,
}

#[derive(Debug, Deserialize)]
pub struct FinishFrame {
    pub event_count: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarketEvent {
    pub op: String,
    pub symbol: String,
    pub order_id: Option<Number>,
    pub side: Option<String>,
    #[serde(default = "default_order_type")]
    pub order_type: String,
    pub price: Option<Number>,
    pub quantity: Option<Number>,
    #[serde(default = "default_owner")]
    pub owner: i64,
    pub timestamp_ns: Option<u64>,
}

fn default_matching_algorithm() -> String {
    "fifo".to_string()
}

fn default_tick_size() -> String {
    "0.01".to_string()
}

fn default_self_trade_policy() -> String {
    "NONE".to_string()
}

const fn default_quantity_decimal_places() -> u32 {
    12
}

fn default_order_type() -> String {
    "LIMIT".to_string()
}

const fn default_owner() -> i64 {
    -1
}

#[derive(Debug, Serialize)]
pub struct ReadyFrame {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub protocol: &'static str,
    pub protocol_version: u64,
    pub engine: EngineMetadata,
}

#[derive(Debug, Serialize)]
pub struct EngineMetadata {
    pub name: &'static str,
    pub version: &'static str,
    pub language: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct Outcome {
    pub status: &'static str,
    pub reason: Option<&'static str>,
    pub message: Option<String>,
}

impl Outcome {
    pub fn applied() -> Self {
        Self {
            status: "applied",
            reason: None,
            message: None,
        }
    }

    pub fn rejected(reason: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: "rejected",
            reason: Some(reason),
            message: Some(message.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TradeFill {
    pub symbol: String,
    pub buy_order_id: u64,
    pub sell_order_id: u64,
    pub price: String,
    pub quantity: String,
}

#[derive(Debug, Serialize)]
pub struct ObservationFrame {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub index: u64,
    pub outcome: Outcome,
    pub trades: Vec<TradeFill>,
    pub state_hash: String,
    pub resting_order_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RestingOrder {
    pub order_id: u64,
    pub price: String,
    pub remaining_quantity: String,
    pub owner: i64,
    pub order_type: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct BookSnapshot {
    pub symbol: String,
    pub bids: Vec<RestingOrder>,
    pub asks: Vec<RestingOrder>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BookState {
    pub books: Vec<BookSnapshot>,
}

impl BookState {
    pub fn order_count(&self) -> usize {
        self.books
            .iter()
            .map(|book| book.bids.len() + book.asks.len())
            .sum()
    }

    pub fn digest(&self) -> Result<String, String> {
        let value = serde_json::to_value(self)
            .map_err(|error| format!("could not serialize canonical state: {error}"))?;
        let payload = canonical_json(&value)?;
        let digest = Sha256::digest(payload.as_bytes());
        Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
    }
}

pub fn canonical_json(value: &Value) -> Result<String, String> {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_string(value).map_err(|error| error.to_string())
        }
        Value::Array(values) => {
            let encoded = values
                .iter()
                .map(canonical_json)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(format!("[{}]", encoded.join(",")))
        }
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let encoded = keys
                .into_iter()
                .map(|key| {
                    let encoded_key =
                        serde_json::to_string(key).map_err(|error| error.to_string())?;
                    let encoded_value = canonical_json(&values[key])?;
                    Ok(format!("{encoded_key}:{encoded_value}"))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(format!("{{{}}}", encoded.join(",")))
        }
    }
}

pub fn write_frame(output: &mut impl Write, value: &impl Serialize) -> io::Result<()> {
    serde_json::to_writer(&mut *output, value)?;
    output.write_all(b"\n")?;
    output.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_every_object_level() {
        let value = serde_json::json!({"z": [{"b": 2, "a": 1}], "a": "BTC-USD"});
        assert_eq!(
            canonical_json(&value).unwrap(),
            r#"{"a":"BTC-USD","z":[{"a":1,"b":2}]}"#
        );
    }

    #[test]
    fn empty_state_hash_matches_tracebook() {
        let state = BookState { books: Vec::new() };
        assert_eq!(
            state.digest().unwrap(),
            "dd8681e973bdf802edf26260297e68e24cdbd75c782c90bbaaaddd401df51090"
        );
    }

    #[test]
    fn omitted_config_uses_protocol_defaults() {
        let hello: HelloFrame = serde_json::from_value(serde_json::json!({
            "type": "hello",
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION
        }))
        .unwrap();
        assert_eq!(hello.config.matching_algorithm, "fifo");
        assert_eq!(hello.config.tick_size, "0.01");
        assert_eq!(hello.config.self_trade_policy, "NONE");
        assert_eq!(hello.config.quantity_decimal_places, 12);
    }
}
