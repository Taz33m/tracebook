mod adapter;
mod wire;

use crate::adapter::Adapter;
use crate::wire::{
    EngineMetadata, EventFrame, FinishFrame, FrameType, HelloFrame, PROTOCOL_NAME,
    PROTOCOL_VERSION, ReadyFrame, SnapshotRequest, write_frame,
};
use serde::Serialize;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufReader, BufWriter};

#[derive(Debug)]
enum ServerError {
    Protocol(String),
    Adapter(String),
    BrokenPipe,
}

fn main() {
    let drop_first_trade = match test_fault_from_args() {
        Ok(value) => value,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut input = BufReader::new(stdin.lock());
    let mut output = BufWriter::new(stdout.lock());

    let exit_code = match serve(&mut input, &mut output, drop_first_trade) {
        Ok(()) => 0,
        Err(ServerError::BrokenPipe) => 2,
        Err(error) => {
            let (code, message) = match error {
                ServerError::Protocol(message) => ("PROTOCOL_ERROR", message),
                ServerError::Adapter(message) => ("ADAPTER_ERROR", message),
                ServerError::BrokenPipe => unreachable!(),
            };
            let frame = json!({"type": "error", "code": code, "message": message});
            let _ = write_frame(&mut output, &frame);
            2
        }
    };
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn serve(
    input: &mut impl BufRead,
    output: &mut impl io::Write,
    drop_first_trade: bool,
) -> Result<(), ServerError> {
    let hello_value = read_required_frame(input, "expected hello message")?;
    let hello: HelloFrame = parse_frame(hello_value)?;
    if hello.kind != "hello" {
        return Err(protocol("first message must be hello"));
    }
    if hello.protocol != PROTOCOL_NAME {
        return Err(protocol(format!("protocol must be {PROTOCOL_NAME:?}")));
    }
    if hello.protocol_version != PROTOCOL_VERSION {
        return Err(protocol(format!(
            "protocol_version must be {PROTOCOL_VERSION}"
        )));
    }

    let mut adapter = if drop_first_trade {
        Adapter::new_with_test_fault(hello.config, true)
    } else {
        Adapter::new(hello.config)
    }
    .map_err(ServerError::Protocol)?;
    send(
        output,
        &ReadyFrame {
            kind: "ready",
            protocol: PROTOCOL_NAME,
            protocol_version: PROTOCOL_VERSION,
            engine: EngineMetadata {
                name: "orderbook-rs FIFO adapter",
                version: "0.10.4",
                language: "Rust",
            },
        },
    )?;

    let mut last_index = 0_u64;
    loop {
        let value = read_required_frame(input, "protocol ended before finish")?;
        let frame_type: FrameType = parse_frame(value.clone())?;
        match frame_type.kind.as_str() {
            "event" => {
                let frame: EventFrame = parse_frame(value)?;
                if frame.index == 0 {
                    return Err(protocol("event index must be a positive integer"));
                }
                if frame.index != last_index + 1 {
                    return Err(protocol("event indexes must be contiguous and start at 1"));
                }
                let observation = adapter
                    .apply(&frame.event, frame.index)
                    .map_err(ServerError::Adapter)?;
                last_index = frame.index;
                send(output, &observation)?;
            }
            "snapshot" => {
                let frame: SnapshotRequest = parse_frame(value)?;
                if frame.index != last_index {
                    return Err(protocol("snapshot index does not match the last event"));
                }
                let state = adapter.snapshot().map_err(ServerError::Adapter)?;
                send(
                    output,
                    &json!({"type": "snapshot", "index": last_index, "state": state}),
                )?;
            }
            "finish" => {
                let frame: FinishFrame = parse_frame(value)?;
                if frame.event_count != last_index {
                    return Err(protocol("finish event_count does not match the last event"));
                }
                send(
                    output,
                    &json!({"type": "complete", "event_count": last_index}),
                )?;
                return Ok(());
            }
            other => {
                return Err(protocol(format!(
                    "unsupported protocol message type: {other:?}"
                )));
            }
        }
    }
}

fn read_required_frame(input: &mut impl BufRead, eof_message: &str) -> Result<Value, ServerError> {
    loop {
        let mut line = String::new();
        let count = input
            .read_line(&mut line)
            .map_err(|error| protocol(error.to_string()))?;
        if count == 0 {
            return Err(protocol(eof_message));
        }
        if line.trim().is_empty() {
            continue;
        }
        return serde_json::from_str(&line).map_err(|error| protocol(error.to_string()));
    }
}

fn parse_frame<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, ServerError> {
    if !value.is_object() {
        return Err(protocol("protocol message must be an object"));
    }
    serde_json::from_value(value).map_err(|error| protocol(error.to_string()))
}

fn send(output: &mut impl io::Write, value: &impl Serialize) -> Result<(), ServerError> {
    write_frame(output, value).map_err(|error| {
        if error.kind() == io::ErrorKind::BrokenPipe {
            ServerError::BrokenPipe
        } else {
            ServerError::Adapter(error.to_string())
        }
    })
}

fn protocol(message: impl Into<String>) -> ServerError {
    ServerError::Protocol(message.into())
}

fn test_fault_from_args() -> Result<bool, String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    parse_test_fault(&args)
}

fn parse_test_fault(args: &[String]) -> Result<bool, String> {
    match args {
        [] => Ok(false),
        [value] if value == "--test-fault=drop-first-trade" => Ok(true),
        _ => Err("usage: tracebook-orderbook-rs [--test-fault=drop-first-trade]".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn command_line_fault_mode_is_named_explicitly() {
        assert!(!parse_test_fault(&[]).unwrap());
        assert!(parse_test_fault(&["--test-fault=drop-first-trade".to_string()]).unwrap());
        assert!(parse_test_fault(&["--fault".to_string()]).is_err());
    }

    #[test]
    fn empty_session_completes_with_protocol_defaults() {
        let frames = concat!(
            "{\"type\":\"hello\",\"protocol\":\"tracebook.conformance\",",
            "\"protocol_version\":1}\n",
            "{\"type\":\"finish\",\"event_count\":0}\n"
        );
        let mut input = Cursor::new(frames.as_bytes());
        let mut output = Vec::new();

        serve(&mut input, &mut output, false).unwrap();

        let values = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0]["type"], "ready");
        assert_eq!(values[0]["engine"]["version"], "0.10.4");
        assert_eq!(values[1], json!({"type": "complete", "event_count": 0}));
    }
}
