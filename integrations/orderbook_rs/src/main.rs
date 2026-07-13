use tracebook_orderbook_rs_adapter::adapter::FaultMode;
use tracebook_orderbook_rs_adapter::server::{EngineIdentity, run};

const ENGINE: EngineIdentity = EngineIdentity {
    name: "orderbook-rs FIFO adapter",
    version: "0.10.4",
};

fn main() {
    let fault_mode = match test_fault_from_args() {
        Ok(value) => value,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    };
    let exit_code = run(fault_mode, ENGINE);
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}

fn test_fault_from_args() -> Result<FaultMode, String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    parse_test_fault(&args)
}

fn parse_test_fault(args: &[String]) -> Result<FaultMode, String> {
    match args {
        [] => Ok(FaultMode::None),
        [value] if value == "--test-fault=drop-first-trade" => Ok(FaultMode::DropFirstTrade),
        _ => Err("usage: tracebook-orderbook-rs [--test-fault=drop-first-trade]".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_line_fault_mode_is_named_explicitly() {
        assert_eq!(parse_test_fault(&[]).unwrap(), FaultMode::None);
        assert_eq!(
            parse_test_fault(&["--test-fault=drop-first-trade".to_string()]).unwrap(),
            FaultMode::DropFirstTrade
        );
        assert!(parse_test_fault(&["--fault".to_string()]).is_err());
    }
}
