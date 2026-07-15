use tracebook_orderbook_rs_adapter::adapter::FaultMode;
use tracebook_orderbook_rs_adapter::server::{EngineIdentity, run};

const ENGINE: EngineIdentity = EngineIdentity {
    name: "historical orderbook-rs issue 88 adapter",
    version: "0.8.0@53b4d2b0+pricelevel-0.7.0",
};

fn main() {
    let exit_code = run(FaultMode::None, ENGINE);
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}
