use tracebook_orderbook_rs_adapter::adapter::FaultMode;
use tracebook_orderbook_rs_adapter::server::{EngineIdentity, run};

const ENGINE: EngineIdentity = EngineIdentity {
    name: "intentionally faulty orderbook-rs queue-priority example",
    version: "0.11.0+queue-priority-fault",
};

fn main() {
    let exit_code = run(FaultMode::QueuePriority, ENGINE);
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}
