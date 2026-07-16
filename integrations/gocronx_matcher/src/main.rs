use tracebook_gocronx_matcher_adapter::server::{EngineIdentity, run};

const ENGINE: EngineIdentity = EngineIdentity {
    name: "gocronx/matcher FIFO adapter",
    version: "0.2.0@b8d48356c8a2",
};

fn main() {
    let exit_code = run(ENGINE);
    if exit_code != 0 {
        std::process::exit(exit_code);
    }
}
