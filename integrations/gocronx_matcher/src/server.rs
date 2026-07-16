use crate::adapter::Adapter;
use tracebook_conformance_protocol::{
    BookState, ConfigWire, EngineAdapter, MarketEvent, ObservationFrame,
};

pub use tracebook_conformance_protocol::EngineIdentity;

impl EngineAdapter for Adapter {
    fn apply(&mut self, event: &MarketEvent, index: u64) -> Result<ObservationFrame, String> {
        Adapter::apply(self, event, index)
    }

    fn snapshot(&self) -> Result<BookState, String> {
        Adapter::snapshot(self)
    }
}

pub fn run(identity: EngineIdentity) -> i32 {
    tracebook_conformance_protocol::run(identity, |config: ConfigWire| Adapter::new(config))
}
