use crate::wire::{
    BookSnapshot, BookState, ConfigWire, MarketEvent, ObservationFrame, Outcome, RestingOrder,
    TradeFill,
};
use matcher::{BookEvent, Order, OrderBook, OrderId, Quantity, RejectReason, Side, Trade};
use rust_decimal::prelude::ToPrimitive;
use rust_decimal::{Decimal, RoundingStrategy};
use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

const SNAP_MAGIC: &[u8; 8] = b"MATCHER\x01";
const SNAP_VERSION: u32 = 1;
const SNAP_HEADER_LEN: usize = 20;
const SNAP_RECORD_LEN: usize = 49;
const KIND_ICEBERG: u8 = 6;

#[derive(Clone)]
struct AdapterConfig {
    tick_size: Decimal,
    tick_size_f64: f64,
    quantity_decimal_places: u32,
    quantity_scale: u64,
}

impl AdapterConfig {
    fn from_wire(config: ConfigWire) -> Result<Self, String> {
        if config.matching_algorithm != "fifo" {
            return Err("matching_algorithm must be 'fifo'".to_string());
        }
        if config.self_trade_policy != "NONE" {
            return Err("self_trade_policy must be NONE".to_string());
        }
        if config.quantity_decimal_places > 18 {
            return Err("quantity_decimal_places must be between 0 and 18".to_string());
        }

        let tick_size = Decimal::from_str(&config.tick_size)
            .map_err(|_| "tick_size must be a positive finite decimal".to_string())?;
        if tick_size <= Decimal::ZERO {
            return Err("tick_size must be a positive finite decimal".to_string());
        }
        let tick_size_f64 = config
            .tick_size
            .parse::<f64>()
            .map_err(|_| "tick_size must be a positive finite decimal".to_string())?;
        if !tick_size_f64.is_finite() || tick_size_f64 <= 0.0 {
            return Err("tick_size must be a positive finite decimal".to_string());
        }
        let quantity_scale = 10_u64
            .checked_pow(config.quantity_decimal_places)
            .ok_or_else(|| "quantity scale exceeds u64".to_string())?;

        Ok(Self {
            tick_size,
            tick_size_f64,
            quantity_decimal_places: config.quantity_decimal_places,
            quantity_scale,
        })
    }

    fn price_ticks(&self, value: &serde_json::Number) -> Result<u64, String> {
        let price = value
            .to_string()
            .parse::<f64>()
            .map_err(|_| "price must be finite".to_string())?;
        if !price.is_finite() {
            return Err("price must be finite".to_string());
        }
        let ticks = (price / self.tick_size_f64).round_ties_even();
        if !ticks.is_finite() || ticks <= 0.0 || ticks >= u64::MAX as f64 {
            return Err("price snaps to a non-positive or unsupported tick".to_string());
        }
        Ok(ticks as u64)
    }

    fn quantity_units(&self, value: &serde_json::Number) -> Result<u64, String> {
        let quantity = Decimal::from_str(&value.to_string())
            .map_err(|_| "quantity must be a finite decimal".to_string())?;
        if quantity <= Decimal::ZERO {
            return Err("quantity must be positive".to_string());
        }
        let normalized = quantity.round_dp_with_strategy(
            self.quantity_decimal_places,
            RoundingStrategy::MidpointNearestEven,
        );
        let scaled = normalized
            .checked_mul(Decimal::from(self.quantity_scale))
            .ok_or_else(|| "quantity exceeds the adapter's integer range".to_string())?;
        let units = scaled
            .to_u64()
            .ok_or_else(|| "quantity exceeds the adapter's integer range".to_string())?;
        if units == 0 {
            return Err("quantity rounds to zero at the configured precision".to_string());
        }
        Ok(units)
    }

    fn format_price(&self, ticks: u64) -> Result<String, String> {
        let price = Decimal::from(ticks)
            .checked_mul(self.tick_size)
            .ok_or_else(|| "price exceeds decimal range".to_string())?;
        Ok(canonical_decimal(price))
    }

    fn format_quantity(&self, units: u64) -> String {
        canonical_decimal(Decimal::from(units) / Decimal::from(self.quantity_scale))
    }
}

struct BookHarness {
    book: OrderBook,
    owners: BTreeMap<u64, i64>,
}

impl BookHarness {
    fn new() -> Self {
        Self {
            book: OrderBook::new(),
            owners: BTreeMap::new(),
        }
    }

    fn native_order(&self, order_id: u64) -> Result<Option<NativeResting>, String> {
        Ok(decode_snapshot(&self.book.snapshot())?
            .into_iter()
            .find(|order| order.order_id == order_id))
    }

    fn reconcile_owners(&mut self) -> Result<(), String> {
        let active = decode_snapshot(&self.book.snapshot())?
            .into_iter()
            .map(|order| order.order_id)
            .collect::<BTreeSet<_>>();
        self.owners.retain(|order_id, _| active.contains(order_id));
        Ok(())
    }
}

struct ApplyResult {
    outcome: Outcome,
    trades: Vec<Trade>,
}

impl ApplyResult {
    fn applied(trades: Vec<Trade>) -> Self {
        Self {
            outcome: Outcome::applied(),
            trades,
        }
    }

    fn rejected(reason: &'static str, message: impl Into<String>) -> Self {
        Self {
            outcome: Outcome::rejected(reason, message),
            trades: Vec::new(),
        }
    }
}

pub struct Adapter {
    config: AdapterConfig,
    books: BTreeMap<String, BookHarness>,
}

impl Adapter {
    pub fn new(config: ConfigWire) -> Result<Self, String> {
        Ok(Self {
            config: AdapterConfig::from_wire(config)?,
            books: BTreeMap::new(),
        })
    }

    pub fn apply(&mut self, event: &MarketEvent, index: u64) -> Result<ObservationFrame, String> {
        let result = if event.op == "clear" {
            self.books.insert(event.symbol.clone(), BookHarness::new());
            ApplyResult::applied(Vec::new())
        } else if !self.books.contains_key(&event.symbol)
            && matches!(event.op.as_str(), "cancel" | "reduce" | "replace")
        {
            ApplyResult::rejected("ORDER_NOT_ACTIVE", "order is not active")
        } else {
            let book = self
                .books
                .entry(event.symbol.clone())
                .or_insert_with(BookHarness::new);
            let result = Self::apply_to_book(&self.config, book, event, index)?;
            book.reconcile_owners()?;
            result
        };

        let trades = result
            .trades
            .into_iter()
            .map(|trade| self.convert_trade(&event.symbol, trade))
            .collect::<Result<Vec<_>, _>>()?;
        let state = self.snapshot()?;
        Ok(ObservationFrame {
            kind: "observation",
            index,
            outcome: result.outcome,
            trades,
            state_hash: state.digest()?,
            resting_order_count: state.order_count(),
        })
    }

    fn apply_to_book(
        config: &AdapterConfig,
        book: &mut BookHarness,
        event: &MarketEvent,
        index: u64,
    ) -> Result<ApplyResult, String> {
        match event.op.as_str() {
            "new" => Ok(Self::apply_new(config, book, event, index)),
            "cancel" => Ok(Self::apply_cancel(book, event, index)),
            "reduce" => Self::apply_reduce(config, book, event, index),
            "replace" => Self::apply_replace(config, book, event, index),
            _ => Ok(ApplyResult::rejected(
                "INVALID_ORDER",
                "unsupported event operation",
            )),
        }
    }

    fn apply_new(
        config: &AdapterConfig,
        book: &mut BookHarness,
        event: &MarketEvent,
        index: u64,
    ) -> ApplyResult {
        let order_id = match parse_order_id(event) {
            Ok(value) => value,
            Err(message) => return ApplyResult::rejected("INVALID_ORDER", message),
        };
        if book.owners.contains_key(&order_id) {
            return ApplyResult::rejected("DUPLICATE_ORDER_ID", "order is already active");
        }
        let side = match parse_side(event) {
            Ok(value) => value,
            Err(message) => return ApplyResult::rejected("INVALID_ORDER", message),
        };
        let quantity = match required_quantity(config, event, "quantity is required") {
            Ok(value) => value,
            Err(message) => return ApplyResult::rejected("INVALID_ORDER", message),
        };
        let price = || {
            event
                .price
                .as_ref()
                .ok_or_else(|| "price is required".to_string())
                .and_then(|value| config.price_ticks(value))
        };
        let order = match event.order_type.as_str() {
            "MARKET" => Order::market(order_id, side, quantity),
            "LIMIT" => match price() {
                Ok(value) => Order::limit(order_id, side, value, quantity),
                Err(message) => return ApplyResult::rejected("INVALID_ORDER", message),
            },
            "IOC" => match price() {
                Ok(value) => Order::ioc(order_id, side, value, quantity),
                Err(message) => return ApplyResult::rejected("INVALID_ORDER", message),
            },
            "FOK" => match price() {
                Ok(value) => Order::fok(order_id, side, value, quantity),
                Err(message) => return ApplyResult::rejected("INVALID_ORDER", message),
            },
            _ => return ApplyResult::rejected("INVALID_ORDER", "unsupported order_type"),
        };
        let events = book.book.submit_events(order, event_time(event, index));
        Self::submission_result(book, event, order_id, events)
    }

    fn submission_result(
        book: &mut BookHarness,
        event: &MarketEvent,
        order_id: u64,
        events: Vec<BookEvent>,
    ) -> ApplyResult {
        if let Some(reason) = events.iter().find_map(|event| match event {
            BookEvent::Rejected { reason, .. } => Some(*reason),
            _ => None,
        }) {
            return match reason {
                RejectReason::DuplicateOrderId => {
                    ApplyResult::rejected("DUPLICATE_ORDER_ID", "order is already active")
                }
                RejectReason::FokNotFillable => ApplyResult::applied(Vec::new()),
                _ => ApplyResult::rejected("INVALID_ORDER", format!("{reason:?}")),
            };
        }
        if events.iter().any(
            |item| matches!(item, BookEvent::Rested { order_id: id, .. } if *id == OrderId(order_id)),
        ) {
            book.owners.insert(order_id, event.owner);
        }
        ApplyResult::applied(trades_from_events(&events))
    }

    fn apply_cancel(book: &mut BookHarness, event: &MarketEvent, index: u64) -> ApplyResult {
        let order_id = match parse_order_id(event) {
            Ok(value) => value,
            Err(message) => return ApplyResult::rejected("INVALID_CANCEL", message),
        };
        let events = book.book.cancel_events(order_id, event_time(event, index));
        if events
            .iter()
            .any(|item| matches!(item, BookEvent::Canceled { .. }))
        {
            book.owners.remove(&order_id);
            ApplyResult::applied(Vec::new())
        } else {
            ApplyResult::rejected("ORDER_NOT_ACTIVE", "order is not active")
        }
    }

    fn apply_reduce(
        config: &AdapterConfig,
        book: &mut BookHarness,
        event: &MarketEvent,
        index: u64,
    ) -> Result<ApplyResult, String> {
        let order_id = match parse_order_id(event) {
            Ok(value) => value,
            Err(message) => return Ok(ApplyResult::rejected("INVALID_ORDER", message)),
        };
        let Some(existing) = book.native_order(order_id)? else {
            return Ok(ApplyResult::rejected(
                "ORDER_NOT_ACTIVE",
                "order is not active",
            ));
        };
        let reduction = match required_quantity(config, event, "reduction quantity is required") {
            Ok(value) => value,
            Err(message) => return Ok(ApplyResult::rejected("INVALID_ORDER", message)),
        };
        if reduction > existing.remaining {
            return Ok(ApplyResult::rejected(
                "INVALID_ORDER",
                "reduction quantity cannot exceed remaining quantity",
            ));
        }
        if reduction == existing.remaining {
            return Ok(Self::apply_cancel(book, event, index));
        }
        let events = book.book.amend(
            order_id,
            None,
            Some(Quantity(existing.remaining - reduction)),
        );
        if events
            .iter()
            .any(|item| matches!(item, BookEvent::Amended { .. }))
        {
            Ok(ApplyResult::applied(Vec::new()))
        } else {
            Err("matcher rejected a prevalidated quantity reduction".to_string())
        }
    }

    fn apply_replace(
        config: &AdapterConfig,
        book: &mut BookHarness,
        event: &MarketEvent,
        index: u64,
    ) -> Result<ApplyResult, String> {
        let order_id = match parse_order_id(event) {
            Ok(value) => value,
            Err(message) => return Ok(ApplyResult::rejected("INVALID_REPLACEMENT", message)),
        };
        let Some(existing) = book.native_order(order_id)? else {
            return Ok(ApplyResult::rejected(
                "ORDER_NOT_ACTIVE",
                "order is not active",
            ));
        };
        let price = match event.price.as_ref() {
            Some(value) => match config.price_ticks(value) {
                Ok(value) => value,
                Err(message) => {
                    return Ok(ApplyResult::rejected("INVALID_REPLACEMENT", message));
                }
            },
            None => existing.price,
        };
        let quantity = match event.quantity.as_ref() {
            Some(value) => match config.quantity_units(value) {
                Ok(value) => value,
                Err(message) => {
                    return Ok(ApplyResult::rejected("INVALID_REPLACEMENT", message));
                }
            },
            None => existing.remaining,
        };

        if !book.book.cancel(order_id) {
            return Err("matcher lost an active order before replacement".to_string());
        }
        let events = book.book.submit_events(
            Order::limit(order_id, existing.side, price, quantity),
            event_time(event, index),
        );
        if events
            .iter()
            .any(|item| matches!(item, BookEvent::Rejected { .. }))
        {
            return Err(
                "matcher rejected a prevalidated cancel-and-submit replacement".to_string(),
            );
        }
        Ok(ApplyResult::applied(trades_from_events(&events)))
    }

    fn convert_trade(&self, symbol: &str, trade: Trade) -> Result<TradeFill, String> {
        Ok(TradeFill {
            symbol: symbol.to_string(),
            buy_order_id: trade.buy_id.get(),
            sell_order_id: trade.sell_id.get(),
            price: self.config.format_price(trade.price.get())?,
            quantity: self.config.format_quantity(trade.quantity.get()),
        })
    }

    pub fn snapshot(&self) -> Result<BookState, String> {
        let books = self
            .books
            .iter()
            .map(|(symbol, harness)| self.book_snapshot(symbol, harness))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(BookState { books })
    }

    fn book_snapshot(&self, symbol: &str, harness: &BookHarness) -> Result<BookSnapshot, String> {
        let (mut bids, mut asks): (Vec<_>, Vec<_>) = decode_snapshot(&harness.book.snapshot())?
            .into_iter()
            .partition(|order| order.side == Side::Buy);
        bids.sort_by(|left, right| right.price.cmp(&left.price));
        asks.sort_by(|left, right| left.price.cmp(&right.price));
        Ok(BookSnapshot {
            symbol: symbol.to_string(),
            bids: self.resting_orders(bids, &harness.owners)?,
            asks: self.resting_orders(asks, &harness.owners)?,
        })
    }

    fn resting_orders(
        &self,
        orders: Vec<NativeResting>,
        owners: &BTreeMap<u64, i64>,
    ) -> Result<Vec<RestingOrder>, String> {
        orders
            .into_iter()
            .map(|order| {
                let owner = owners.get(&order.order_id).copied().ok_or_else(|| {
                    format!(
                        "adapter has no source owner for resting order {}",
                        order.order_id
                    )
                })?;
                Ok(RestingOrder {
                    order_id: order.order_id,
                    price: self.config.format_price(order.price)?,
                    remaining_quantity: self.config.format_quantity(order.remaining),
                    owner,
                    order_type: "LIMIT",
                })
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct NativeResting {
    order_id: u64,
    side: Side,
    price: u64,
    remaining: u64,
}

fn decode_snapshot(bytes: &[u8]) -> Result<Vec<NativeResting>, String> {
    if bytes.len() < SNAP_HEADER_LEN {
        return Err("matcher snapshot is truncated".to_string());
    }
    if &bytes[..8] != SNAP_MAGIC {
        return Err("matcher snapshot magic changed".to_string());
    }
    let version = read_u32(bytes, 8)?;
    if version != SNAP_VERSION {
        return Err(format!("unsupported matcher snapshot version {version}"));
    }
    let count = read_u64(bytes, 12)? as usize;
    let mut orders = Vec::with_capacity(count);
    let mut offset = SNAP_HEADER_LEN;
    for _ in 0..count {
        if bytes.len().saturating_sub(offset) < SNAP_RECORD_LEN {
            return Err("matcher snapshot record is truncated".to_string());
        }
        let side = match bytes[offset] {
            1 => Side::Buy,
            2 => Side::Sell,
            value => return Err(format!("matcher snapshot has invalid side {value}")),
        };
        let kind = bytes[offset + 1];
        if !matches!(kind, 2 | 5 | KIND_ICEBERG) {
            return Err(format!(
                "matcher snapshot has unsupported order kind {kind}"
            ));
        }
        let order_id = read_u64(bytes, offset + 8)?;
        let price = read_u64(bytes, offset + 16)?;
        let quantity = read_u64(bytes, offset + 24)?;
        let filled = read_u64(bytes, offset + 32)?;
        let hidden = read_u64(bytes, offset + 40)?;
        let visible_set = bytes[offset + 48];
        offset += SNAP_RECORD_LEN;
        if kind == KIND_ICEBERG && visible_set != 0 {
            read_u64(bytes, offset)?;
            offset += 8;
        }
        let remaining = quantity
            .checked_sub(filled)
            .and_then(|value| value.checked_add(hidden))
            .ok_or_else(|| "matcher snapshot has invalid remaining quantity".to_string())?;
        orders.push(NativeResting {
            order_id,
            side,
            price,
            remaining,
        });
    }
    if offset != bytes.len() {
        return Err("matcher snapshot has trailing bytes".to_string());
    }
    Ok(orders)
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let value = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| "matcher snapshot is truncated".to_string())?;
    Ok(u64::from_be_bytes(value.try_into().map_err(|_| {
        "matcher snapshot is truncated".to_string()
    })?))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let value = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| "matcher snapshot is truncated".to_string())?;
    Ok(u32::from_be_bytes(value.try_into().map_err(|_| {
        "matcher snapshot is truncated".to_string()
    })?))
}

fn trades_from_events(events: &[BookEvent]) -> Vec<Trade> {
    events
        .iter()
        .filter_map(|event| match event {
            BookEvent::Trade(trade) => Some(*trade),
            _ => None,
        })
        .collect()
}

fn required_quantity(
    config: &AdapterConfig,
    event: &MarketEvent,
    message: &str,
) -> Result<u64, String> {
    event
        .quantity
        .as_ref()
        .ok_or_else(|| message.to_string())
        .and_then(|value| config.quantity_units(value))
}

fn parse_order_id(event: &MarketEvent) -> Result<u64, String> {
    let value = event
        .order_id
        .as_ref()
        .ok_or_else(|| "order_id is required".to_string())?;
    let order_id = value
        .as_u64()
        .ok_or_else(|| "order_id must fit in an unsigned 64-bit integer".to_string())?;
    if order_id == 0 {
        return Err("order_id must be positive".to_string());
    }
    Ok(order_id)
}

fn parse_side(event: &MarketEvent) -> Result<Side, String> {
    match event.side.as_deref() {
        Some("BUY") => Ok(Side::Buy),
        Some("SELL") => Ok(Side::Sell),
        _ => Err("side must be BUY or SELL".to_string()),
    }
}

fn event_time(event: &MarketEvent, index: u64) -> u64 {
    event.timestamp_ns.unwrap_or(index)
}

fn canonical_decimal(value: Decimal) -> String {
    if value == Decimal::ZERO {
        "0".to_string()
    } else {
        value.normalize().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(value: serde_json::Value) -> MarketEvent {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn snapshot_is_canonical_price_time_order() {
        let mut adapter = Adapter::new(ConfigWire::default()).unwrap();
        for (index, value) in [
            serde_json::json!({"op":"new","symbol":"TEST","order_id":1,"side":"BUY","price":99,"quantity":1,"owner":11}),
            serde_json::json!({"op":"new","symbol":"TEST","order_id":2,"side":"BUY","price":100,"quantity":1,"owner":22}),
            serde_json::json!({"op":"new","symbol":"TEST","order_id":3,"side":"BUY","price":100,"quantity":1,"owner":33}),
        ]
        .into_iter()
        .enumerate()
        {
            adapter.apply(&event(value), index as u64 + 1).unwrap();
        }

        let state = adapter.snapshot().unwrap();
        let bids = &state.books[0].bids;
        assert_eq!(
            bids.iter().map(|order| order.order_id).collect::<Vec<_>>(),
            vec![2, 3, 1]
        );
        assert_eq!(
            bids.iter().map(|order| order.owner).collect::<Vec<_>>(),
            vec![22, 33, 11]
        );
    }

    #[test]
    fn reduction_keeps_priority_and_replacement_loses_it() {
        let mut adapter = Adapter::new(ConfigWire::default()).unwrap();
        let values = [
            serde_json::json!({"op":"new","symbol":"TEST","order_id":1,"side":"SELL","price":100,"quantity":5,"owner":1}),
            serde_json::json!({"op":"new","symbol":"TEST","order_id":2,"side":"SELL","price":100,"quantity":5,"owner":2}),
            serde_json::json!({"op":"reduce","symbol":"TEST","order_id":1,"quantity":1}),
            serde_json::json!({"op":"replace","symbol":"TEST","order_id":1,"quantity":3}),
            serde_json::json!({"op":"new","symbol":"TEST","order_id":3,"side":"BUY","price":100,"quantity":5,"owner":3}),
        ];
        let mut final_observation = None;
        for (index, value) in values.into_iter().enumerate() {
            final_observation = Some(adapter.apply(&event(value), index as u64 + 1).unwrap());
        }
        let observation = final_observation.unwrap();
        assert_eq!(observation.trades.len(), 1);
        assert_eq!(observation.trades[0].sell_order_id, 2);
    }

    #[test]
    fn snapshot_decoder_rejects_an_unexpected_version() {
        let mut bytes = OrderBook::new().snapshot();
        bytes[11] = 2;
        assert_eq!(
            decode_snapshot(&bytes).unwrap_err(),
            "unsupported matcher snapshot version 2"
        );
    }

    #[test]
    fn price_conversion_rejects_the_exclusive_u64_boundary() {
        let config = AdapterConfig::from_wire(ConfigWire {
            tick_size: "1".to_string(),
            ..ConfigWire::default()
        })
        .unwrap();
        let boundary = serde_json::Number::from_str("18446744073709551616").unwrap();

        assert_eq!(
            config.price_ticks(&boundary).unwrap_err(),
            "price snaps to a non-positive or unsupported tick"
        );
    }
}
