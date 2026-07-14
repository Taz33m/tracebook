use crate::wire::{
    BookSnapshot, BookState, ConfigWire, MarketEvent, ObservationFrame, Outcome, RestingOrder,
    TradeFill,
};
use orderbook_rs::{OrderBook, OrderBookError, STPMode, StubClock, TradeResult};
use pricelevel::{Hash32, Id, PriceLevelSnapshot, Quantity, Side, TimeInForce};
use rust_decimal::prelude::{FromPrimitive, ToPrimitive};
use rust_decimal::{Decimal, RoundingStrategy};
use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

const NO_OWNER: i64 = -1;
const QUEUE_PRIORITY_PROBE_SYMBOL: &str = "FIFO-PRIORITY-PROBE";
const TRADE_ID_NAMESPACE_PREFIX: &str = "https://github.com/Taz33m/tracebook/orderbook-rs-adapter/";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaultMode {
    None,
    DropFirstTrade,
    QueuePriority,
}

#[derive(Clone)]
struct AdapterConfig {
    tick_size: Decimal,
    tick_size_f64: f64,
    quantity_decimal_places: u32,
    quantity_scale: u64,
    stp_mode: STPMode,
}

impl AdapterConfig {
    fn from_wire(config: ConfigWire) -> Result<Self, String> {
        if !matches!(config.matching_algorithm.as_str(), "fifo" | "pro_rata") {
            return Err("matching_algorithm must be 'fifo' or 'pro_rata'".to_string());
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

        let stp_mode = match config.self_trade_policy.as_str() {
            "NONE" => STPMode::None,
            "CANCEL_RESTING" => STPMode::CancelMaker,
            "CANCEL_INCOMING" => STPMode::CancelTaker,
            _ => {
                return Err(
                    "self_trade_policy must be NONE, CANCEL_RESTING, or CANCEL_INCOMING"
                        .to_string(),
                );
            }
        };
        let quantity_scale = 10_u64
            .checked_pow(config.quantity_decimal_places)
            .ok_or_else(|| "quantity scale exceeds u64".to_string())?;

        Ok(Self {
            tick_size,
            tick_size_f64,
            quantity_decimal_places: config.quantity_decimal_places,
            quantity_scale,
            stp_mode,
        })
    }

    fn price_ticks(&self, value: &serde_json::Number) -> Result<u128, String> {
        let price = value
            .to_string()
            .parse::<f64>()
            .map_err(|_| "price must be finite".to_string())?;
        if !price.is_finite() {
            return Err("price must be finite".to_string());
        }
        let ticks = (price / self.tick_size_f64).round_ties_even();
        if !ticks.is_finite() || ticks <= 0.0 || ticks > u128::MAX as f64 {
            return Err("price snaps to a non-positive or unsupported tick".to_string());
        }
        Ok(ticks as u128)
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

    fn format_price(&self, ticks: u128) -> Result<String, String> {
        let ticks = Decimal::from_u128(ticks)
            .ok_or_else(|| "price tick count exceeds decimal range".to_string())?;
        let price = ticks
            .checked_mul(self.tick_size)
            .ok_or_else(|| "price exceeds decimal range".to_string())?;
        Ok(canonical_decimal(price))
    }

    fn format_quantity(&self, units: u64) -> String {
        let quantity = Decimal::from(units) / Decimal::from(self.quantity_scale);
        canonical_decimal(quantity)
    }
}

struct BookHarness {
    book: OrderBook<()>,
    trade_capture: Arc<Mutex<Vec<TradeResult>>>,
    owners: BTreeMap<u64, i64>,
}

impl BookHarness {
    fn new(symbol: &str, stp_mode: STPMode) -> Self {
        let trade_capture = Arc::new(Mutex::new(Vec::new()));
        let listener_capture = Arc::clone(&trade_capture);
        let mut book = OrderBook::<()>::with_clock_and_namespace(
            symbol,
            Arc::new(StubClock::new()),
            trade_id_namespace(symbol),
        );
        book.set_tick_size(1);
        book.set_stp_mode(stp_mode);
        book.set_trade_listener(Arc::new(move |result| {
            if let Ok(mut captured) = listener_capture.lock() {
                captured.push(result.clone());
            }
        }));
        Self {
            book,
            trade_capture,
            owners: BTreeMap::new(),
        }
    }

    fn clear_trades(&self) -> Result<(), String> {
        self.trade_capture
            .lock()
            .map_err(|_| "trade listener lock was poisoned".to_string())?
            .clear();
        Ok(())
    }

    fn take_trades(&self) -> Result<Vec<TradeResult>, String> {
        let mut captured = self
            .trade_capture
            .lock()
            .map_err(|_| "trade listener lock was poisoned".to_string())?;
        Ok(std::mem::take(&mut *captured))
    }

    fn reconcile_owners(&mut self) {
        let snapshot = self.book.create_snapshot(usize::MAX);
        let active = snapshot
            .bids
            .iter()
            .chain(&snapshot.asks)
            .flat_map(PriceLevelSnapshot::orders)
            .filter_map(|order| order.id().as_u64())
            .collect::<BTreeSet<_>>();
        self.owners.retain(|order_id, _| active.contains(order_id));
    }

    fn inject_queue_priority_fault(
        &mut self,
        config: &AdapterConfig,
        event: &MarketEvent,
        marked_order_id: u64,
    ) -> Result<bool, String> {
        if event.op != "new" || event.symbol != QUEUE_PRIORITY_PROBE_SYMBOL {
            return Ok(false);
        }
        let incoming_side = parse_side(event)?;
        let incoming_price = match event.order_type.as_str() {
            "MARKET" => None,
            _ => Some(
                event
                    .price
                    .as_ref()
                    .ok_or_else(|| "price is required".to_string())
                    .and_then(|value| config.price_ticks(value))?,
            ),
        };
        let snapshot = self.book.create_snapshot(usize::MAX);
        let levels = match incoming_side {
            Side::Buy => &snapshot.asks,
            Side::Sell => &snapshot.bids,
        };

        for level in levels {
            let crosses = match (incoming_side, incoming_price) {
                (_, None) => true,
                (Side::Buy, Some(price)) => price >= level.price().as_u128(),
                (Side::Sell, Some(price)) => price <= level.price().as_u128(),
            };
            if !crosses {
                continue;
            }
            let Some(position) = level
                .orders()
                .iter()
                .position(|order| order.id().as_u64() == Some(marked_order_id))
            else {
                continue;
            };
            if position == 0 {
                return Ok(false);
            }
            let predecessors = level.orders()[..position]
                .iter()
                .map(|order| {
                    let order_id = order.id().as_u64().ok_or_else(|| {
                        "orderbook-rs returned a non-sequential resting id".to_string()
                    })?;
                    let owner = self.owners.get(&order_id).copied().ok_or_else(|| {
                        format!("adapter has no source owner for resting order {order_id}")
                    })?;
                    Ok((
                        order_id,
                        order.price().as_u128(),
                        order.visible_quantity().as_u64(),
                        order.side(),
                        owner,
                    ))
                })
                .collect::<Result<Vec<_>, String>>()?;

            for (order_id, price, quantity, side, owner) in predecessors {
                let id = Id::sequential(order_id);
                match self.book.cancel_order(id) {
                    Ok(Some(_)) => {}
                    Ok(None) => return Err("fault injection lost a resting order".to_string()),
                    Err(error) => return Err(error.to_string()),
                }
                self.book
                    .add_limit_order_with_user(
                        id,
                        price,
                        quantity,
                        side,
                        TimeInForce::Gtc,
                        owner_hash(owner, order_id),
                        None,
                    )
                    .map_err(|error| error.to_string())?;
            }
            return Ok(true);
        }
        Ok(false)
    }
}

pub struct Adapter {
    config: AdapterConfig,
    books: BTreeMap<String, BookHarness>,
    fault_mode: FaultMode,
    fault_fired: bool,
    reduced_orders: BTreeSet<(String, u64)>,
    pending_priority_faults: BTreeSet<(String, u64)>,
}

impl Adapter {
    pub fn new(config: ConfigWire) -> Result<Self, String> {
        Self::new_with_fault(config, FaultMode::None)
    }

    pub fn new_with_test_fault(config: ConfigWire, drop_first_trade: bool) -> Result<Self, String> {
        let mode = if drop_first_trade {
            FaultMode::DropFirstTrade
        } else {
            FaultMode::None
        };
        Self::new_with_fault(config, mode)
    }

    pub fn new_with_fault(config: ConfigWire, fault_mode: FaultMode) -> Result<Self, String> {
        Ok(Self {
            config: AdapterConfig::from_wire(config)?,
            books: BTreeMap::new(),
            fault_mode,
            fault_fired: false,
            reduced_orders: BTreeSet::new(),
            pending_priority_faults: BTreeSet::new(),
        })
    }

    pub fn apply(&mut self, event: &MarketEvent, index: u64) -> Result<ObservationFrame, String> {
        let outcome;
        let native_trades;

        if event.op == "clear" {
            self.books.insert(
                event.symbol.clone(),
                BookHarness::new(&event.symbol, self.config.stp_mode),
            );
            outcome = Outcome::applied();
            native_trades = Vec::new();
            self.reduced_orders
                .retain(|(symbol, _)| symbol != &event.symbol);
            self.pending_priority_faults
                .retain(|(symbol, _)| symbol != &event.symbol);
        } else if !self.books.contains_key(&event.symbol)
            && matches!(event.op.as_str(), "cancel" | "reduce" | "replace")
        {
            outcome = Outcome::rejected("ORDER_NOT_ACTIVE", "order is not active");
            native_trades = Vec::new();
        } else {
            if !self.books.contains_key(&event.symbol) {
                self.books.insert(
                    event.symbol.clone(),
                    BookHarness::new(&event.symbol, self.config.stp_mode),
                );
            }
            let book = self
                .books
                .get_mut(&event.symbol)
                .ok_or_else(|| "book creation failed".to_string())?;
            book.clear_trades()?;
            if self.fault_mode == FaultMode::QueuePriority && !self.fault_fired {
                let candidate = self
                    .pending_priority_faults
                    .iter()
                    .find(|(symbol, _)| symbol == &event.symbol)
                    .map(|(_, order_id)| *order_id);
                if let Some(order_id) = candidate {
                    self.fault_fired =
                        book.inject_queue_priority_fault(&self.config, event, order_id)?;
                }
            }
            outcome = Self::apply_to_book(&self.config, book, event);
            native_trades = book.take_trades()?;
            book.reconcile_owners();
        }

        if self.fault_mode == FaultMode::QueuePriority && outcome.status == "applied" {
            if let Some(order_id) = event.order_id.as_ref().and_then(serde_json::Number::as_u64) {
                let key = (event.symbol.clone(), order_id);
                match event.op.as_str() {
                    "reduce" => {
                        self.reduced_orders.insert(key);
                    }
                    "replace" if self.reduced_orders.contains(&key) => {
                        self.pending_priority_faults.insert(key);
                    }
                    "cancel" => {
                        self.reduced_orders.remove(&key);
                        self.pending_priority_faults.remove(&key);
                    }
                    _ => {}
                }
            }
        }

        let mut trades = self.convert_trades(native_trades)?;
        if self.fault_mode == FaultMode::DropFirstTrade && !self.fault_fired && !trades.is_empty() {
            trades.remove(0);
            self.fault_fired = true;
        }
        let state = self.snapshot()?;
        Ok(ObservationFrame {
            kind: "observation",
            index,
            outcome,
            trades,
            state_hash: state.digest()?,
            resting_order_count: state.order_count(),
        })
    }

    fn apply_to_book(
        config: &AdapterConfig,
        book: &mut BookHarness,
        event: &MarketEvent,
    ) -> Outcome {
        match event.op.as_str() {
            "new" => Self::apply_new(config, book, event),
            "cancel" => Self::apply_cancel(book, event),
            "reduce" => Self::apply_reduce(config, book, event),
            "replace" => Self::apply_replace(config, book, event),
            _ => Outcome::rejected("INVALID_ORDER", "unsupported event operation"),
        }
    }

    fn apply_new(config: &AdapterConfig, book: &mut BookHarness, event: &MarketEvent) -> Outcome {
        let order_id = match parse_order_id(event) {
            Ok(value) => value,
            Err(message) => return Outcome::rejected("INVALID_ORDER", message),
        };
        let id = Id::sequential(order_id);
        if book.book.get_order(id).is_some() {
            return Outcome::rejected("DUPLICATE_ORDER_ID", "order is already active");
        }
        let side = match parse_side(event) {
            Ok(value) => value,
            Err(message) => return Outcome::rejected("INVALID_ORDER", message),
        };
        let quantity = match event
            .quantity
            .as_ref()
            .ok_or_else(|| "quantity is required".to_string())
            .and_then(|value| config.quantity_units(value))
        {
            Ok(value) => value,
            Err(message) => return Outcome::rejected("INVALID_ORDER", message),
        };
        let user_id = owner_hash(event.owner, order_id);
        book.owners.insert(order_id, event.owner);

        let result = match event.order_type.as_str() {
            "MARKET" => book
                .book
                .submit_market_order_with_user(id, quantity, side, user_id)
                .map(|_| ()),
            "LIMIT" | "IOC" | "FOK" => {
                let price = match event
                    .price
                    .as_ref()
                    .ok_or_else(|| "price is required".to_string())
                    .and_then(|value| config.price_ticks(value))
                {
                    Ok(value) => value,
                    Err(message) => {
                        book.owners.remove(&order_id);
                        return Outcome::rejected("INVALID_ORDER", message);
                    }
                };
                let time_in_force = match event.order_type.as_str() {
                    "IOC" => TimeInForce::Ioc,
                    "FOK" => TimeInForce::Fok,
                    _ => TimeInForce::Gtc,
                };
                book.book
                    .add_limit_order_with_user(
                        id,
                        price,
                        quantity,
                        side,
                        time_in_force,
                        user_id,
                        None,
                    )
                    .map(|_| ())
            }
            _ => {
                book.owners.remove(&order_id);
                return Outcome::rejected("INVALID_ORDER", "unsupported order_type");
            }
        };

        match result {
            Ok(()) => Outcome::applied(),
            Err(OrderBookError::DuplicateOrderId { .. }) => {
                book.owners.remove(&order_id);
                Outcome::rejected("DUPLICATE_ORDER_ID", "order is already active")
            }
            Err(error) if expected_immediate_outcome(&error, &event.order_type) => {
                Outcome::applied()
            }
            Err(error) => {
                book.owners.remove(&order_id);
                Outcome::rejected("INVALID_ORDER", error.to_string())
            }
        }
    }

    fn apply_cancel(book: &mut BookHarness, event: &MarketEvent) -> Outcome {
        let order_id = match parse_order_id(event) {
            Ok(value) => value,
            Err(message) => return Outcome::rejected("INVALID_CANCEL", message),
        };
        match book.book.cancel_order(Id::sequential(order_id)) {
            Ok(Some(_)) => {
                book.owners.remove(&order_id);
                Outcome::applied()
            }
            Ok(None) => Outcome::rejected("ORDER_NOT_ACTIVE", "order is not active"),
            Err(error) => Outcome::rejected("INVALID_CANCEL", error.to_string()),
        }
    }

    fn apply_reduce(
        config: &AdapterConfig,
        book: &mut BookHarness,
        event: &MarketEvent,
    ) -> Outcome {
        let order_id = match parse_order_id(event) {
            Ok(value) => value,
            Err(message) => return Outcome::rejected("INVALID_ORDER", message),
        };
        let id = Id::sequential(order_id);
        let Some(existing) = book.book.get_order(id) else {
            return Outcome::rejected("ORDER_NOT_ACTIVE", "order is not active");
        };
        let reduction = match event
            .quantity
            .as_ref()
            .ok_or_else(|| "reduction quantity is required".to_string())
            .and_then(|value| config.quantity_units(value))
        {
            Ok(value) => value,
            Err(message) => return Outcome::rejected("INVALID_ORDER", message),
        };
        let remaining = existing.visible_quantity().as_u64();
        if reduction > remaining {
            return Outcome::rejected(
                "INVALID_ORDER",
                "reduction quantity cannot exceed remaining quantity",
            );
        }
        if reduction == remaining {
            return Self::apply_cancel(book, event);
        }

        match book
            .book
            .update_order(pricelevel::OrderUpdate::UpdateQuantity {
                order_id: id,
                new_quantity: Quantity::new(remaining - reduction),
            }) {
            Ok(Some(_)) => Outcome::applied(),
            Ok(None) => Outcome::rejected("ORDER_NOT_ACTIVE", "order is not active"),
            Err(error) => Outcome::rejected("INVALID_ORDER", error.to_string()),
        }
    }

    fn apply_replace(
        config: &AdapterConfig,
        book: &mut BookHarness,
        event: &MarketEvent,
    ) -> Outcome {
        let order_id = match parse_order_id(event) {
            Ok(value) => value,
            Err(message) => return Outcome::rejected("INVALID_REPLACEMENT", message),
        };
        let id = Id::sequential(order_id);
        let Some(existing) = book.book.get_order(id) else {
            return Outcome::rejected("ORDER_NOT_ACTIVE", "order is not active");
        };
        let price = match event.price.as_ref() {
            Some(value) => match config.price_ticks(value) {
                Ok(value) => value,
                Err(message) => return Outcome::rejected("INVALID_REPLACEMENT", message),
            },
            None => existing.price().as_u128(),
        };
        let quantity = match event.quantity.as_ref() {
            Some(value) => match config.quantity_units(value) {
                Ok(value) => value,
                Err(message) => return Outcome::rejected("INVALID_REPLACEMENT", message),
            },
            None => existing.visible_quantity().as_u64(),
        };
        let side = existing.side();
        let owner = book.owners.get(&order_id).copied().unwrap_or(NO_OWNER);
        let user_id = owner_hash(owner, order_id);

        match book.book.cancel_order(id) {
            Ok(Some(_)) => {}
            Ok(None) => return Outcome::rejected("ORDER_NOT_ACTIVE", "order is not active"),
            Err(error) => return Outcome::rejected("INVALID_REPLACEMENT", error.to_string()),
        }
        match book.book.add_limit_order_with_user(
            id,
            price,
            quantity,
            side,
            TimeInForce::Gtc,
            user_id,
            None,
        ) {
            Ok(_) => Outcome::applied(),
            Err(OrderBookError::SelfTradePrevented { .. }) => Outcome::applied(),
            Err(error) => Outcome::rejected("INVALID_REPLACEMENT", error.to_string()),
        }
    }

    fn convert_trades(&self, results: Vec<TradeResult>) -> Result<Vec<TradeFill>, String> {
        let mut fills = Vec::new();
        for result in results {
            for trade in result.match_result.trades().as_vec() {
                let taker = trade
                    .taker_order_id()
                    .as_u64()
                    .ok_or_else(|| "orderbook-rs returned a non-sequential taker id".to_string())?;
                let maker = trade
                    .maker_order_id()
                    .as_u64()
                    .ok_or_else(|| "orderbook-rs returned a non-sequential maker id".to_string())?;
                let (buy_order_id, sell_order_id) = match trade.taker_side() {
                    Side::Buy => (taker, maker),
                    Side::Sell => (maker, taker),
                };
                fills.push(TradeFill {
                    symbol: result.symbol.clone(),
                    buy_order_id,
                    sell_order_id,
                    price: self.config.format_price(trade.price().as_u128())?,
                    quantity: self.config.format_quantity(trade.quantity().as_u64()),
                });
            }
        }
        Ok(fills)
    }

    pub fn snapshot(&self) -> Result<BookState, String> {
        let books = self
            .books
            .iter()
            .map(|(symbol, harness)| {
                let native = harness.book.create_snapshot(usize::MAX);
                Ok(BookSnapshot {
                    symbol: symbol.clone(),
                    bids: self.resting_orders(&native.bids, &harness.owners)?,
                    asks: self.resting_orders(&native.asks, &harness.owners)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(BookState { books })
    }

    fn resting_orders(
        &self,
        levels: &[PriceLevelSnapshot],
        owners: &BTreeMap<u64, i64>,
    ) -> Result<Vec<RestingOrder>, String> {
        levels
            .iter()
            .flat_map(PriceLevelSnapshot::orders)
            .map(|order| {
                let order_id = order.id().as_u64().ok_or_else(|| {
                    "orderbook-rs returned a non-sequential resting id".to_string()
                })?;
                let owner = owners.get(&order_id).copied().ok_or_else(|| {
                    format!("adapter has no source owner for resting order {order_id}")
                })?;
                Ok(RestingOrder {
                    order_id,
                    price: self.config.format_price(order.price().as_u128())?,
                    remaining_quantity: self
                        .config
                        .format_quantity(order.visible_quantity().as_u64()),
                    owner,
                    order_type: "LIMIT",
                })
            })
            .collect()
    }
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

fn owner_hash(owner: i64, order_id: u64) -> Hash32 {
    let mut bytes = [0_u8; 32];
    if owner == NO_OWNER {
        bytes[0] = 2;
        bytes[24..].copy_from_slice(&order_id.to_be_bytes());
    } else {
        bytes[0] = 1;
        bytes[24..].copy_from_slice(&owner.to_be_bytes());
    }
    Hash32::new(bytes)
}

fn trade_id_namespace(symbol: &str) -> Uuid {
    let name = format!("{TRADE_ID_NAMESPACE_PREFIX}{symbol}");
    Uuid::new_v5(&Uuid::NAMESPACE_URL, name.as_bytes())
}

fn expected_immediate_outcome(error: &OrderBookError, order_type: &str) -> bool {
    matches!(error, OrderBookError::SelfTradePrevented { .. })
        || (matches!(order_type, "MARKET" | "IOC" | "FOK")
            && matches!(error, OrderBookError::InsufficientLiquidity { .. }))
}

fn canonical_decimal(value: Decimal) -> String {
    if value.is_zero() {
        "0".to_string()
    } else {
        value.normalize().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> AdapterConfig {
        AdapterConfig::from_wire(ConfigWire {
            matching_algorithm: "fifo".to_string(),
            tick_size: "0.01".to_string(),
            self_trade_policy: "NONE".to_string(),
            quantity_decimal_places: 3,
        })
        .unwrap()
    }

    #[test]
    fn quantity_conversion_uses_half_even_rounding() {
        let config = config();
        let low = serde_json::Number::from_str("1.2345").unwrap();
        let high = serde_json::Number::from_str("1.2355").unwrap();
        assert_eq!(config.quantity_units(&low).unwrap(), 1_234);
        assert_eq!(config.quantity_units(&high).unwrap(), 1_236);
    }

    #[test]
    fn anonymous_orders_get_distinct_stp_identities() {
        assert_ne!(owner_hash(NO_OWNER, 10), owner_hash(NO_OWNER, 11));
        assert_eq!(owner_hash(42, 10), owner_hash(42, 11));
    }

    #[test]
    fn price_conversion_matches_half_even_tick_snapping() {
        let config = config();
        let number = serde_json::Number::from_str("100.005").unwrap();
        assert_eq!(config.price_ticks(&number).unwrap(), 10_000);
    }

    fn first_native_trade_id(symbol: &str) -> String {
        let harness = BookHarness::new(symbol, STPMode::None);
        harness
            .book
            .add_limit_order(
                Id::sequential(1),
                10_000,
                1,
                Side::Sell,
                TimeInForce::Gtc,
                None,
            )
            .unwrap();
        let result = harness
            .book
            .match_market_order(Id::sequential(2), 1, Side::Buy)
            .unwrap();
        result.trades().as_vec()[0].trade_id().to_string()
    }

    #[test]
    fn native_trade_ids_are_deterministic_and_symbol_scoped() {
        assert_eq!(
            first_native_trade_id("ALPHA"),
            first_native_trade_id("ALPHA")
        );
        assert_ne!(
            first_native_trade_id("ALPHA"),
            first_native_trade_id("BETA")
        );
    }

    #[test]
    fn profile_queue_view_preserves_reduce_and_demotes_replace() {
        let mut adapter = Adapter::new(ConfigWire {
            matching_algorithm: "fifo".to_string(),
            tick_size: "0.01".to_string(),
            self_trade_policy: "NONE".to_string(),
            quantity_decimal_places: 12,
        })
        .unwrap();
        let events = [
            serde_json::json!({
                "op": "new", "symbol": "ALPHA", "order_id": 1,
                "side": "SELL", "order_type": "LIMIT", "price": 100.0,
                "quantity": 5.0, "owner": 101
            }),
            serde_json::json!({
                "op": "new", "symbol": "ALPHA", "order_id": 2,
                "side": "SELL", "order_type": "LIMIT", "price": 100.0,
                "quantity": 5.0, "owner": 102
            }),
            serde_json::json!({
                "op": "reduce", "symbol": "ALPHA", "order_id": 1,
                "quantity": 1.0
            }),
            serde_json::json!({
                "op": "replace", "symbol": "ALPHA", "order_id": 1,
                "price": 100.0, "quantity": 4.0
            }),
        ]
        .map(|value| serde_json::from_value::<MarketEvent>(value).unwrap());

        for (position, event) in events[..3].iter().enumerate() {
            adapter.apply(event, (position + 1) as u64).unwrap();
        }
        let after_reduce = adapter.snapshot().unwrap();
        let reduced_ids = after_reduce.books[0]
            .asks
            .iter()
            .map(|order| order.order_id)
            .collect::<Vec<_>>();
        assert_eq!(reduced_ids, vec![1, 2]);

        adapter.apply(&events[3], 4).unwrap();
        let after_replace = adapter.snapshot().unwrap();
        let replaced_ids = after_replace.books[0]
            .asks
            .iter()
            .map(|order| order.order_id)
            .collect::<Vec<_>>();
        assert_eq!(replaced_ids, vec![2, 1]);
    }

    #[test]
    fn queue_priority_fault_changes_only_the_probe_trade() {
        let wire_config = ConfigWire {
            matching_algorithm: "fifo".to_string(),
            tick_size: "0.01".to_string(),
            self_trade_policy: "NONE".to_string(),
            quantity_decimal_places: 12,
        };
        let mut correct = Adapter::new(wire_config.clone()).unwrap();
        let mut faulty = Adapter::new_with_fault(wire_config, FaultMode::QueuePriority).unwrap();
        let events = [
            serde_json::json!({
                "op": "new", "symbol": QUEUE_PRIORITY_PROBE_SYMBOL,
                "order_id": 9000000001_u64, "side": "SELL", "order_type": "LIMIT",
                "price": 100.0, "quantity": 5.0, "owner": 101
            }),
            serde_json::json!({
                "op": "new", "symbol": QUEUE_PRIORITY_PROBE_SYMBOL,
                "order_id": 9000000002_u64, "side": "SELL", "order_type": "LIMIT",
                "price": 100.0, "quantity": 5.0, "owner": 102
            }),
            serde_json::json!({
                "op": "reduce", "symbol": QUEUE_PRIORITY_PROBE_SYMBOL,
                "order_id": 9000000001_u64, "quantity": 1.0
            }),
            serde_json::json!({
                "op": "replace", "symbol": QUEUE_PRIORITY_PROBE_SYMBOL,
                "order_id": 9000000001_u64, "price": 100.0, "quantity": 4.0
            }),
            serde_json::json!({
                "op": "new", "symbol": QUEUE_PRIORITY_PROBE_SYMBOL,
                "order_id": 9000000003_u64, "side": "BUY", "order_type": "LIMIT",
                "price": 100.0, "quantity": 1.0, "owner": 103
            }),
        ]
        .map(|value| serde_json::from_value::<MarketEvent>(value).unwrap());

        for (position, event) in events.iter().enumerate() {
            let index = (position + 1) as u64;
            let correct_observation = correct.apply(event, index).unwrap();
            let faulty_observation = faulty.apply(event, index).unwrap();
            if index < 5 {
                assert_eq!(
                    faulty_observation.state_hash,
                    correct_observation.state_hash
                );
                assert!(faulty_observation.trades.is_empty());
            } else {
                assert_eq!(correct_observation.trades[0].sell_order_id, 9_000_000_002);
                assert_eq!(faulty_observation.trades[0].sell_order_id, 9_000_000_001);
            }
        }
    }
}
