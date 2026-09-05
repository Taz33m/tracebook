package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"time"

	"github.com/intrepidkarthi/orderbook/pkg/matching"
	"github.com/intrepidkarthi/orderbook/pkg/types"
	"github.com/shopspring/decimal"
)

type adapterConfig struct {
	matchingAlgorithm string
	tickSize          decimal.Decimal
	tickSizeFloat     float64
	quantityPlaces    uint32
	nativeQtyPlaces   uint32
	quantityScale     int64
	stp               matching.SelfTradePrevention
}

func parseConfig(wire configWire) (adapterConfig, error) {
	algorithm := wire.MatchingAlgorithm
	if algorithm == "" {
		algorithm = "fifo"
	}
	if algorithm != "fifo" && algorithm != "pro_rata" {
		return adapterConfig{}, errors.New("matching_algorithm must be 'fifo' or 'pro_rata'")
	}
	tickText := wire.TickSize
	if tickText == "" {
		tickText = "0.01"
	}
	tick, err := decimal.NewFromString(tickText)
	if err != nil || !tick.IsPositive() {
		return adapterConfig{}, errors.New("tick_size must be a positive finite decimal")
	}
	tickFloat, err := strconv.ParseFloat(tickText, 64)
	if err != nil || math.IsInf(tickFloat, 0) || math.IsNaN(tickFloat) || tickFloat <= 0 {
		return adapterConfig{}, errors.New("tick_size must be a positive finite binary64 value")
	}
	places := uint32(12)
	if wire.QuantityDecimalPlaces != nil {
		places = *wire.QuantityDecimalPlaces
	}
	if places > 18 {
		return adapterConfig{}, errors.New("quantity_decimal_places must be between 0 and 18")
	}
	// Protocol precision controls observations, not submitted quantities.
	// Keep six native decimal places regardless of that output setting, and
	// reject inputs that would require rounding before the engine sees them.
	const nativePlaces = uint32(6)
	const scale = int64(1_000_000)
	policy := wire.SelfTradePolicy
	if policy == "" {
		policy = "NONE"
	}
	var stp matching.SelfTradePrevention
	switch policy {
	case "NONE":
		stp = matching.STPAllow
	case "CANCEL_RESTING":
		stp = matching.STPCancelOldest
	case "CANCEL_INCOMING":
		stp = matching.STPCancelNewest
	default:
		return adapterConfig{}, errors.New(
			"self_trade_policy must be NONE, CANCEL_RESTING, or CANCEL_INCOMING",
		)
	}
	return adapterConfig{
		matchingAlgorithm: algorithm, tickSize: tick, tickSizeFloat: tickFloat,
		quantityPlaces: places, nativeQtyPlaces: nativePlaces, quantityScale: scale, stp: stp,
	}, nil
}

func (config adapterConfig) priceTicks(value *jsonNumber) (int64, error) {
	if value == nil {
		return 0, errors.New("price is required")
	}
	price, err := strconv.ParseFloat(value.String(), 64)
	if err != nil || math.IsInf(price, 0) || math.IsNaN(price) {
		return 0, errors.New("price must be finite")
	}
	// The reference snaps the normalized binary64 input with ties to even.
	// Decimal division differs at boundaries such as 1.015 / 0.01.
	ticks := math.RoundToEven(price / config.tickSizeFloat)
	if math.IsInf(ticks, 0) || math.IsNaN(ticks) || ticks <= 0 || ticks >= math.Exp2(63) {
		return 0, errors.New("price snaps to a non-positive or unsupported tick")
	}
	return int64(ticks), nil
}

// jsonNumber preserves the wire text for each explicit numeric conversion.
type jsonNumber = json.Number

func (config adapterConfig) quantityUnits(value *jsonNumber) (int64, error) {
	if value == nil {
		return 0, errors.New("quantity is required")
	}
	quantity, err := decimal.NewFromString(value.String())
	if err != nil || !quantity.IsPositive() {
		return 0, errors.New("quantity must be positive")
	}
	scaled := quantity.Mul(
		decimal.NewFromInt(config.quantityScale),
	)
	if !scaled.Equal(scaled.Truncate(0)) {
		return 0, fmt.Errorf(
			"quantity requires more than %d native decimal places",
			config.nativeQtyPlaces,
		)
	}
	units := scaled.BigInt()
	if !units.IsInt64() || units.Sign() <= 0 {
		return 0, errors.New("quantity exceeds the adapter's integer range")
	}
	return units.Int64(), nil
}

func (config adapterConfig) formatPrice(ticks int64) string {
	return decimal.NewFromInt(ticks).Mul(config.tickSize).String()
}

func (config adapterConfig) formatQuantity(units int64) string {
	return decimal.NewFromInt(units).Shift(-int32(config.nativeQtyPlaces)).
		RoundBank(int32(config.quantityPlaces)).String()
}

type bookHarness struct {
	engine         *matching.Engine
	sourceToNative map[int64]int64
	nativeToSource map[int64]int64
	nativeToOwner  map[int64]int64
}

func newBook(symbol string, config adapterConfig) *bookHarness {
	nativeConfig := matching.DefaultConfig(symbol)
	nativeConfig.SelfTradePrevention = config.stp
	nativeConfig.ProRata = config.matchingAlgorithm == "pro_rata"
	nativeConfig.Clock = func() time.Time { return time.Unix(0, 0).UTC() }
	return &bookHarness{
		engine:         matching.NewEngine(nativeConfig),
		sourceToNative: make(map[int64]int64),
		nativeToSource: make(map[int64]int64),
		nativeToOwner:  make(map[int64]int64),
	}
}

func (book *bookHarness) reconcile() {
	active := make(map[int64]bool)
	for _, order := range book.engine.RestingOrders() {
		active[order.ID] = true
	}
	for nativeID, sourceID := range book.nativeToSource {
		if !active[nativeID] {
			delete(book.nativeToSource, nativeID)
			delete(book.nativeToOwner, nativeID)
			if book.sourceToNative[sourceID] == nativeID {
				delete(book.sourceToNative, sourceID)
			}
		}
	}
}

type adapter struct {
	config adapterConfig
	books  map[string]*bookHarness
}

func newAdapter(wire configWire) (*adapter, error) {
	config, err := parseConfig(wire)
	if err != nil {
		return nil, err
	}
	return &adapter{config: config, books: make(map[string]*bookHarness)}, nil
}

func (adapter *adapter) book(symbol string) *bookHarness {
	book := adapter.books[symbol]
	if book == nil {
		book = newBook(symbol, adapter.config)
		adapter.books[symbol] = book
	}
	return book
}

func parseOrderID(event marketEvent) (int64, error) {
	if event.OrderID == nil {
		return 0, errors.New("order_id is required")
	}
	value, err := strconv.ParseInt(event.OrderID.String(), 10, 64)
	if err != nil || value <= 0 {
		return 0, errors.New("order_id must be a positive integer")
	}
	return value, nil
}

func eventOwner(event marketEvent) int64 {
	if event.Owner == nil {
		return -1
	}
	return *event.Owner
}

func nativeOwner(sourceID, owner int64) string {
	if owner == -1 {
		// Anonymous source orders never self-match. Give each active anonymous
		// order its own native account, disjoint from every numeric owner id.
		return "anonymous:" + strconv.FormatInt(sourceID, 10)
	}
	return strconv.FormatInt(owner, 10)
}

// checkQuantityCapacity rejects unsupported native arithmetic before mutation.
// It reserves the entire incoming limit quantity on its side, even if some
// would execute immediately, and checks each potentially crossed pro-rata
// maker conservatively using the entire incoming quantity. This is a numeric
// envelope, not a replacement for the engine's liquidity or matching decisions.
func (book *bookHarness) checkQuantityCapacity(
	config adapterConfig, side types.Side, price, quantity int64,
	orderType string, replacedID int64,
) error {
	// Process checks this for native limit orders (including IOC/FOK). Repeat
	// the same representability check before Replace can cancel its old order.
	// Market orders have no submitted notional and retain the native exemption.
	if orderType != "MARKET" && price > math.MaxInt64/quantity {
		return types.ErrNotionalOverflow
	}
	var sameSideTotal int64
	for _, resting := range book.engine.RestingOrders() {
		if resting.ID == replacedID {
			continue
		}
		if resting.Side == side {
			if resting.RemainingQty > math.MaxInt64-sameSideTotal {
				return errors.New("resting quantity exceeds the native int64 aggregate range")
			}
			sameSideTotal += resting.RemainingQty
			continue
		}
		crosses := orderType == "MARKET" ||
			(side == types.SideBuy && price >= resting.Price) ||
			(side == types.SideSell && price <= resting.Price)
		if config.matchingAlgorithm == "pro_rata" && crosses &&
			quantity > math.MaxInt64/resting.RemainingQty {
			return errors.New("quantity exceeds the native int64 pro-rata product range")
		}
	}
	if orderType == "LIMIT" && quantity > math.MaxInt64-sameSideTotal {
		return errors.New("quantity exceeds the native int64 side aggregate range")
	}
	return nil
}

func nativeSide(value *string) (types.Side, error) {
	if value == nil {
		return "", errors.New("side is required")
	}
	switch *value {
	case "BUY":
		return types.SideBuy, nil
	case "SELL":
		return types.SideSell, nil
	default:
		return "", errors.New("side must be BUY or SELL")
	}
}

func sourceID(book *bookHarness, nativeID int64) (int64, error) {
	value, ok := book.nativeToSource[nativeID]
	if !ok {
		return 0, fmt.Errorf("adapter has no source id for native order %d", nativeID)
	}
	return value, nil
}

func (adapter *adapter) convertTrades(
	book *bookHarness,
	symbol string,
	trades []*types.Trade,
) ([]tradeFill, error) {
	converted := make([]tradeFill, 0, len(trades))
	for _, trade := range trades {
		buyID, err := sourceID(book, trade.BuyOrderID)
		if err != nil {
			return nil, err
		}
		sellID, err := sourceID(book, trade.SellOrderID)
		if err != nil {
			return nil, err
		}
		converted = append(converted, tradeFill{
			Symbol: symbol, BuyOrderID: buyID, SellOrderID: sellID,
			Price:    adapter.config.formatPrice(trade.Price),
			Quantity: adapter.config.formatQuantity(trade.Quantity),
		})
	}
	return converted, nil
}

func (adapter *adapter) apply(event marketEvent, index uint64) (observationFrame, error) {
	result := applied()
	var trades []tradeFill
	if event.Op == "clear" {
		adapter.books[event.Symbol] = newBook(event.Symbol, adapter.config)
	} else if event.Op == "cancel" || event.Op == "reduce" || event.Op == "replace" {
		book := adapter.books[event.Symbol]
		if book == nil {
			result = rejected("ORDER_NOT_ACTIVE", "order is not active")
		} else {
			var err error
			result, trades, err = adapter.applyLifecycle(book, event)
			if err != nil {
				return observationFrame{}, err
			}
		}
	} else if event.Op == "new" {
		book := adapter.book(event.Symbol)
		var err error
		result, trades, err = adapter.applyNew(book, event)
		if err != nil {
			return observationFrame{}, err
		}
	} else {
		result = rejected("INVALID_ORDER", "unsupported event operation")
	}

	state, err := adapter.snapshot()
	if err != nil {
		return observationFrame{}, err
	}
	digest, err := state.digest()
	if err != nil {
		return observationFrame{}, err
	}
	if trades == nil {
		trades = []tradeFill{}
	}
	return observationFrame{
		Type: "observation", Index: index, Outcome: result, Trades: trades,
		StateHash: digest, RestingOrderCount: state.orderCount(),
	}, nil
}

func (adapter *adapter) applyNew(
	book *bookHarness,
	event marketEvent,
) (outcome, []tradeFill, error) {
	source, err := parseOrderID(event)
	if err != nil {
		return rejected("INVALID_ORDER", err.Error()), nil, nil
	}
	if _, exists := book.sourceToNative[source]; exists {
		return rejected("DUPLICATE_ORDER_ID", "order is already active"), nil, nil
	}
	side, err := nativeSide(event.Side)
	if err != nil {
		return rejected("INVALID_ORDER", err.Error()), nil, nil
	}
	quantity, err := adapter.config.quantityUnits(event.Quantity)
	if err != nil {
		return rejected("INVALID_ORDER", err.Error()), nil, nil
	}
	orderType := event.OrderType
	if orderType == "" {
		orderType = "LIMIT"
	}
	price := int64(0)
	if orderType != "MARKET" {
		price, err = adapter.config.priceTicks(event.Price)
		if err != nil {
			return rejected("INVALID_ORDER", err.Error()), nil, nil
		}
	}
	var nativeType types.OrderType
	var tif types.TimeInForce
	switch orderType {
	case "LIMIT":
		nativeType, tif = types.OrderTypeLimit, types.TIFGoodTillCancel
	case "MARKET":
		nativeType, tif = types.OrderTypeMarket, types.TIFImmediateOrCancel
	case "IOC":
		nativeType, tif = types.OrderTypeLimit, types.TIFImmediateOrCancel
	case "FOK":
		nativeType, tif = types.OrderTypeLimit, types.TIFFillOrKill
	default:
		return rejected("INVALID_ORDER", "unsupported order_type"), nil, nil
	}
	if err := book.checkQuantityCapacity(adapter.config, side, price, quantity, orderType, 0); err != nil {
		return rejected("INVALID_ORDER", err.Error()), nil, nil
	}
	sourceOwner := eventOwner(event)
	owner := nativeOwner(source, sourceOwner)
	order, err := types.NewOrder(owner, event.Symbol, side, nativeType, price, quantity, tif)
	if err != nil {
		return rejected("INVALID_ORDER", err.Error()), nil, nil
	}
	order.ClientOrderID = strconv.FormatInt(source, 10)
	match := book.engine.Process(order)
	book.nativeToSource[order.ID] = source
	book.nativeToOwner[order.ID] = sourceOwner
	converted, err := adapter.convertTrades(book, event.Symbol, match.Trades)
	if err != nil {
		return outcome{}, nil, err
	}
	if order.IsActive() && order.RemainingQty > 0 {
		book.sourceToNative[source] = order.ID
	}
	book.reconcile()
	if match.Status == types.OrderStatusRejected {
		message := "order was rejected"
		if match.RejectionReason != nil {
			message = match.RejectionReason.Error()
		}
		return rejected("INVALID_ORDER", message), converted, nil
	}
	return applied(), converted, nil
}

func (adapter *adapter) applyLifecycle(
	book *bookHarness,
	event marketEvent,
) (outcome, []tradeFill, error) {
	source, err := parseOrderID(event)
	if err != nil {
		return rejected("ORDER_NOT_ACTIVE", err.Error()), nil, nil
	}
	nativeID, exists := book.sourceToNative[source]
	if !exists {
		return rejected("ORDER_NOT_ACTIVE", "order is not active"), nil, nil
	}
	orders := book.engine.RestingOrders()
	var current *types.Order
	for _, order := range orders {
		if order.ID == nativeID {
			current = order
			break
		}
	}
	if current == nil {
		book.reconcile()
		return rejected("ORDER_NOT_ACTIVE", "order is not active"), nil, nil
	}
	owner := current.UserID
	sourceOwner, ok := book.nativeToOwner[nativeID]
	if !ok {
		return outcome{}, nil, fmt.Errorf("adapter has no source owner for native order %d", nativeID)
	}
	switch event.Op {
	case "cancel":
		if _, err := book.engine.Cancel(nativeID, owner); err != nil {
			return rejected("ORDER_NOT_ACTIVE", err.Error()), nil, nil
		}
		book.reconcile()
		return applied(), nil, nil
	case "reduce":
		reduction, err := adapter.config.quantityUnits(event.Quantity)
		if err != nil || reduction > current.RemainingQty {
			message := "reduction exceeds remaining quantity"
			if err != nil {
				message = err.Error()
			}
			return rejected("INVALID_ORDER", message), nil, nil
		}
		if reduction == current.RemainingQty {
			if _, err := book.engine.Cancel(nativeID, owner); err != nil {
				return rejected("ORDER_NOT_ACTIVE", err.Error()), nil, nil
			}
		} else {
			newTotal := current.Quantity - reduction
			if _, err := book.engine.Reduce(nativeID, newTotal, owner); err != nil {
				return rejected("INVALID_ORDER", err.Error()), nil, nil
			}
		}
		book.reconcile()
		return applied(), nil, nil
	case "replace":
		price := current.Price
		if event.Price != nil {
			price, err = adapter.config.priceTicks(event.Price)
			if err != nil {
				return rejected("INVALID_REPLACEMENT", err.Error()), nil, nil
			}
		}
		quantity := current.RemainingQty
		if event.Quantity != nil {
			quantity, err = adapter.config.quantityUnits(event.Quantity)
			if err != nil {
				return rejected("INVALID_REPLACEMENT", err.Error()), nil, nil
			}
		}
		if err := book.checkQuantityCapacity(
			adapter.config, current.Side, price, quantity, "LIMIT", nativeID,
		); err != nil {
			return rejected("INVALID_REPLACEMENT", err.Error()), nil, nil
		}
		replacement, err := types.NewOrder(
			owner, event.Symbol, current.Side, types.OrderTypeLimit,
			price, quantity, types.TIFGoodTillCancel,
		)
		if err != nil {
			return rejected("INVALID_REPLACEMENT", err.Error()), nil, nil
		}
		replacement.ClientOrderID = strconv.FormatInt(source, 10)
		match, err := book.engine.Replace(nativeID, owner, replacement)
		if err != nil {
			return rejected("INVALID_REPLACEMENT", err.Error()), nil, nil
		}
		book.nativeToSource[replacement.ID] = source
		book.nativeToOwner[replacement.ID] = sourceOwner
		converted, err := adapter.convertTrades(book, event.Symbol, match.Trades)
		if err != nil {
			return outcome{}, nil, err
		}
		delete(book.sourceToNative, source)
		if replacement.IsActive() && replacement.RemainingQty > 0 {
			book.sourceToNative[source] = replacement.ID
		}
		book.reconcile()
		if match.Status == types.OrderStatusRejected {
			message := "replacement was rejected"
			if match.RejectionReason != nil {
				message = match.RejectionReason.Error()
			}
			return rejected("INVALID_REPLACEMENT", message), converted, nil
		}
		return applied(), converted, nil
	default:
		return rejected("INVALID_ORDER", "unsupported lifecycle operation"), nil, nil
	}
}

func (adapter *adapter) snapshot() (bookState, error) {
	symbols := make([]string, 0, len(adapter.books))
	for symbol := range adapter.books {
		symbols = append(symbols, symbol)
	}
	sort.Strings(symbols)
	state := bookState{Books: make([]bookSnapshot, 0, len(symbols))}
	for _, symbol := range symbols {
		book := adapter.books[symbol]
		snapshot := bookSnapshot{Symbol: symbol, Bids: []restingOrder{}, Asks: []restingOrder{}}
		for _, native := range book.engine.RestingOrders() {
			source, ok := book.nativeToSource[native.ID]
			if !ok {
				return bookState{}, fmt.Errorf(
					"adapter has no source id for resting native order %d",
					native.ID,
				)
			}
			owner, ok := book.nativeToOwner[native.ID]
			if !ok {
				return bookState{}, fmt.Errorf(
					"adapter has no source owner for native order %d",
					native.ID,
				)
			}
			resting := restingOrder{
				OrderID: source, Price: adapter.config.formatPrice(native.Price),
				RemainingQuantity: adapter.config.formatQuantity(native.RemainingQty),
				Owner:             owner, OrderType: "LIMIT",
			}
			if native.Side == types.SideBuy {
				snapshot.Bids = append(snapshot.Bids, resting)
			} else {
				snapshot.Asks = append(snapshot.Asks, resting)
			}
		}
		state.Books = append(state.Books, snapshot)
	}
	return state, nil
}
