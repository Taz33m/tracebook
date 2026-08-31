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
	places := uint32(12)
	if wire.QuantityDecimalPlaces != nil {
		places = *wire.QuantityDecimalPlaces
	}
	if places > 18 {
		return adapterConfig{}, errors.New("quantity_decimal_places must be between 0 and 18")
	}
	// The upstream pro-rata path multiplies two int64 lot counts. Scaling all
	// protocol quantities by 10^12 would overflow that native multiplication on
	// ordinary five-lot fixtures. Six decimal places preserve the campaign's
	// exact values while keeping its bounded native arithmetic safely in range.
	nativePlaces := min(places, uint32(6))
	scale := int64(1)
	for range nativePlaces {
		if scale > math.MaxInt64/10 {
			return adapterConfig{}, errors.New("quantity scale exceeds int64")
		}
		scale *= 10
	}
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
	return adapterConfig{algorithm, tick, places, nativePlaces, scale, stp}, nil
}

func (config adapterConfig) priceTicks(value *jsonNumber) (int64, error) {
	if value == nil {
		return 0, errors.New("price is required")
	}
	price, err := decimal.NewFromString(value.String())
	if err != nil {
		return 0, errors.New("price must be finite")
	}
	ticks := price.Div(config.tickSize).RoundBank(0).BigInt()
	if !ticks.IsInt64() || ticks.Sign() <= 0 {
		return 0, errors.New("price snaps to a non-positive or unsupported tick")
	}
	return ticks.Int64(), nil
}

// jsonNumber is an alias so conversion helpers read cleanly without allowing
// floating-point parsing into this integer-tick boundary.
type jsonNumber = json.Number

func (config adapterConfig) quantityUnits(value *jsonNumber) (int64, error) {
	if value == nil {
		return 0, errors.New("quantity is required")
	}
	quantity, err := decimal.NewFromString(value.String())
	if err != nil || !quantity.IsPositive() {
		return 0, errors.New("quantity must be positive")
	}
	normalized := quantity.RoundBank(int32(config.quantityPlaces))
	scaled := normalized.Mul(
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
	return decimal.NewFromInt(units).Div(decimal.NewFromInt(config.quantityScale)).String()
}

type bookHarness struct {
	engine         *matching.Engine
	sourceToNative map[int64]int64
	nativeToSource map[int64]int64
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
	owner := strconv.FormatInt(eventOwner(event), 10)
	order, err := types.NewOrder(owner, event.Symbol, side, nativeType, price, quantity, tif)
	if err != nil {
		return rejected("INVALID_ORDER", err.Error()), nil, nil
	}
	order.ClientOrderID = strconv.FormatInt(source, 10)
	match := book.engine.Process(order)
	book.nativeToSource[order.ID] = source
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
			owner, err := strconv.ParseInt(native.UserID, 10, 64)
			if err != nil {
				return bookState{}, fmt.Errorf(
					"adapter has invalid source owner for native order %d: %w",
					native.ID,
					err,
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
