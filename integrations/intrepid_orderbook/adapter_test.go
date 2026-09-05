package main

import (
	"bytes"
	"encoding/json"
	"math"
	"reflect"
	"strings"
	"testing"
)

func number(value string) *json.Number {
	number := json.Number(value)
	return &number
}

func textPointer(value string) *string { return &value }

func int64Pointer(value int64) *int64 { return &value }

func testAdapter(t *testing.T, config configWire) *adapter {
	t.Helper()
	harness, err := newAdapter(config)
	if err != nil {
		t.Fatal(err)
	}
	return harness
}

func applyForTest(t *testing.T, harness *adapter, event marketEvent) observationFrame {
	t.Helper()
	observation, err := harness.apply(event, 1)
	if err != nil {
		t.Fatal(err)
	}
	return observation
}

func limitEvent(id, side, price, quantity string) marketEvent {
	return marketEvent{Op: "new", Symbol: "TEST", OrderID: number(id),
		Side: textPointer(side), OrderType: "LIMIT", Price: number(price), Quantity: number(quantity)}
}

func TestEmptyStateDigestMatchesProtocolCanonicalJSON(t *testing.T) {
	digest, err := (bookState{Books: []bookSnapshot{}}).digest()
	if err != nil {
		t.Fatal(err)
	}
	const expected = "dd8681e973bdf802edf26260297e68e24cdbd75c782c90bbaaaddd401df51090"
	if digest != expected {
		t.Fatalf("digest = %q, want %q", digest, expected)
	}
}

func TestNativeCrossingPreservesFIFOAfterReduction(t *testing.T) {
	harness, err := newAdapter(configWire{})
	if err != nil {
		t.Fatal(err)
	}
	events := []marketEvent{
		{Op: "new", Symbol: "TEST", OrderID: number("1"), Side: textPointer("BUY"),
			OrderType: "LIMIT", Price: number("100"), Quantity: number("5"), Owner: int64Pointer(10)},
		{Op: "new", Symbol: "TEST", OrderID: number("2"), Side: textPointer("BUY"),
			OrderType: "LIMIT", Price: number("100"), Quantity: number("3"), Owner: int64Pointer(20)},
		{Op: "reduce", Symbol: "TEST", OrderID: number("1"), Quantity: number("2")},
		{Op: "new", Symbol: "TEST", OrderID: number("3"), Side: textPointer("SELL"),
			OrderType: "LIMIT", Price: number("100"), Quantity: number("4"), Owner: int64Pointer(30)},
	}
	var observation observationFrame
	for index, event := range events {
		observation, err = harness.apply(event, uint64(index+1))
		if err != nil {
			t.Fatalf("event %d: %v", index+1, err)
		}
		if observation.Outcome.Status != "applied" {
			t.Fatalf("event %d outcome = %#v", index+1, observation.Outcome)
		}
	}

	if len(observation.Trades) != 2 {
		t.Fatalf("trades = %#v, want two native fills", observation.Trades)
	}
	if observation.Trades[0].BuyOrderID != 1 || observation.Trades[0].Quantity != "3" {
		t.Fatalf("first fill = %#v, want reduced order 1 for quantity 3", observation.Trades[0])
	}
	if observation.Trades[1].BuyOrderID != 2 || observation.Trades[1].Quantity != "1" {
		t.Fatalf("second fill = %#v, want order 2 for quantity 1", observation.Trades[1])
	}
	state, err := harness.snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(state.Books) != 1 || len(state.Books[0].Bids) != 1 {
		t.Fatalf("state = %#v, want one residual bid", state)
	}
	remaining := state.Books[0].Bids[0]
	if remaining.OrderID != 2 || remaining.RemainingQuantity != "2" {
		t.Fatalf("remaining bid = %#v, want order 2 quantity 2", remaining)
	}
}

func TestUnfillableFOKPreservesNativeRejection(t *testing.T) {
	harness, err := newAdapter(configWire{})
	if err != nil {
		t.Fatal(err)
	}
	observation, err := harness.apply(marketEvent{
		Op: "new", Symbol: "TEST", OrderID: number("1"), Side: textPointer("BUY"),
		OrderType: "FOK", Price: number("100"), Quantity: number("1"), Owner: int64Pointer(10),
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if observation.Outcome.Status != "rejected" || observation.Outcome.Reason == nil ||
		*observation.Outcome.Reason != "INVALID_ORDER" {
		t.Fatalf("outcome = %#v, want the upstream FOK rejection", observation.Outcome)
	}
	if observation.RestingOrderCount != 0 || len(observation.Trades) != 0 {
		t.Fatalf("observation = %#v, want no book mutation", observation)
	}
}

func TestQuantityBeyondNativeSixPlaceBoundaryIsRejected(t *testing.T) {
	places := uint32(12)
	harness, err := newAdapter(configWire{QuantityDecimalPlaces: &places})
	if err != nil {
		t.Fatal(err)
	}
	observation, err := harness.apply(marketEvent{
		Op: "new", Symbol: "TEST", OrderID: number("1"), Side: textPointer("BUY"),
		OrderType: "LIMIT", Price: number("100"), Quantity: number("0.0000001"), Owner: int64Pointer(10),
	}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if observation.Outcome.Status != "rejected" || observation.Outcome.Message == nil ||
		!strings.Contains(*observation.Outcome.Message, "more than 6 native decimal places") {
		t.Fatalf("outcome = %#v, want an explicit native-precision rejection", observation.Outcome)
	}
}

func TestServerHandshakeAndEmptyCompletion(t *testing.T) {
	input := strings.Join([]string{
		`{"type":"hello","protocol":"tracebook.conformance","protocol_version":1,"config":{"matching_algorithm":"fifo","tick_size":"0.01","self_trade_policy":"NONE","quantity_decimal_places":12}}`,
		`{"type":"finish","event_count":0}`,
		"",
	}, "\n")
	var output bytes.Buffer
	if status := serve(strings.NewReader(input), &output); status != 0 {
		t.Fatalf("status = %d, output = %s", status, output.String())
	}
	decoder := json.NewDecoder(&output)
	var ready map[string]any
	var complete map[string]any
	if err := decoder.Decode(&ready); err != nil {
		t.Fatal(err)
	}
	if err := decoder.Decode(&complete); err != nil {
		t.Fatal(err)
	}
	engine := ready["engine"].(map[string]any)
	if ready["type"] != "ready" || engine["name"] != engineName || engine["version"] != engineVersion {
		t.Fatalf("ready = %#v", ready)
	}
	if _, ok := engine["revision"]; ok {
		t.Fatal("default handshake unexpectedly binds a revision")
	}
	if _, ok := engine["snapshot_id"]; ok {
		t.Fatal("default handshake unexpectedly binds a snapshot")
	}
	if complete["type"] != "complete" || complete["event_count"] != float64(0) {
		t.Fatalf("complete = %#v", complete)
	}
}

func TestServerHandshakeReportsInjectedEvidenceIdentity(t *testing.T) {
	previousRevision, previousSnapshot := engineRevision, engineSnapshot
	t.Cleanup(func() {
		engineRevision, engineSnapshot = previousRevision, previousSnapshot
	})
	engineRevision = "reviewed-source-revision"
	engineSnapshot = "sha256:captured-source-snapshot"
	input := `{"type":"hello","protocol":"tracebook.conformance","protocol_version":1,"config":{}}
{"type":"finish","event_count":0}`
	var output bytes.Buffer
	if status := serve(strings.NewReader(input), &output); status != 0 {
		t.Fatalf("status = %d, output = %s", status, output.String())
	}
	var ready struct {
		Engine map[string]string `json:"engine"`
	}
	if err := json.NewDecoder(&output).Decode(&ready); err != nil {
		t.Fatal(err)
	}
	if ready.Engine["revision"] != engineRevision || ready.Engine["snapshot_id"] != engineSnapshot ||
		ready.Engine["name"] != engineName || ready.Engine["version"] != engineVersion {
		t.Fatalf("bound identity = %#v", ready.Engine)
	}
}

func TestAnonymousOwnersRemainExemptFromNativeSTPAfterReplace(t *testing.T) {
	for _, algorithm := range []string{"fifo", "pro_rata"} {
		for _, policy := range []string{"CANCEL_RESTING", "CANCEL_INCOMING"} {
			for _, test := range []struct {
				name       string
				makerOwner *int64
				takerOwner *int64
				selfMatch  bool
			}{
				{"omitted", nil, nil, false},
				{"explicit-anonymous", int64Pointer(-1), int64Pointer(-1), false},
				{"mixed-anonymous", nil, int64Pointer(-1), false},
				{"anonymous-vs-numeric", int64Pointer(-1), int64Pointer(1), false},
				{"same-numeric", int64Pointer(1), int64Pointer(1), true},
				{"same-negative-numeric", int64Pointer(-2), int64Pointer(-2), true},
			} {
				t.Run(algorithm+"/"+policy+"/"+test.name, func(t *testing.T) {
					harness := testAdapter(t, configWire{MatchingAlgorithm: algorithm, SelfTradePolicy: policy})
					maker := limitEvent("1", "BUY", "100", "2")
					maker.Owner = test.makerOwner
					for _, event := range []marketEvent{
						maker,
						{Op: "replace", Symbol: "TEST", OrderID: number("1"), Quantity: number("3")},
						{Op: "reduce", Symbol: "TEST", OrderID: number("1"), Quantity: number("1")},
					} {
						if got := applyForTest(t, harness, event); got.Outcome.Status != "applied" {
							t.Fatalf("%s outcome = %#v", event.Op, got.Outcome)
						}
					}
					state, err := harness.snapshot()
					if err != nil {
						t.Fatal(err)
					}
					if got := state.Books[0].Bids[0].Owner; got != eventOwner(maker) {
						t.Fatalf("owner after replace/reduce = %d, want %d", got, eventOwner(maker))
					}
					taker := limitEvent("2", "SELL", "100", "1")
					taker.Owner = test.takerOwner
					got := applyForTest(t, harness, taker)
					if got.Outcome.Status != "applied" {
						t.Fatalf("taker outcome = %#v", got.Outcome)
					}
					if !test.selfMatch {
						if len(got.Trades) != 1 || got.Trades[0].BuyOrderID != 1 || got.Trades[0].Quantity != "1" {
							t.Fatalf("anonymous crossing = %#v, want a native fill", got)
						}
					} else if len(got.Trades) != 0 {
						t.Fatalf("same-owner order traded under %s: %#v", policy, got.Trades)
					}
					activeID := "1"
					if test.selfMatch && policy == "CANCEL_RESTING" {
						activeID = "2"
					}
					cancel := applyForTest(t, harness, marketEvent{Op: "cancel", Symbol: "TEST", OrderID: number(activeID)})
					if cancel.Outcome.Status != "applied" || cancel.RestingOrderCount != 0 {
						t.Fatalf("cancel after STP = %#v", cancel)
					}
					book := harness.books["TEST"]
					if len(book.nativeToOwner) != 0 || len(book.nativeToSource) != 0 || len(book.sourceToNative) != 0 {
						t.Fatal("inactive source and owner mappings were retained")
					}
				})
			}
		}
	}
}

func TestQuantityPrecisionOnlyRoundsObservations(t *testing.T) {
	places := uint32(0)
	harness := testAdapter(t, configWire{QuantityDecimalPlaces: &places})
	first := applyForTest(t, harness, limitEvent("1", "BUY", "100", "2.4"))
	if first.Outcome.Status != "applied" {
		t.Fatalf("first outcome = %#v", first.Outcome)
	}
	if got := harness.books["TEST"].engine.RestingOrders()[0].RemainingQty; got != 2_400_000 {
		t.Fatalf("native quantity = %d, want the unrounded input 2400000", got)
	}
	second := applyForTest(t, harness, limitEvent("2", "SELL", "100", "3.5"))
	if second.Outcome.Status != "applied" || len(second.Trades) != 1 || second.Trades[0].Quantity != "2" {
		t.Fatalf("crossing = %#v, want the native 2.4 fill reported at zero decimal places", second)
	}
	state, err := harness.snapshot()
	if err != nil {
		t.Fatal(err)
	}
	if len(state.Books[0].Asks) != 1 || state.Books[0].Asks[0].RemainingQuantity != "1" {
		t.Fatalf("state = %#v, want 1.1 native quantity reported as 1", state)
	}
	if got := harness.books["TEST"].engine.RestingOrders()[0].RemainingQty; got != 1_100_000 {
		t.Fatalf("native remainder = %d, want 1100000", got)
	}
	if _, err := harness.config.quantityUnits(number("1.0000001")); err == nil {
		t.Fatal("input beyond six native places must not be rounded into range")
	}
}

func TestPriceTicksMatchReferenceBinary64Snapping(t *testing.T) {
	config := testAdapter(t, configWire{}).config
	for _, test := range []struct {
		price string
		ticks int64
	}{
		{"1.015", 101}, // Binary64 division is just below 101.5.
		{"0.105", 10},
		{"0.115", 12},
		{"100.004", 10000},
		{"100.006", 10001},
	} {
		t.Run(test.price, func(t *testing.T) {
			got, err := config.priceTicks(number(test.price))
			if err != nil || got != test.ticks {
				t.Fatalf("ticks = %d, %v; want %d", got, err, test.ticks)
			}
		})
	}
	for _, price := range []string{"0.005", "0", "-1", "NaN", "Inf", "1e309", "92233720368547758.08"} {
		if _, err := config.priceTicks(number(price)); err == nil {
			t.Errorf("unsupported price %s was accepted", price)
		}
	}
	for _, tick := range []string{"NaN", "Inf", "1e309", "1e-330"} {
		if _, err := parseConfig(configWire{TickSize: tick}); err == nil {
			t.Errorf("unsupported tick %s was accepted", tick)
		}
	}
}

func TestNativeQuantityIntegerBoundary(t *testing.T) {
	config := testAdapter(t, configWire{}).config
	got, err := config.quantityUnits(number("9223372036854.775807"))
	if err != nil || got != math.MaxInt64 {
		t.Fatalf("maximum native quantity = %d, %v", got, err)
	}
	for _, quantity := range []string{"9223372036854.775808", "0", "-1", "NaN"} {
		if _, err := config.quantityUnits(number(quantity)); err == nil {
			t.Errorf("unsupported quantity %s was accepted", quantity)
		}
	}
}

func TestUnsafeProRataProductIsRejectedBeforeMutation(t *testing.T) {
	for _, orderType := range []string{"LIMIT", "MARKET", "IOC", "FOK"} {
		t.Run(orderType, func(t *testing.T) {
			harness := testAdapter(t, configWire{MatchingAlgorithm: "pro_rata"})
			maker := applyForTest(t, harness, limitEvent("1", "SELL", "100", "4000"))
			taker := limitEvent("2", "BUY", "100", "4000")
			taker.OrderType = orderType
			got := applyForTest(t, harness, taker)
			if got.Outcome.Status != "rejected" || got.Outcome.Message == nil ||
				!strings.Contains(*got.Outcome.Message, "pro-rata product range") {
				t.Fatalf("unsafe product outcome = %#v", got.Outcome)
			}
			if got.StateHash != maker.StateHash || len(got.Trades) != 0 {
				t.Fatalf("unsafe product mutated state: %#v", got)
			}
		})
	}
	for _, test := range []struct{ algorithm, quantity string }{
		{"fifo", "4000"}, {"pro_rata", "3000"},
	} {
		t.Run(test.algorithm+"/safe-product", func(t *testing.T) {
			harness := testAdapter(t, configWire{MatchingAlgorithm: test.algorithm})
			applyForTest(t, harness, limitEvent("1", "SELL", "100", test.quantity))
			got := applyForTest(t, harness, limitEvent("2", "BUY", "100", test.quantity))
			if got.Outcome.Status != "applied" || len(got.Trades) != 1 || got.Trades[0].Quantity != test.quantity {
				t.Fatalf("supported large crossing = %#v", got)
			}
		})
	}
}

func TestSideAggregateAndReplacementCapacityPreserveExistingOrders(t *testing.T) {
	harness := testAdapter(t, configWire{})
	first := applyForTest(t, harness, limitEvent("1", "BUY", "0.01", "4700000000000"))
	if first.Outcome.Status != "applied" {
		t.Fatalf("large first order = %#v", first)
	}
	tooLarge := applyForTest(t, harness, limitEvent("2", "BUY", "0.01", "4700000000000"))
	if tooLarge.Outcome.Status != "rejected" || tooLarge.StateHash != first.StateHash ||
		tooLarge.Outcome.Message == nil || !strings.Contains(*tooLarge.Outcome.Message, "side aggregate range") {
		t.Fatalf("aggregate overflow = %#v", tooLarge)
	}
	second := applyForTest(t, harness, limitEvent("2", "BUY", "0.01", "4000000000000"))
	if second.Outcome.Status != "applied" {
		t.Fatalf("supported aggregate = %#v", second)
	}
	replace := applyForTest(t, harness, marketEvent{Op: "replace", Symbol: "TEST",
		OrderID: number("2"), Quantity: number("4700000000000")})
	if replace.Outcome.Reason == nil || *replace.Outcome.Reason != "INVALID_REPLACEMENT" ||
		replace.StateHash != second.StateHash {
		t.Fatalf("unsafe replacement = %#v", replace)
	}
	noChange := applyForTest(t, harness, marketEvent{Op: "replace", Symbol: "TEST",
		OrderID: number("2"), Quantity: number("4000000000000")})
	if noChange.Outcome.Status != "applied" {
		t.Fatalf("replacement must exclude the old quantity from its capacity check: %#v", noChange)
	}
}

func TestNativeNotionalLimitAppliesOnlyToPriceBearingInstructions(t *testing.T) {
	for _, orderType := range []string{"LIMIT", "IOC", "FOK", "MARKET"} {
		t.Run(orderType, func(t *testing.T) {
			harness := testAdapter(t, configWire{})
			event := limitEvent("1", "BUY", "100000", "1000000")
			event.OrderType = orderType
			got := applyForTest(t, harness, event)
			if orderType == "MARKET" {
				if got.Outcome.Status != "applied" || got.RestingOrderCount != 0 {
					t.Fatalf("market order should retain the native notional exemption: %#v", got)
				}
			} else if got.Outcome.Status != "rejected" || got.Outcome.Message == nil ||
				!strings.Contains(*got.Outcome.Message, "order notional") || got.RestingOrderCount != 0 {
				t.Fatalf("native notional overflow = %#v", got)
			}
		})
	}
}

func TestNotionalOverflowReplacementPreservesNativeOrderAndSourceMetadata(t *testing.T) {
	for _, test := range []struct {
		name  string
		owner *int64
	}{{"anonymous", nil}, {"identified", int64Pointer(8)}} {
		t.Run(test.name, func(t *testing.T) {
			harness := testAdapter(t, configWire{})
			event := limitEvent("1", "BUY", "100", "1")
			event.Owner = test.owner
			first := applyForTest(t, harness, event)
			book := harness.books["TEST"]
			original := *book.engine.RestingOrders()[0]
			got := applyForTest(t, harness, marketEvent{
				Op: "replace", Symbol: "TEST", OrderID: number("1"),
				Price: number("100000"), Quantity: number("1000000"),
			})
			if got.Outcome.Reason == nil || *got.Outcome.Reason != "INVALID_REPLACEMENT" ||
				got.Outcome.Message == nil || !strings.Contains(*got.Outcome.Message, "order notional") {
				t.Fatalf("overflow replacement outcome = %#v", got.Outcome)
			}
			if got.StateHash != first.StateHash || got.RestingOrderCount != 1 || len(got.Trades) != 0 {
				t.Fatalf("overflow replacement changed the source state: %#v", got)
			}
			orders := book.engine.RestingOrders()
			if len(orders) != 1 || !reflect.DeepEqual(original, *orders[0]) {
				t.Fatalf("overflow replacement changed the native order: %#v", orders)
			}
			if len(book.sourceToNative) != 1 || book.sourceToNative[1] != original.ID ||
				len(book.nativeToSource) != 1 || book.nativeToSource[original.ID] != 1 ||
				len(book.nativeToOwner) != 1 || book.nativeToOwner[original.ID] != eventOwner(event) {
				t.Fatal("overflow replacement changed source identity metadata")
			}
		})
	}
}

func TestHighOutputPrecisionPreservesNativeDecimalRemainder(t *testing.T) {
	places := uint32(18)
	harness := testAdapter(t, configWire{QuantityDecimalPlaces: &places})
	applyForTest(t, harness, limitEvent("1", "BUY", "100", "0.3"))
	got := applyForTest(t, harness, limitEvent("2", "SELL", "100", "0.1"))
	state, err := harness.snapshot()
	if err != nil {
		t.Fatal(err)
	}
	// The binary64 reference retains 0.19999999999999998 at 18 places.
	// Preserve the native arithmetic here so this known difference is visible.
	if got.Outcome.Status != "applied" || len(state.Books[0].Bids) != 1 ||
		state.Books[0].Bids[0].RemainingQuantity != "0.2" {
		t.Fatalf("native decimal remainder was not preserved: %#v", state)
	}
}

func TestFineTickOutputPreservesNativeIntegerPrice(t *testing.T) {
	harness := testAdapter(t, configWire{TickSize: "0.000000000000000001"})
	got := applyForTest(t, harness, limitEvent("1", "BUY", "1", "0.000001"))
	state, err := harness.snapshot()
	if err != nil {
		t.Fatal(err)
	}
	// The reference multiplies and rounds in binary64, yielding
	// 0.9999999999999999. Native tick output must not imitate that arithmetic.
	if got.Outcome.Status != "applied" || len(state.Books[0].Bids) != 1 ||
		state.Books[0].Bids[0].Price != "0.999999999999999872" {
		t.Fatalf("native fine-grid price was not preserved: %#v", state)
	}
}
