package main

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func number(value string) *json.Number {
	number := json.Number(value)
	return &number
}

func textPointer(value string) *string { return &value }

func int64Pointer(value int64) *int64 { return &value }

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
	if complete["type"] != "complete" || complete["event_count"] != float64(0) {
		t.Fatalf("complete = %#v", complete)
	}
}
