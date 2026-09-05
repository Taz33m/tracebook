package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
)

const (
	protocolName    = "tracebook.conformance"
	protocolVersion = 1
	engineName      = "intrepidkarthi/orderbook CLOB"
	engineVersion   = "v0.26.0@51d480cdb68b"
)

// Optional build-time identity binds an evidence run to its captured source.
// Set with -ldflags '-X main.engineRevision=... -X main.engineSnapshot=...'.
// Empty defaults preserve the unbound qualification handshake.
var engineRevision string
var engineSnapshot string

func engineMetadata() map[string]string {
	metadata := map[string]string{"name": engineName, "version": engineVersion, "language": "Go"}
	if engineRevision != "" {
		metadata["revision"] = engineRevision
	}
	if engineSnapshot != "" {
		metadata["snapshot_id"] = engineSnapshot
	}
	return metadata
}

type configWire struct {
	MatchingAlgorithm     string  `json:"matching_algorithm"`
	TickSize              string  `json:"tick_size"`
	SelfTradePolicy       string  `json:"self_trade_policy"`
	QuantityDecimalPlaces *uint32 `json:"quantity_decimal_places"`
}

type helloFrame struct {
	Type            string     `json:"type"`
	Protocol        string     `json:"protocol"`
	ProtocolVersion int        `json:"protocol_version"`
	Config          configWire `json:"config"`
}

type marketEvent struct {
	Op          string       `json:"op"`
	Symbol      string       `json:"symbol"`
	OrderID     *json.Number `json:"order_id"`
	Side        *string      `json:"side"`
	OrderType   string       `json:"order_type"`
	Price       *json.Number `json:"price"`
	Quantity    *json.Number `json:"quantity"`
	Owner       *int64       `json:"owner"`
	TimestampNS *uint64      `json:"timestamp_ns"`
}

type eventFrame struct {
	Type  string      `json:"type"`
	Index uint64      `json:"index"`
	Event marketEvent `json:"event"`
}

type outcome struct {
	Status  string  `json:"status"`
	Reason  *string `json:"reason"`
	Message *string `json:"message"`
}

func applied() outcome { return outcome{Status: "applied"} }

func rejected(reason, message string) outcome {
	return outcome{Status: "rejected", Reason: &reason, Message: &message}
}

type tradeFill struct {
	Symbol      string `json:"symbol"`
	BuyOrderID  int64  `json:"buy_order_id"`
	SellOrderID int64  `json:"sell_order_id"`
	Price       string `json:"price"`
	Quantity    string `json:"quantity"`
}

type restingOrder struct {
	OrderID           int64  `json:"order_id"`
	Price             string `json:"price"`
	RemainingQuantity string `json:"remaining_quantity"`
	Owner             int64  `json:"owner"`
	OrderType         string `json:"order_type"`
}

type bookSnapshot struct {
	Symbol string         `json:"symbol"`
	Bids   []restingOrder `json:"bids"`
	Asks   []restingOrder `json:"asks"`
}

type bookState struct {
	Books []bookSnapshot `json:"books"`
}

func (state bookState) orderCount() int {
	count := 0
	for _, book := range state.Books {
		count += len(book.Bids) + len(book.Asks)
	}
	return count
}

func (state bookState) digest() (string, error) {
	raw, err := json.Marshal(state)
	if err != nil {
		return "", err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return "", err
	}
	canonical, err := canonicalJSON(value)
	if err != nil {
		return "", err
	}
	hash := sha256.Sum256(canonical)
	return hex.EncodeToString(hash[:]), nil
}

type observationFrame struct {
	Type              string      `json:"type"`
	Index             uint64      `json:"index"`
	Outcome           outcome     `json:"outcome"`
	Trades            []tradeFill `json:"trades"`
	StateHash         string      `json:"state_hash"`
	RestingOrderCount int         `json:"resting_order_count"`
}

func canonicalJSON(value any) ([]byte, error) {
	switch typed := value.(type) {
	case nil, bool, string, json.Number:
		return json.Marshal(typed)
	case []any:
		var buffer bytes.Buffer
		buffer.WriteByte('[')
		for index, item := range typed {
			if index > 0 {
				buffer.WriteByte(',')
			}
			encoded, err := canonicalJSON(item)
			if err != nil {
				return nil, err
			}
			buffer.Write(encoded)
		}
		buffer.WriteByte(']')
		return buffer.Bytes(), nil
	case map[string]any:
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		var buffer bytes.Buffer
		buffer.WriteByte('{')
		for index, key := range keys {
			if index > 0 {
				buffer.WriteByte(',')
			}
			encodedKey, _ := json.Marshal(key)
			buffer.Write(encodedKey)
			buffer.WriteByte(':')
			encodedValue, err := canonicalJSON(typed[key])
			if err != nil {
				return nil, err
			}
			buffer.Write(encodedValue)
		}
		buffer.WriteByte('}')
		return buffer.Bytes(), nil
	default:
		return nil, fmt.Errorf("unsupported canonical JSON value %T", value)
	}
}

type serverError struct {
	code    string
	message string
}

func (err *serverError) Error() string { return err.message }

func protocolError(message string) error {
	return &serverError{code: "PROTOCOL_ERROR", message: message}
}

func adapterError(err error) error {
	return &serverError{code: "ADAPTER_ERROR", message: err.Error()}
}

func readObject(decoder *json.Decoder) (map[string]json.RawMessage, error) {
	var object map[string]json.RawMessage
	if err := decoder.Decode(&object); err != nil {
		return nil, err
	}
	if object == nil {
		return nil, errors.New("protocol message must be an object")
	}
	return object, nil
}

func frameType(object map[string]json.RawMessage) (string, error) {
	var value string
	if err := json.Unmarshal(object["type"], &value); err != nil || value == "" {
		return "", errors.New("protocol message requires a string type")
	}
	return value, nil
}

func decodeObject(object map[string]json.RawMessage, target any) error {
	raw, err := json.Marshal(object)
	if err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	return decoder.Decode(target)
}

func writeFrame(writer *bufio.Writer, value any) error {
	if err := json.NewEncoder(writer).Encode(value); err != nil {
		return err
	}
	return writer.Flush()
}

func serve(input io.Reader, output io.Writer) int {
	decoder := json.NewDecoder(input)
	decoder.UseNumber()
	writer := bufio.NewWriter(output)
	err := runServer(decoder, writer)
	if err == nil {
		return 0
	}
	var typed *serverError
	if !errors.As(err, &typed) {
		typed = &serverError{code: "PROTOCOL_ERROR", message: err.Error()}
	}
	_ = writeFrame(writer, map[string]any{
		"type": "error", "code": typed.code, "message": typed.message,
	})
	return 2
}

func runServer(decoder *json.Decoder, writer *bufio.Writer) error {
	first, err := readObject(decoder)
	if err != nil {
		return protocolError("expected hello message: " + err.Error())
	}
	var hello helloFrame
	if err := decodeObject(first, &hello); err != nil {
		return protocolError(err.Error())
	}
	if hello.Type != "hello" {
		return protocolError("first message must be hello")
	}
	if hello.Protocol != protocolName {
		return protocolError(fmt.Sprintf("protocol must be %q", protocolName))
	}
	if hello.ProtocolVersion != protocolVersion {
		return protocolError(fmt.Sprintf("protocol_version must be %d", protocolVersion))
	}
	adapter, err := newAdapter(hello.Config)
	if err != nil {
		return protocolError(err.Error())
	}
	if err := writeFrame(writer, map[string]any{
		"type":             "ready",
		"protocol":         protocolName,
		"protocol_version": protocolVersion,
		"engine":           engineMetadata(),
	}); err != nil {
		return adapterError(err)
	}

	var lastIndex uint64
	for {
		object, err := readObject(decoder)
		if err != nil {
			if errors.Is(err, io.EOF) {
				return protocolError("protocol ended before finish")
			}
			return protocolError(err.Error())
		}
		kind, err := frameType(object)
		if err != nil {
			return protocolError(err.Error())
		}
		switch kind {
		case "event":
			if raw := object["event"]; len(raw) == 0 || bytes.Equal(bytes.TrimSpace(raw), []byte("null")) {
				return protocolError("event payload must be an object")
			}
			var frame eventFrame
			if err := decodeObject(object, &frame); err != nil {
				return protocolError(err.Error())
			}
			if frame.Index == 0 || frame.Index != lastIndex+1 {
				return protocolError("event indexes must be contiguous and start at 1")
			}
			observation, err := adapter.apply(frame.Event, frame.Index)
			if err != nil {
				return adapterError(err)
			}
			lastIndex = frame.Index
			if err := writeFrame(writer, observation); err != nil {
				return adapterError(err)
			}
		case "snapshot":
			var frame struct {
				Index *uint64 `json:"index"`
			}
			if err := decodeObject(object, &frame); err != nil {
				return protocolError(err.Error())
			}
			if frame.Index == nil || *frame.Index != lastIndex {
				return protocolError("snapshot index does not match the last event")
			}
			state, err := adapter.snapshot()
			if err != nil {
				return adapterError(err)
			}
			if err := writeFrame(writer, map[string]any{
				"type": "snapshot", "index": lastIndex, "state": state,
			}); err != nil {
				return adapterError(err)
			}
		case "finish":
			var frame struct {
				EventCount *uint64 `json:"event_count"`
			}
			if err := decodeObject(object, &frame); err != nil {
				return protocolError(err.Error())
			}
			if frame.EventCount == nil || *frame.EventCount != lastIndex {
				return protocolError("finish event_count does not match the last event")
			}
			return writeFrame(writer, map[string]any{
				"type": "complete", "event_count": lastIndex,
			})
		default:
			return protocolError(fmt.Sprintf("unsupported protocol message type: %q", kind))
		}
	}
}
