"use strict";

// Poll the local server for order-book state and render a live depth ladder,
// top-of-book quote, engine metrics, and a trade tape. No frameworks.

const POLL_MS = 500;
const els = {
  symbol: document.getElementById("symbol"),
  status: document.getElementById("status"),
  statusText: document.getElementById("status-text"),
  bestBid: document.getElementById("best-bid"),
  bestAsk: document.getElementById("best-ask"),
  mid: document.getElementById("mid"),
  spread: document.getElementById("spread"),
  mThru: document.getElementById("m-thru"),
  mP50: document.getElementById("m-p50"),
  mP95: document.getElementById("m-p95"),
  mP99: document.getElementById("m-p99"),
  mOrders: document.getElementById("m-orders"),
  mTrades: document.getElementById("m-trades"),
  asks: document.getElementById("asks"),
  bids: document.getElementById("bids"),
  bookMid: document.getElementById("book-mid"),
  trades: document.getElementById("trades"),
};

let lastTradePrice = null;

function num(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function setStatus(state, text) {
  els.status.dataset.state = state;
  els.statusText.textContent = text;
}

function priceDigits(state) {
  // Infer a sensible price precision from the tick grid the server exposes.
  const t = state.tick_size;
  if (!t) return 2;
  const s = String(t);
  const dot = s.indexOf(".");
  return dot === -1 ? 0 : Math.min(s.length - dot - 1, 8);
}

function renderLadder(container, levels, side, maxSize, pd) {
  container.innerHTML = "";
  if (!levels || levels.length === 0) {
    container.innerHTML = `<div class="empty">no ${side}s</div>`;
    return;
  }
  const frag = document.createDocumentFragment();
  for (const [price, size, count] of levels) {
    const row = document.createElement("div");
    row.className = "row";
    const pct = maxSize > 0 ? Math.max(2, (size / maxSize) * 100) : 0;
    row.innerHTML =
      `<span class="bar" style="width:${pct}%"></span>` +
      `<span class="price">${num(price, pd)}</span>` +
      `<span class="size">${num(size, 4)}</span>` +
      `<span class="count">${count}</span>`;
    frag.appendChild(row);
  }
  container.appendChild(frag);
}

function renderTrades(trades, pd) {
  els.trades.innerHTML = "";
  if (!trades || trades.length === 0) {
    els.trades.innerHTML = `<div class="empty">no trades yet</div>`;
    return;
  }
  const frag = document.createDocumentFragment();
  let prev = null;
  // Server sends oldest-first; show newest at the top.
  for (const t of [...trades].reverse()) {
    const row = document.createElement("div");
    row.className = "trade";
    let dir = "flat";
    if (prev !== null) dir = t.price > prev ? "up" : t.price < prev ? "down" : "flat";
    prev = t.price;
    const time = t.timestamp
      ? new Date(t.timestamp / 1e6).toLocaleTimeString(undefined, { hour12: false })
      : "";
    row.innerHTML =
      `<span class="t-price ${dir}">${num(t.price, pd)}</span>` +
      `<span class="t-size">${num(t.quantity, 4)}</span>` +
      `<span class="t-time">${time}</span>`;
    frag.appendChild(row);
  }
  els.trades.appendChild(frag);
}

function render(state) {
  const pd = priceDigits(state);
  els.symbol.textContent = state.symbol || "—";
  els.bestBid.textContent = num(state.best_bid, pd);
  els.bestAsk.textContent = num(state.best_ask, pd);
  els.mid.textContent = num(state.mid, pd + 1);
  els.spread.textContent = num(state.spread, pd);
  els.bookMid.textContent = state.mid != null ? `mid ${num(state.mid, pd + 1)}` : "—";

  const s = state.stats || {};
  els.mThru.textContent = num(s.throughput, 0);
  els.mP50.textContent = num(s.latency_p50_ms, 3);
  els.mP95.textContent = num(s.latency_p95_ms, 3);
  els.mP99.textContent = num(s.latency_p99_ms, 3);
  els.mOrders.textContent = num(s.orders, 0);
  els.mTrades.textContent = num(s.trades, 0);

  const maxSize = Math.max(
    1e-9,
    ...(state.asks || []).map((l) => l[1]),
    ...(state.bids || []).map((l) => l[1]),
  );
  // Asks are best-first (lowest); show highest at the top of the ladder.
  renderLadder(els.asks, [...(state.asks || [])].reverse(), "ask", maxSize, pd);
  renderLadder(els.bids, state.bids || [], "bid", maxSize, pd);
  renderTrades(state.trades, pd);
}

async function tick() {
  try {
    const res = await fetch("api/state", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const state = await res.json();
    render(state);
    setStatus("live", "live");
  } catch (err) {
    setStatus("down", "disconnected");
  }
}

tick();
setInterval(tick, POLL_MS);
