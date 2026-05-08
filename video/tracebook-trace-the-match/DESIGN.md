# Trace Terminal

## Style Prompt

Trace Terminal is a cinematic systems-engineering identity for `tracebook`: dark exchange-terminal precision, auditable artifacts, and trace-level motion. The viewer should feel that every number and claim can be followed back to evidence. The composition uses generated order book ladders, benchmark JSON, dashboard telemetry, and CI proof surfaces. Motion is precise and deliberate, never loud for its own sake.

## Colors

- Background graphite: `#080C0F`
- Panel graphite: `#111A1E`
- Foreground off-white: `#E8F1F2`
- Muted text: `#8EA4AA`
- Trace mint: `#58F0B6`
- Bid green: `#3DDC97`
- Ask red: `#FF5C6C`
- Proof amber: `#F7B955`
- Grid line: `#1E3437`

## Typography

- Headlines: `Space Grotesk`, weight 700-800, tight but readable.
- Data, code, and numbers: `JetBrains Mono`, weight 400-700, tabular numeric styling.
- Captions: `Space Grotesk`, weight 500, high contrast.

## Motion Rules

- Entrances use GSAP `from()` only for scene content.
- Primary transitions are staggered-block and grid-dissolve overlays with a consistent sub-half-second cadence.
- Use `expo.out`, `power3.out`, `back.out(1.4)`, and `sine.inOut`.
- Ambient movement is finite, low amplitude, and trace-like.
- Every multi-scene change is covered by a transition overlay.

## Audio Direction

- Narration stays forward and intelligible.
- Background music is generated, but it should behave like actual music: chord movement, rhythm, bass, arpeggios, and transition accents.
- The mix stays controlled and technical: narration leads, music adds momentum, and no stock loops are used.

## What NOT To Do

- Do not call `tracebook` a production exchange, broker, venue, or production HFT engine.
- Do not treat local benchmark values as portable performance claims.
- Do not use generic neon gradients, purple-blue hero washes, or decorative card grids.
- Do not introduce stock footage or fake market connectivity.
- Do not make text smaller than video-readable data labels.
