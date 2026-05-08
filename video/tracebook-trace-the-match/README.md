# Trace The Match

HyperFrames companion video source for `tracebook`.

## Composition

- Title: `Trace The Match`
- Composition id: `tracebook-video`
- Format: `1920x1080`, `30fps`, `62s`
- Visual identity: see `DESIGN.md`
- Main source: `index.html`
- Narration script: `script.txt`
- Timed voiceover source segments: `voiceover/scene-*.txt`
- Rendered narration asset: `media/narration.wav`
- Generated background music bed: `media/background-music.wav`

## Local Checks

Use HyperFrames `0.5.3`.

```bash
npx hyperframes doctor
npx hyperframes lint video/tracebook-trace-the-match
npx hyperframes validate video/tracebook-trace-the-match
npx hyperframes inspect video/tracebook-trace-the-match --samples 15
npx hyperframes inspect video/tracebook-trace-the-match --at 4,12,24,36,49,61,72,82,90
```

## Rendering

Rendered MP4 files are local artifacts and are intentionally ignored by git.

```bash
npx hyperframes render video/tracebook-trace-the-match --quality draft --output /private/tmp/tracebook-trace-the-match-draft.mp4
npx hyperframes render video/tracebook-trace-the-match --quality high --fps 30 --output /private/tmp/tracebook-trace-the-match-final.mp4
```

The README embed should wait until the final MP4 is uploaded to a stable host.
