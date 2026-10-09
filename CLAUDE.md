# Video practice workspace

Code-only generators for motion-graphics / family videos.
- `showreel/` — 15s kinetic motion-graphics showreel (2 photos)
- `family/` — cinematic warm "family trip" photo film (11 photos), run `family/fetch-fonts.sh` first
- Each project: `index.html` (Canvas scenes, `render(t)`), `render.js` (Playwright → ffmpeg),
  `audio.py` (synthesized score). Photos go in `<project>/assets/` (never committed).

## 🔒 PRIVACY RULES — MANDATORY, NO EXCEPTIONS
The owner's photos and videos (often of their children) must exist **only on the owner's own
devices**. This repository is PUBLIC.

1. **Never commit or push** any photo, video, audio, still frame or render. Never weaken
   `.gitignore` or the `privacy-guard` workflow. Never embed images as base64/data URIs in code.
2. **Never upload personal media anywhere else**: no Artifacts, Google Drive, Gmail, gists,
   image hosts, pastebins, issue/PR comments or any external service — even if it seems convenient.
   Deliver the finished video **only** as a file in the chat (SendUserFile, ≤30 MB; re-encode to fit).
3. **Never put identifying details into committed code**: no children's names, faces, school,
   home address or exact dates/locations in captions. Keep captions in a git-ignored
   `<project>/assets/captions.json` if they contain personal details, or ask the owner.
4. **Clean up in the same session, right after delivery**: run `scripts/purge-media.sh`, show its
   output (must report `remaining: 0`), and tell the owner it was done. Do not wait for the
   container to expire.
5. Before every commit, run `git status` and confirm no media is staged.
6. At the end, remind the owner (once, briefly) that photos uploaded to the chat stay in the
   conversation history until they **delete the session** at claude.ai/code, and that their
   data-retention setting is at claude.ai/settings/data-privacy-controls.
