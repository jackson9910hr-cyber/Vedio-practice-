// Usage: node render.js [frames.mp4] [--stills t1,t2,...]
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
const { spawn } = require('child_process');
const path = require('path'), fs = require('fs');
(async () => {
  const args = process.argv.slice(2);
  const browser = await chromium.launch({ args: ['--allow-file-access-from-files'] });
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
  page.on('console', m => console.log('[page]', m.text()));
  page.on('pageerror', e => console.error('[pageerror]', e.message));
  await page.goto('file://' + path.resolve(__dirname, 'index.html'));
  await page.evaluate(() => window.READY);
  const canvas = await page.$('#c');
  if (args[0] === '--stills') {
    fs.mkdirSync('stills', { recursive: true });
    for (const t of args[1].split(',').map(Number)) {
      await page.evaluate(t => render(t), t);
      await canvas.screenshot({ path: `stills/t${t.toFixed(2)}.png` });
    }
  } else {
    const out = args[0] || 'video.mp4';
    const ff = spawn('ffmpeg', ['-y', '-f', 'image2pipe', '-framerate', '30', '-i', '-',
      '-c:v', 'libx264', '-preset', 'slow', '-crf', '16', '-pix_fmt', 'yuv420p', out], { stdio: ['pipe', 'ignore', 'inherit'] });
    const t0 = Date.now();
    const N = await page.evaluate(() => Math.round(DUR * 30));
    for (let f = 0; f < N; f++) {
      await page.evaluate(t => render(t), f / 30);
      const buf = await canvas.screenshot({ type: 'png' });
      if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
      if (f % 50 === 0) console.log(`frame ${f} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    }
    ff.stdin.end();
    await new Promise(r => ff.on('close', r));
  }
  await browser.close();
})();
