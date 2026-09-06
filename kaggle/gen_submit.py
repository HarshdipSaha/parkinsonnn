"""Emit .playwright-mcp/dd_submit.js for the DrivenData submissions page.
python kaggle/gen_submit.py <zip_path> "<note>" smoke|normal
"""
import json, sys
from pathlib import Path
zip_path, note, kind = sys.argv[1], sys.argv[2], sys.argv[3]
assert kind in ("smoke", "normal")
js = r"""async (page) => {
  const ZIP = %s, NOTE = %s, KIND = %s;
  const ctx = page.context();
  let dd = ctx.pages().find(p => p.url().includes('drivendata.org'));
  if (!dd) dd = await ctx.newPage();
  await dd.goto('https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/submissions/', { waitUntil: 'load', timeout: 60000 });
  await dd.waitForTimeout(2000);
  const before = await dd.locator('body').innerText();
  const used = (before.match(/Submissions used[^\d]*(\d) of (\d)/) || []).slice(1, 3);
  await dd.getByRole('button', { name: 'Make new submission' }).click();
  await dd.waitForTimeout(1500);
  const dialog = dd.getByRole('dialog');
  await dialog.locator('input[type=file]').first().setInputFiles(ZIP);
  await dialog.getByRole('textbox', { name: 'Private note (optional)' }).fill(NOTE);
  await dialog.getByRole('button', { name: KIND === 'smoke' ? 'Smoke test' : 'Normal submission' }).click();
  await dd.waitForTimeout(800);
  const sel = await dialog.getByRole('button', { name: KIND === 'smoke' ? 'Smoke test' : 'Normal submission' }).getAttribute('class');
  if (!/active/.test(sel || '')) return { error: 'kind not selected', sel };
  await dialog.getByRole('button', { name: 'Submit', exact: true }).click();
  // a 400 MB zip uploads for 10-20 min; reloading during the upload raises a beforeunload dialog (and would abort it if
  // accepted), so wait for the upload dialog to disappear on its own before touching the page
  dd.on('dialog', async d => { try { await d.dismiss(); } catch (e) {} });
  for (let k = 0; k < 240; k++) {
    const t = await dd.locator('body').innerText().catch(() => '');
    if (!/Upload in progress/.test(t) && !(await dd.locator('[role=dialog]').count())) break;
    await dd.waitForTimeout(10000);
  }
  await dd.waitForLoadState('load', { timeout: 300000 }).catch(() => {});
  await dd.waitForTimeout(4000);
  for (let k = 0; k < 20; k++) {
    const body = await dd.locator('body').innerText();
    const m = body.match(/(Pending|Running|Completed|Failed)[\s\S]{0,120}?id-(\d+)[\s\S]{0,200}?/g) || [];
    const mine = m.filter(x => x.includes(NOTE.slice(0, 25)));
    if (mine.length) return { usedBefore: used, kind: KIND, row: mine[0].replace(/\s+/g, ' ').slice(0, 300) };
    await dd.waitForTimeout(6000); await dd.reload({ waitUntil: 'load' }).catch(() => {});
  }
  return { usedBefore: used, kind: KIND, error: 'row not found yet' };
}
""" % (json.dumps(zip_path), json.dumps(note), json.dumps(kind))
Path(".playwright-mcp/dd_submit.js").write_text(js, encoding="utf-8"); print("dd_submit.js written for", kind)
