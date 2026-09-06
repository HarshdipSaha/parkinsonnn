"""Emit .playwright-mcp/push_update.js: update notebook cells by prefix from the generated cell set, run the updated
%%writefile cells, run a cleanup command, then relaunch run-all-below from the runner cell.
python kaggle/push_update.py "rm -f artifacts_v3/r2d_efficientnet_b0_f*_s0.* artifacts_v3/oof_r2d_efficientnet_b0_f*_s0.csv"
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.argv = sys.argv[:1] + sys.argv[1:]
cleanup = sys.argv[1] if len(sys.argv) > 1 else "true"
ns = {"__file__": str(ROOT / "kaggle" / "gen_run3_inject.py"), "__name__": "gen"}
exec(compile((ROOT / "kaggle" / "gen_run3_inject.py").read_text(encoding="utf-8"), "gen", "exec"), ns)
cells = ns["cells"]
updates = [c["source"] for c in cells if c["source"].startswith(("%%writefile datsc/train_v3.py", "%%writefile scripts/train_v3.py", "# === RESUMABLE"))]
js = """async (page) => {
  const updates = %s; const cleanup = %s;
  const frame = page.frames().find(f => f.url().includes('jupyterlab'));
  const idx = await frame.evaluate(({updates, cleanup}) => {
    const nb = window.jupyterapp.shell.currentWidget.content; const model = nb.model; const sm = model.sharedModel;
    const NL = String.fromCharCode(10); const prefixOf = (s) => s.split(NL)[0];
    const out = {writefile: [], runner: -1, cleanup: -1};
    for (const src of updates) {
      const pre = prefixOf(src);
      for (let i = 0; i < model.cells.length; i++) {
        if (prefixOf(model.cells.get(i).sharedModel.getSource()) === pre) {
          model.cells.get(i).sharedModel.setSource(src);
          if (src.startsWith('%%%%writefile')) out.writefile.push(i); else out.runner = i;
        }
      }
    }
    let ci = -1; for (let i = 0; i < model.cells.length; i++) if (model.cells.get(i).sharedModel.getSource().startsWith('# cleanup:')) ci = i;
    const csrc = '# cleanup: ' + new Date().toISOString() + NL + '!' + cleanup + NL + '!ls artifacts_v3 | wc -l; nvidia-smi --query-gpu=memory.used --format=csv,noheader';
    if (ci < 0) { sm.insertCell(out.runner, {cell_type: 'code', source: csrc}); ci = out.runner; out.runner += 1; } else { model.cells.get(ci).sharedModel.setSource(csrc); }
    out.cleanup = ci;
    return out;
  }, {updates, cleanup});
  const run = async (i) => { await frame.evaluate((i) => { const nb = window.jupyterapp.shell.currentWidget.content; nb.activeCellIndex = i; nb.deselectAll(); nb.select(nb.widgets[i]); window.jupyterapp.commands.execute('notebook:run-cell'); }, i); await page.waitForTimeout(5000); };
  for (const i of idx.writefile) await run(i);
  await run(idx.cleanup); await page.waitForTimeout(10000);
  await frame.evaluate((i) => { const nb = window.jupyterapp.shell.currentWidget.content; nb.activeCellIndex = i; nb.deselectAll(); nb.select(nb.widgets[i]); window.jupyterapp.commands.execute('notebook:run-all-below'); window.jupyterapp.commands.execute('docmanager:save'); }, idx.runner);
  await page.waitForTimeout(30000);
  return await frame.evaluate((idx) => {
    const model = window.jupyterapp.shell.currentWidget.content.model; const dom = document.querySelectorAll('.jp-Cell');
    const get = (i) => { const c = model.cells.get(i); const outs = c.sharedModel.getOutputs ? c.sharedModel.getOutputs() : []; let t = ''; for (const o of outs) { if (o.text) t += Array.isArray(o.text) ? o.text.join('') : String(o.text); if (o.ename) t += `[${o.ename}] ${o.evalue}`; } return { i, prompt: dom[i].querySelector('.jp-InputPrompt').textContent.trim(), tail: t.slice(-400) }; };
    return { writefile: idx.writefile.map(get), cleanup: get(idx.cleanup), runner: get(idx.runner) };
  }, idx);
}
""" % (json.dumps(updates), json.dumps(cleanup))
(ROOT / ".playwright-mcp" / "push_update.js").write_text(js, encoding="utf-8")
print("push_update.js:", len(updates), "cells; cleanup:", cleanup)
