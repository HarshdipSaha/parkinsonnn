"""Emit .playwright-mcp/push_cells.js: sync the notebook with the generated cell set WITHOUT running the job runners.
- cells matched by first line: source updated in place
- missing cells: inserted at the matching position of the generated order
- %%writefile cells whose content changed (or that are new) are executed so the files on Kaggle match the repo
python kaggle/push_cells.py
"""
import json, sys
from pathlib import Path
NO_RUN = "--no-run" in sys.argv

ROOT = Path(__file__).resolve().parent.parent
ns = {"__file__": str(ROOT / "kaggle" / "gen_run3_inject.py"), "__name__": "gen"}
exec(compile((ROOT / "kaggle" / "gen_run3_inject.py").read_text(encoding="utf-8"), "gen", "exec"), ns)
cells = [{"type": c["type"], "source": c["source"]} for c in ns["cells"]]
js = """async (page) => {
  const cells = %s;
  const frame = page.frames().find(f => f.url().includes('jupyterlab'));
  const plan = await frame.evaluate((cells) => {
    const nb = window.jupyterapp.shell.currentWidget.content; const model = nb.model; const sm = model.sharedModel;
    const NL = String.fromCharCode(10); const head = (s) => s.split(NL)[0];
    const find = (h) => { for (let i = 0; i < model.cells.length; i++) if (head(model.cells.get(i).sharedModel.getSource()) === h) return i; return -1; };
    const toRun = []; let inserted = 0, updated = 0, unchanged = 0;
    let cursor = 0;
    for (const c of cells) {
      const h = head(c.source); let i = find(h);
      if (i < 0) {
        sm.insertCell(cursor, {cell_type: c.type, source: c.source}); i = cursor; inserted += 1;
        if (c.source.startsWith('%%%%writefile')) toRun.push(i);
      } else {
        const cur = model.cells.get(i).sharedModel.getSource();
        if (cur !== c.source) { model.cells.get(i).sharedModel.setSource(c.source); updated += 1; if (c.source.startsWith('%%%%writefile')) toRun.push(i); } else unchanged += 1;
      }
      cursor = i + 1;
    }
    return { toRun, inserted, updated, unchanged, n: model.cells.length };
  }, cells);
  for (const i of (%s ? [] : plan.toRun)) {
    await frame.evaluate((i) => { const nb = window.jupyterapp.shell.currentWidget.content; nb.activeCellIndex = i; nb.deselectAll(); nb.select(nb.widgets[i]); window.jupyterapp.commands.execute('notebook:run-cell'); }, i);
    await page.waitForTimeout(4000);
  }
  await frame.evaluate(() => window.jupyterapp.commands.execute('docmanager:save'));
  await page.waitForTimeout(3000);
  const outs = await frame.evaluate((idx) => {
    const model = window.jupyterapp.shell.currentWidget.content.model; const dom = document.querySelectorAll('.jp-Cell');
    return idx.map(i => { const c = model.cells.get(i); const o = c.sharedModel.getOutputs ? c.sharedModel.getOutputs() : []; let t = ''; for (const x of o) { if (x.text) t += Array.isArray(x.text) ? x.text.join('') : String(x.text); if (x.ename) t += `[${x.ename}] ${x.evalue}`; } return { i, head: c.sharedModel.getSource().split(String.fromCharCode(10))[0].slice(0, 50), prompt: dom[i].querySelector('.jp-InputPrompt').textContent.trim(), tail: t.slice(-120) }; });
  }, plan.toRun);
  return { ...plan, outs };
}
""" % (json.dumps(cells), "true" if NO_RUN else "false")
(ROOT / ".playwright-mcp" / "push_cells.js").write_text(js, encoding="utf-8")
print(f"push_cells.js: {len(cells)} cells")
