# ruff: noqa: E501
"""The Community Edition local dashboard, as a single self-contained HTML page.

No build step, no framework, no external requests. Served by `grip-community
dashboard` (and at `/` by `metrics`). Long lines below are HTML/CSS/JS, hence the
file-level E501 exemption; the Python that serves it (cli.py) stays lint-strict.

Honest by construction: it renders the transparent single-device example detector,
never peer-relative reliability intelligence (that is the commercial product).
"""

DASHBOARD_HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>GRIP Community</title>
<style>
:root{color-scheme:light dark;--bg:#f6f7f9;--card:#fff;--ink:#101720;--mut:#68757f;--line:#dce1e6;
--ok:#146c3a;--notice:#0b6e99;--warn:#9a5008;--crit:#a4291c;--okbg:#e2f2e8;--noticebg:#e3eef4;--warnbg:#fbeedd;--critbg:#fbe9e7}
@media(prefers-color-scheme:dark){:root{--bg:#0d1218;--card:#141b23;--ink:#e8edf2;--mut:#8b98a4;--line:#26303b;
--ok:#69c493;--notice:#54b3dc;--warn:#e0a35c;--crit:#e79087;--okbg:#102c1d;--noticebg:#13303d;--warnbg:#33260f;--critbg:#331714}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:60rem;margin:0 auto;padding:24px 16px}
h1{font-size:1.2rem;margin:0;letter-spacing:.14em;font-weight:700}
.sub{color:var(--mut);font-size:.8rem;margin-top:2px}
.bar{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;border-bottom:2px solid var(--ink);padding-bottom:10px}
.demo{background:var(--warnbg);color:var(--warn);font-size:.7rem;padding:2px 7px;border-radius:3px;font-weight:600}
table{width:100%;border-collapse:collapse;margin-top:16px;font-variant-numeric:tabular-nums}
th{text-align:left;font-size:.68rem;text-transform:uppercase;letter-spacing:.07em;color:var(--mut);
font-weight:500;padding:6px 10px 6px 0;border-bottom:1px solid var(--line)}
td{padding:9px 10px 9px 0;border-bottom:1px solid var(--line)}
.num{font-family:ui-monospace,Menlo,Consolas,monospace}
.chip{display:inline-block;font-size:.72rem;font-weight:600;padding:2px 8px;border-radius:3px}
.ok{background:var(--okbg);color:var(--ok)}.notice{background:var(--noticebg);color:var(--notice)}
.warn{background:var(--warnbg);color:var(--warn)}.critical{background:var(--critbg);color:var(--crit)}
.note{color:var(--mut);font-size:.8rem}
.foot{margin-top:22px;padding:14px 16px;background:var(--card);border:1px solid var(--line);
border-left:3px solid var(--notice);border-radius:3px;color:var(--mut);font-size:.82rem}
.foot b{color:var(--ink)}
</style></head><body><div class=wrap>
<div class=bar><h1>GRIP</h1><span class=sub id=sub>connecting…</span><span class=demo id=demo hidden>DEMO — synthetic</span></div>
<table><thead><tr><th>GPU</th><th>Temp</th><th>Power</th><th>Util</th><th>SM clock</th><th>Mem</th><th>Status</th></tr></thead>
<tbody id=rows></tbody></table>
<div class=foot><b>Community Edition.</b> Single host, and the status column is the transparent
single-device example detector — over-temperature, hardware slowdown, uncorrectable faults. It is
<b>not</b> peer-relative reliability detection (a GPU quietly slow versus comparable peers); that is
the commercial GRIP product. Metrics also available at <span class=num>/metrics</span> for Prometheus.</div>
</div><script>
const bands={ok:'ok',notice:'notice',warn:'warn',critical:'critical'};
async function tick(){
 try{const r=await fetch('/api/telemetry');const d=await r.json();
  document.getElementById('demo').hidden=!d.demo;
  document.getElementById('sub').textContent='node '+d.node+' · updated '+new Date(d.generated).toLocaleTimeString();
  document.getElementById('rows').innerHTML=d.gpus.map(g=>{
   const cls=bands[g.status]||'ok';const note=g.findings.length?' <span class=note>'+g.findings.join('; ')+'</span>':'';
   return `<tr><td class=num>${g.index}</td><td class=num>${g.temp}°C</td>
   <td class=num>${g.power}/${g.limit} W</td><td class=num>${g.util}%</td><td class=num>${g.clock}</td>
   <td class=num>${(g.mem/1024).toFixed(1)} GB</td><td><span class="chip ${cls}">${g.status}</span>${note}</td></tr>`;
  }).join('');
 }catch(e){document.getElementById('sub').textContent='waiting for telemetry…';}
}
tick();setInterval(tick,5000);
</script></body></html>"""
