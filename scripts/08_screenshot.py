"""Render HTML maps/pages to PNG with headless Chrome. CDN Leaflet is swapped for a local copy
(pass --leaflet-dir) so this works on machines without internet; on a normal machine just open the HTML."""
import os, re, sys, subprocess, tempfile
src, out, ldir = sys.argv[1], sys.argv[2], sys.argv[3]
w, h = (sys.argv[4] if len(sys.argv) > 4 else "1400,900").split(",")
html = open(src).read()
html = re.sub(r'https://[^"\']*leaflet(@[\d.]+)?/dist/leaflet\.js', f"file://{ldir}/leaflet.js", html)
html = re.sub(r'https://[^"\']*leaflet(@[\d.]+)?/dist/leaflet\.css', f"file://{ldir}/leaflet.css", html)
tmp = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, dir=os.path.dirname(os.path.abspath(src))); tmp.write(html); tmp.close()
subprocess.run(["google-chrome", "--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", f"--window-size={w},{h}",
                "--virtual-time-budget=8000", "--allow-file-access-from-files", f"--screenshot={os.path.abspath(out)}", f"file://{tmp.name}"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
os.unlink(tmp.name); print(out, os.path.exists(out))
