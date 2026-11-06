"""Draws the How it works diagrams, as images for the README and as a single page.

    python scripts/build_diagrams.py
    python scripts/build_diagrams.py --page how-it-works.html

Images go to docs/diagrams, in a light and a dark version. They need Chrome and the
playwright package, and network access for the fonts. GitHub strips a README's styles
and most of its SVG, so the README shows images rather than the drawings themselves.
"""
import argparse
from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "diagrams"
FONTS = "https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&display=swap"


class Fig:
    def __init__(self, name, width, height, claim):
        self.name, self.width, self.height, self.claim = name, width, height, claim
        self.parts = []

    def add(self, text):
        self.parts.append(text)

    def text(self, x, y, value, cls="lbl", anchor="middle"):
        self.add(f'<text x="{x}" y="{y}" text-anchor="{anchor}" class="{cls}">{escape(value)}</text>')

    def box(self, x, y, w, h, title=None, lines=(), kind="", align="middle"):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" class="box {kind}"/>')
        rows = ([title] if title else []) + list(lines)
        block = (18 if title else 0) + 15 * len(lines)
        top = y + (h - block) / 2 + 12.5
        tx = x + w / 2 if align == "middle" else x + 12
        if title:
            self.text(tx, top, title, "ttl", align)
            top += 18
        for line in lines:
            self.text(tx, top, line, "sub", align)
            top += 15
        return rows

    def diamond(self, cx, cy, w, h, lines):
        points = f"{cx},{cy - h / 2} {cx + w / 2},{cy} {cx},{cy + h / 2} {cx - w / 2},{cy}"
        self.add(f'<polygon points="{points}" class="box ask"/>')
        top = cy - 15 * (len(lines) - 1) / 2 + 4.5
        for line in lines:
            self.text(cx, top, line, "ttl small")
            top += 15

    def path(self, points, kind="route", start=False, end=True):
        d = "M" + " L".join(f"{x} {y}" for x, y in points)
        marks = ""
        if end:
            marks += f' marker-end="url(#{self.name}-{kind})"'
        if start:
            marks += f' marker-start="url(#{self.name}-{kind})"'
        self.add(f'<path d="{d}" class="edge {kind}"{marks}/>')

    def svg(self):
        kinds = ("route", "mesh", "sign", "ink", "good", "warn")
        markers = "".join(
            f'<marker id="{self.name}-{kind}" viewBox="0 0 10 10" refX="9" refY="5" markerUnits="userSpaceOnUse" '
            f'markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="tip-{kind}"/></marker>'
            for kind in kinds
        )
        body = "\n".join(self.parts)
        return (
            f'<svg viewBox="0 0 {self.width} {self.height}" role="img" aria-label="{escape(self.claim)}" '
            f'style="min-width:{min(self.width, 780)}px"><defs>{markers}</defs>\n{body}\n</svg>'
        )


# ---------------------------------------------------------------- 1. structure
f1 = Fig("f1", 960, 520, "Three nodes. Control planes gossip with each other, while a reply's data runs in a line through the workers.")
xs, ny, nw = [20, 345, 670], 100, 270
f1.box(40, 14, 170, 52, "People and apps", ["chat page, API"])
roles = [
    ("Node A", "Control plane", ["API, chat page, copy of the route"], "", "Worker", ["entry: runs the token loop"], "layers 0 to 21", "VRAM + RAM, draft model"),
    ("Node B", "Control plane, coordinating", ["plans routes, repairs them"], "lead", "Worker", ["middle stage"], "layers 22 to 43", "VRAM"),
    ("Node C", "Control plane", ["API, chat page, copy of the route"], "", "Worker", ["exit: picks the tokens"], "layers 44 to 63", "RAM + disk"),
]
for index, (x, role) in enumerate(zip(xs, roles)):
    name, control, control_lines, kind, worker, worker_lines, layers, memory = role
    f1.add(f'<rect x="{x}" y="{ny}" width="{nw}" height="404" rx="11" class="node"/>')
    f1.box(x + 20, ny + 24, 230, 70, control, control_lines, kind)
    f1.box(x + 20, ny + 164, 230, 62, worker, worker_lines)
    f1.box(x + 20, ny + 296, 230, 72, "llama.cpp stage process", [layers, memory])
    f1.path([(x + 190, ny + 164), (x + 190, ny + 94)], "mesh", start=False)
    f1.text(x + 198, ny + 133, "checks in", "lbl", "start")
    f1.path([(x + 110, ny + 226), (x + 110, ny + 296)], "route", start=True)
    f1.text(x + 118, ny + 265, "local pipe", "lbl", "start")
    f1.text(x + 20, ny + 392, name, "ttl", "start")
    f1.text(x + 250, ny + 392, "supervisor restarts all of this", "sub", "end")
f1.path([(125, 66), (125, ny + 24)], "route", start=True)
f1.text(134, 92, "any node", "lbl", "start")
f1.path([(250, ny + 24), (250, 82), (730, 82), (730, ny + 24)], "mesh", start=True)
f1.text(480, 86, "gossip reaches every node", "lbl halo")
f1.path([(xs[0] + 80, ny + 94), (xs[0] + 80, ny + 164)], "route", start=True)
f1.text(xs[0] + 72, ny + 133, "tokens", "lbl", "end")
for left, right in ((xs[0], xs[1]), (xs[1], xs[2])):
    a, b = left + 250, right + 20
    mid = (a + b) / 2
    f1.path([(a, ny + 59), (b, ny + 59)], "mesh", start=True)
    f1.text(mid, ny + 51, "every 3 s", "lbl halo")
    f1.path([(a, ny + 185), (b, ny + 185)], "route")
    f1.text(mid, ny + 177, "hidden states", "lbl halo")
    f1.path([(b, ny + 205), (a, ny + 205)], "route")
    f1.text(mid, ny + 220, "tokens", "lbl halo")

# ---------------------------------------------------------------- 2. one reply
f2 = Fig("f2", 960, 520, "A reply goes from any control plane to the entry worker, then loops through the stages. The coordinator is not on this path.")
cols = {"client": 84, "control": 280, "entry": 480, "middle": 670, "exit": 860}
heads = {"client": "You", "control": "Control plane", "entry": "Entry worker", "middle": "Middle stage", "exit": "Exit stage"}
for key, x in cols.items():
    f2.add(f'<line x1="{x}" y1="56" x2="{x}" y2="506" class="life"/>')
    f2.box(x - 72, 12, 144, 44, heads[key], (), "head")
f2.path([(cols["client"], 88), (cols["control"], 88)], "route")
f2.text(182, 80, "message + key", "lbl halo")
f2.box(cols["control"] - 92, 104, 184, 36, None, ["checks key, makes tokens"], "note")
f2.path([(cols["control"], 164), (cols["entry"], 164)], "route")
f2.text(380, 156, "prompt tokens", "lbl halo")
f2.add('<rect x="24" y="186" width="912" height="316" rx="9" class="frame"/>')
f2.text(38, 204, "repeats until a stop token or the length limit", "sub halo", "start")
f2.box(cols["entry"] - 88, 214, 176, 36, None, ["draft guesses up to 6 tokens"], "note")
f2.path([(cols["entry"], 276), (cols["middle"], 276)], "route")
f2.text(575, 268, "hidden states", "lbl halo")
f2.path([(cols["middle"], 312), (cols["exit"], 312)], "route")
f2.text(765, 304, "hidden states", "lbl halo")
f2.box(cols["exit"] - 66, 328, 132, 50, None, ["checks every guess", "in one pass"], "note")
f2.path([(cols["exit"], 402), (cols["middle"], 402)], "route")
f2.text(765, 394, "accepted tokens", "lbl halo")
f2.path([(cols["middle"], 402), (cols["entry"], 402)], "route")
f2.text(575, 394, "accepted tokens", "lbl halo")
f2.path([(cols["entry"], 442), (cols["control"], 442)], "route")
f2.text(380, 434, "token ids, as they come", "lbl halo")
f2.path([(cols["control"], 482), (cols["client"], 482)], "route")
f2.text(182, 474, "text, as it comes", "lbl halo")

# ---------------------------------------------------------------- 3. coordinating
f3 = Fig("f3", 890, 214, "Every node runs the same code. Coordinating is a role that moves between them.")
f3.box(20, 92, 170, 76, "Joining", ["listens for 7 s"])
f3.box(330, 92, 200, 76, "Following", ["copies the route,", "serves replies itself"])
f3.box(670, 92, 200, 76, "Coordinating", ["plans and repairs routes,", "shares its state"], "lead")
f3.path([(190, 130), (330, 130)], "mesh")
f3.text(260, 122, "hears a leader", "lbl halo")
f3.path([(530, 112), (670, 112)], "mesh")
f3.text(600, 86, "12 s of silence, and", "lbl halo")
f3.text(600, 102, "lowest name alive", "lbl halo")
f3.path([(670, 148), (530, 148)], "mesh")
f3.text(600, 164, "hears a higher term", "lbl halo")
f3.path([(105, 92), (105, 34), (770, 34), (770, 92)], "mesh")
f3.text(438, 38, "nobody leads after 7 s, and lowest name alive", "lbl halo")

# ---------------------------------------------------------------- 4. loading
f4 = Fig("f4", 960, 790, "A model loads in the background while the old layout keeps serving, and weights move only when they must.")
spine_x, spine_w, cx = 30, 280, 170
steps = [
    (14, "An admin asks any node to load", ["the request goes to the coordinator"]),
    (98, "Read the model's header", ["megabytes, from any node holding it"]),
    (182, "Plan who holds which layers", ["fastest memory first, on each node"]),
    (266, "Workers fetch their units", ["the old layout keeps serving"]),
    (350, "Let replies finish", ["entry workers drain, up to 45 s"]),
    (434, "Load the stages", ["unchanged stages keep their weights"]),
]
for y, title, lines in steps:
    f4.box(spine_x, y, spine_w, 60, title, lines)
for y, _, _ in steps[:-1]:
    f4.path([(cx, y + 60), (cx, y + 84)], "ink")
f4.path([(cx, 494), (cx, 518)], "ink")
f4.diamond(cx, 562, 200, 88, ["Every stage", "loaded?"])
f4.path([(cx, 606), (cx, 636)], "ink")
f4.text(cx + 8, 625, "yes", "lbl", "start")
f4.box(spine_x, 636, spine_w, 60, "New epoch, new route", ["drained workers let go"], "okay")
f4.path([(cx, 696), (cx, 720)], "mesh")
f4.box(spine_x, 720, spine_w, 60, "The route spreads by gossip", ["every node can now serve replies"], "okay")
# one unit
f4.add('<rect x="350" y="98" width="590" height="288" rx="9" class="frame"/>')
f4.text(366, 120, "For each unit a worker needs", "ttl", "start")
f4.path([(310, 296), (350, 296)], "ink")
for x, lines in ((430, ["Already", "cached?"]), (570, ["File on this", "machine?"]), (710, ["Does a peer", "have it?"])):
    f4.diamond(x, 178, 112, 84, lines)
f4.path([(486, 178), (514, 178)], "ink")
f4.path([(626, 178), (654, 178)], "ink")
f4.path([(766, 178), (790, 178)], "ink")
f4.text(500, 170, "no", "lbl halo")
f4.text(640, 170, "no", "lbl halo")
f4.text(778, 170, "no", "lbl halo")
f4.box(790, 148, 138, 60, "Read a holder", ["or the model hub"])
f4.box(370, 250, 558, 44, None, ["If you gave a hash list, check the unit against it, wherever it came from"], "note")
for x in (430, 570, 710):
    f4.path([(x, 220), (x, 250)], "ink")
    f4.text(x + 8, 240, "yes", "lbl", "start")
f4.path([(859, 208), (859, 250)], "ink")
f4.path([(510, 294), (510, 322)], "good")
f4.path([(790, 294), (790, 322)], "warn")
f4.box(370, 322, 280, 48, None, ["matches, or no list: the unit is ready"], "okay")
f4.box(668, 322, 260, 48, None, ["no match: discard, try the next source"], "bad")
# stage failed
f4.path([(270, 562), (350, 562)], "warn")
f4.text(308, 554, "no", "lbl halo")
f4.box(350, 532, 130, 60, "Retry twice", ["a live node"])
f4.path([(480, 562), (520, 562)], "warn")
f4.diamond(620, 562, 200, 96, ["Would replanning", "move under 2 GiB?"])
f4.path([(720, 562), (770, 562)], "ink")
f4.text(745, 554, "yes", "lbl halo")
f4.box(770, 532, 158, 60, "Replan around it", ["back to planning"])
f4.path([(620, 610), (620, 640)], "warn")
f4.text(628, 629, "no", "lbl", "start")
f4.box(500, 640, 240, 56, "Stop and say why", ["the old layout keeps serving"], "bad")

# ---------------------------------------------------------------- 5. failures
f5 = Fig("f5", 960, 610, "Seven kinds of failure keep the reply going. Only losing the node you are connected to restarts it.")
columns = [(16, 214), (256, 204), (486, 244), (756, 188)]
for (x, w), label in zip(columns, ("What happens", "Who notices", "What the pool does", "A reply in progress")):
    f5.text(x + w / 2, 16, label, "ttl")
rows = [
    ("Planned cordon", ["An admin asks"], ["Replies finish, layers move,", "drained workers let go"], ["Finishes. Under load,", "16 to 18 s quiet"], "okay"),
    ("A stage process crashes", ["Its worker: the pipe breaks"], ["Starts a new process in place,", "asks for a replay"], ["Pauses, replays,", "same text"], "okay"),
    ("A worker or node restarts", ["The node's supervisor"], ["Restarts it, and the coordinator", "gives it its stage back"], ["Pauses, replays.", "Back in 25 s"], "okay"),
    ("A node disappears", ["Coordinator: no check-in", "for 15 s, then 30 s grace"], ["Replans on the survivors"], ["Waits for the move.", "22 s if layers are there"], "okay"),
    ("The coordinator dies", ["Peers: 12 s of silence"], ["Lowest name left takes over", "and adopts the running route"], ["Unaffected.", "0.22 s longest pause"], "okay"),
    ("A link goes silent", ["The hop before it: 30 s,", "plus 1 s per token"], ["Blames the right stage,", "repairs the route"], ["Replays, same text.", "21 to 37 s pause"], "okay"),
    ("The network splits", ["Both sides"], ["Each side elects and serves.", "On healing, the higher term wins"], ["Finishes on its side.", "One leader in 0.7 s"], "okay"),
    ("Your own node dies", ["Your browser or app"], ["Nothing can hold your connection.", "Chat asks, then uses another node"], ["Restarts from", "the beginning"], "bad"),
]
top = 30
for event, notice, action, reply, kind in rows:
    (x0, w0), (x1, w1), (x2, w2), (x3, w3) = columns
    f5.box(x0, top, w0, 58, event, ())
    f5.box(x1, top, w1, 58, None, notice)
    f5.box(x2, top, w2, 58, None, action)
    f5.box(x3, top, w3, 58, None, reply, kind)
    middle = top + 29
    f5.path([(x0 + w0, middle), (x1, middle)], "ink")
    f5.path([(x1 + w1, middle), (x2, middle)], "ink")
    f5.path([(x2 + w2, middle), (x3, middle)], "good" if kind == "okay" else "warn")
    top += 72

# ---------------------------------------------------------------- 6. joining
f6 = Fig("f6", 960, 500, "A node asks through any member, the owner signs the answer offline, and neither private key ever crosses the network.")
lanes = {"new": 110, "member": 370, "others": 610, "owner": 850}
names = {"new": "New node", "member": "Any member", "others": "Other members", "owner": "Owner's machine"}
for key, x in lanes.items():
    f6.add(f'<line x1="{x}" y1="56" x2="{x}" y2="488" class="life"/>')
    f6.box(x - 78, 12, 156, 44, names[key], (), "head")
f6.box(lanes["new"] - 90, 72, 180, 36, None, ["makes its own key"], "note")
f6.path([(lanes["new"], 134), (lanes["member"], 134)], "route")
f6.text(240, 126, "request to join", "lbl halo")
f6.path([(lanes["member"], 164), (lanes["others"], 164)], "mesh")
f6.text(490, 156, "gossip", "lbl halo")
f6.path([(lanes["owner"], 200), (lanes["member"], 200)], "route")
f6.text(730, 192, "lists who is waiting", "lbl halo")
f6.box(lanes["owner"] - 96, 216, 192, 50, None, ["checks the fingerprint,", "signs with its key, offline"], "sign-note")
f6.path([(lanes["owner"], 292), (lanes["member"], 292)], "sign")
f6.text(730, 284, "signed approval + certificate", "lbl halo")
f6.path([(lanes["member"], 328), (lanes["others"], 328)], "sign")
f6.text(490, 320, "gossip", "lbl halo")
f6.box(lanes["others"] - 92, 342, 184, 36, None, ["each checks the signature"], "note")
f6.path([(lanes["member"], 402), (lanes["new"], 402)], "sign")
f6.text(240, 394, "collects its certificate", "lbl halo")
f6.path([(lanes["new"], 446), (lanes["others"], 446)], "route")
f6.text(240, 438, "every request carries a ticket", "lbl halo")
f6.text(490, 438, "good for one node, 2 minutes", "lbl halo")

# ---------------------------------------------------------------- 7. owned or ownerless
f7 = Fig("f7", 960, 356, "The difference between the two kinds of network is the owner and its signing arrows, which disappear in the ownerless version.")
for x0, title, tag, kind in ((16, "Owned", "built", "tag-built"), (492, "Ownerless", "built", "tag-built")):
    f7.add(f'<rect x="{x0}" y="10" width="452" height="336" rx="11" class="node"/>')
    f7.text(x0 + 16, 34, title, "ttl big", "start")
    f7.text(x0 + 436, 34, tag, f"ttl {kind}", "end")
    a, b, c = (x0 + 22, 240), (x0 + 171, 284), (x0 + 320, 240)
    for x, y in (a, b, c):
        f7.box(x, y, 110, 44, "Node", ())
    f7.path([(a[0] + 110, a[1] + 32), (b[0], b[1] + 12)], "mesh", start=True)
    f7.path([(b[0] + 110, b[1] + 12), (c[0], c[1] + 32)], "mesh", start=True)
    f7.path([(a[0] + 110, a[1] + 8), (c[0], c[1] + 8)], "mesh", start=True)
    f7.box(x0 + 320, 56, 110, 44, "New node", (), "note")
ox = 16 + 171
f7.box(ox - 6, 70, 122, 52, "Owner", ["key stays offline"], "sign-box")
f7.path([(ox + 20, 122), (16 + 77, 240)], "sign")
f7.path([(ox + 55, 122), (ox + 55, 284)], "sign")
f7.path([(ox + 90, 122), (16 + 375, 240)], "sign")
f7.text(ox + 55, 190, "signs", "lbl halo")
f7.path([(16 + 395, 100), (16 + 395, 240)], "route")
f7.text(16 + 404, 150, "asks", "lbl", "start")
px = 492 + 171
f7.add(f'<rect x="{px - 6}" y="70" width="122" height="52" rx="7" class="ghost"/>')
f7.text(px + 55, 101, "no owner", "sub")
f7.path([(492 + 395, 100), (492 + 395, 240)], "route")
f7.text(492 + 386, 150, "joins, or is", "lbl", "end")
f7.text(492 + 386, 166, "vouched for", "lbl", "end")

STYLE = """
/* A signal schematic: one plate per mechanism, three line colors that mean the same thing everywhere. */
:root {
  --paper: #eef2f5; --plate: #fbfcfd; --ink: #14222e; --muted: #526471; --edge: #c3cfd8; --node: #f1f5f8;
  --route: #08738a; --mesh: #6b45c9; --sign: #b32a68; --good: #22794a; --warn: #b4530a; --lead: #ece7fb; --okay: #e4f3ea; --bad: #fbeadb; --mark: #fbe4ee;
  --display: "Bricolage Grotesque", "Segoe UI", system-ui, sans-serif;
  --body: "Atkinson Hyperlegible", "Segoe UI", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --paper: #0f171e; --plate: #15202a; --ink: #e5edf3; --muted: #9db0bd; --edge: #36475a; --node: #1b2935;
  --route: #4cc3d9; --mesh: #b49cff; --sign: #f27dae; --good: #6fd29b; --warn: #f5a35c; --lead: #2c2550; --okay: #17382a; --bad: #3f2a17; --mark: #43203a; color-scheme: dark;
} }
:root[data-theme="dark"] {
  --paper: #0f171e; --plate: #15202a; --ink: #e5edf3; --muted: #9db0bd; --edge: #36475a; --node: #1b2935;
  --route: #4cc3d9; --mesh: #b49cff; --sign: #f27dae; --good: #6fd29b; --warn: #f5a35c; --lead: #2c2550; --okay: #17382a; --bad: #3f2a17; --mark: #43203a; color-scheme: dark;
}
body { background: var(--paper); color: var(--ink); font-family: var(--body); font-size: 17px; line-height: 1.55; }
.page { max-width: 1040px; margin: 0 auto; padding-block: 44px 72px; padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); display: grid; gap: 52px; }
header { display: grid; gap: 16px; }
h1 { font-family: var(--display); font-weight: 700; font-size: clamp(2.3rem, 6vw, 3.6rem); line-height: 1.04; letter-spacing: -0.02em; margin: 0; text-wrap: balance; }
h2 { font-family: var(--display); font-weight: 600; font-size: 1.55rem; line-height: 1.2; margin: 0; text-wrap: balance; }
p { margin: 0; max-width: 66ch; }
.lead { font-size: 1.15rem; }
.status { color: var(--muted); }
.key { display: flex; flex-wrap: wrap; gap: 10px 28px; align-items: center; padding-block: 4px; }
.key span { display: inline-flex; align-items: center; gap: 10px; font-size: 0.95rem; }
.key i { display: block; width: 42px; height: 0; border-top-width: 3px; border-top-style: solid; }
.key .k-route { border-top-color: var(--route); }
.key .k-mesh { border-top-color: var(--mesh); border-top-style: dashed; }
.key .k-sign { border-top-color: var(--sign); }
nav ul { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 6px 20px; }
nav a { color: var(--route); text-underline-offset: 3px; }
a:focus-visible, .scroll:focus-visible { outline: 3px solid var(--route); outline-offset: 2px; }
section { display: grid; gap: 14px; min-width: 0; scroll-margin-top: 16px; }
figure { margin: 0; display: grid; gap: 10px; min-width: 0; }
.scroll { overflow-x: auto; background: var(--plate); border: 1px solid var(--edge); border-radius: 12px; padding: 12px; }
svg { display: block; width: 100%; height: auto; max-width: none; font-family: var(--body); }
figcaption { color: var(--muted); font-size: 0.97rem; max-width: 70ch; }
table { border-collapse: collapse; width: 100%; font-size: 0.97rem; }
.table { overflow-x: auto; }
th, td { text-align: left; padding: 9px 14px 9px 0; border-bottom: 1px solid var(--edge); vertical-align: top; min-width: 120px; }
th { font-family: var(--display); font-weight: 600; }
.box { fill: var(--plate); stroke: var(--edge); stroke-width: 1.4; }
.node { fill: var(--node); stroke: var(--edge); stroke-width: 1.2; }
.box.lead { fill: var(--lead); stroke: var(--mesh); }
.box.head { fill: var(--node); }
.box.note { fill: var(--node); stroke-dasharray: none; }
.box.ask { fill: var(--node); stroke: var(--muted); }
.box.okay { fill: var(--okay); stroke: var(--good); }
.box.bad { fill: var(--bad); stroke: var(--warn); }
.box.sign-note, .box.sign-box { fill: var(--mark); stroke: var(--sign); }
.ghost { fill: none; stroke: var(--muted); stroke-width: 1.2; stroke-dasharray: 5 5; }
.frame { fill: none; stroke: var(--edge); stroke-width: 1.4; stroke-dasharray: 2 5; stroke-linecap: round; }
.life { stroke: var(--edge); stroke-width: 1.2; }
.edge { fill: none; stroke-width: 2.2; stroke-linejoin: round; }
.edge.route { stroke: var(--route); }
.edge.mesh { stroke: var(--mesh); stroke-dasharray: 7 5; }
.edge.sign { stroke: var(--sign); stroke-width: 2.6; }
.edge.ink { stroke: var(--muted); stroke-width: 1.8; }
.edge.good { stroke: var(--good); stroke-width: 1.8; }
.edge.warn { stroke: var(--warn); stroke-width: 1.8; }
.tip-route { fill: var(--route); } .tip-mesh { fill: var(--mesh); } .tip-sign { fill: var(--sign); }
.tip-ink { fill: var(--muted); } .tip-good { fill: var(--good); } .tip-warn { fill: var(--warn); }
text { fill: var(--ink); }
.ttl { font-size: 13.5px; font-weight: 700; }
.ttl.small { font-size: 12.5px; }
.ttl.big { font-family: var(--display); font-size: 17px; font-weight: 600; }
.ttl.tag-built { fill: var(--good); }
.ttl.tag-plan { fill: var(--warn); }
.sub { font-size: 12.5px; fill: var(--muted); }
.box.okay + text, .box.bad + text { fill: var(--ink); }
.lbl { font-size: 12.5px; }
.halo { paint-order: stroke; stroke: var(--plate); stroke-width: 5px; stroke-linejoin: round; }
"""


def figure(fig, caption):
    return (
        f'<figure><div class="scroll" tabindex="0" role="region" aria-label="{escape(fig.claim)}">{fig.svg()}</div>'
        f"<figcaption>{caption}</figcaption></figure>"
    )


PAGE = f"""<title>How ShardMesh Works</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="{FONTS}">
<style>{STYLE}</style>
<main class="page">
<header>
  <h1>How ShardMesh Works</h1>
  <p class="lead">ShardMesh runs one language model across several machines. Each machine holds some of the model's layers, and any machine can answer a request. No server is in charge.</p>
  <p class="status">These diagrams show what is built. Measured figures come from the test pool, which is two machines so far.</p>
  <div class="key" aria-label="Line colors">
    <span><i class="k-route"></i>A reply's data</span>
    <span><i class="k-mesh"></i>Control messages</span>
    <span><i class="k-sign"></i>Signed by the owner</span>
  </div>
  <nav aria-label="Diagrams"><ul>
    <li><a href="#structure">Structure</a></li>
    <li><a href="#reply">One reply</a></li>
    <li><a href="#coordinating">Coordinating</a></li>
    <li><a href="#loading">Loading a model</a></li>
    <li><a href="#failures">Failures</a></li>
    <li><a href="#joining">Joining</a></li>
    <li><a href="#ownership">Ownership</a></li>
  </ul></nav>
</header>

<section id="structure">
  <h2>1. Two networks in one</h2>
  <p>Every node runs the same software. Control messages spread between all nodes, while a reply's data runs in a line through the stages.</p>
  {figure(f1, "Layers are split by what each machine can hold, so one model can span a GPU on one node and RAM and disk on another. On the test pool, a laptop RTX 3060 holds layers 0 to 7 and a desktop RX 7900 XT holds layers 8 to 63, over Tailscale. Workers check in with every control plane, so any of them can take over.")}
</section>

<section id="reply">
  <h2>2. One reply, from message to text</h2>
  <p>The node you ask runs your reply itself. The coordinator isn't involved, so losing it doesn't interrupt anything.</p>
  {figure(f2, "Only hidden states and tokens cross the network during a reply, never weights. A small draft model guesses ahead so one pass can produce several tokens. Conversations sharing a stage also share a pass when they can, which took four conversations from 12 to 32 tok/s on one stage.")}
</section>

<section id="coordinating">
  <h2>3. Any node can coordinate</h2>
  <p>Nodes agree on a coordinator without voting: they all apply the same rule to the same member list.</p>
  {figure(f3, "The term only goes up, and the higher term wins, so two coordinators from a split settle themselves when they reconnect. Workers obey only the highest term they have seen.")}
</section>

<section id="loading">
  <h2>4. Loading a model</h2>
  <p>Replies keep running through the old layout until the new one is ready, and only then do they switch.</p>
  {figure(f4, "A closed network works the same way: a GGUF file on one node is enough, and the others read the parts they need from it. The 2 GiB limit applies to a node that is alive but failed to load. If a node has died, the pool replans whatever that moves.")}
</section>

<section id="failures">
  <h2>5. What happens when something fails</h2>
  <p>The rule is that a failure must not drop a reply, and weights move only when they must.</p>
  {figure(f5, "Times are measured on the test pool, except the stage crash, measured on a small model. In the last soak test, 405 replies finished and none were lost while processes were killed on both machines for 21 minutes.")}
</section>

<section id="joining">
  <h2>6. Joining an owned network</h2>
  <p>A node makes its own key and asks to join through any member. The owner approves it from their own machine.</p>
  {figure(f6, "Asking grants nothing. The new node's private key never leaves it, and the owner's key never touches the network, so a compromised node can't add a key or revoke a peer. Tickets prove a node's name without TLS, so antivirus that intercepts TLS can't break them. The link still needs Tailscale or another private network, because tickets don't encrypt anything.")}
</section>

<section id="ownership">
  <h2>7. Owned or ownerless</h2>
  <p>An owned network runs without a central server, but one person decides who belongs. Removing the owner removes the signing arrows, and everything they did has to be done another way.</p>
  {figure(f7, "Both kinds work for private or public networks.")}
  <div class="table"><table>
    <thead><tr><th>Question</th><th>Owned</th><th>Ownerless</th></tr></thead>
    <tbody>
      <tr><td>Who lets a node in?</td><td>The owner approves it</td><td>Each node decides: anyone, or nodes vouched for by those it already accepts</td></tr>
      <tr><td>Who removes a node?</td><td>The owner, for everyone</td><td>Nobody can. Each node blocks whoever it chooses, for itself</td></tr>
      <tr><td>Who picks the models?</td><td>The owner, or anyone with an admin key</td><td>Anyone can propose one beside what runs. Each node chooses what it hosts, and users pick from what's running</td></tr>
      <tr><td>What is still unsolved?</td><td>A member can read prompts and return wrong answers</td><td>The same, plus one person could run thousands of nodes</td></tr>
    </tbody>
  </table></div>
</section>
</main>
"""
FIGURES = {"structure": f1, "reply": f2, "coordinating": f3, "loading": f4, "failures": f5, "joining": f6, "ownership": f7}


def render() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        for theme in ("light", "dark"):
            for name, fig in FIGURES.items():
                tab = browser.new_page(
                    viewport={"width": fig.width + 24, "height": fig.height + 24}, device_scale_factor=2, color_scheme=theme
                )
                tab.set_content(
                    f'<!doctype html><html data-theme="{theme}"><head><meta charset="utf-8"><link rel="stylesheet" href="{FONTS}">'
                    f"<style>{STYLE} body {{ margin: 0; background: var(--plate); }} #plate {{ padding: 12px; }}</style></head>"
                    f'<body><div id="plate">{fig.svg()}</div></body></html>',
                    wait_until="networkidle",
                )
                loaded = tab.evaluate("""async () => { await document.fonts.ready; return document.fonts.check('13px "Atkinson Hyperlegible"'); }""")
                if not loaded:
                    raise SystemExit("the fonts did not load, so the images would not match")
                tab.locator("#plate").screenshot(path=str(OUT / f"{name}-{theme}.png"))
                tab.close()
        browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--page", help="write the single page here, instead of the images")
    args = parser.parse_args()
    if args.page:
        Path(args.page).write_text(PAGE, encoding="utf-8")
    else:
        render()


if __name__ == "__main__":
    main()
