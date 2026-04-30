"""
Manim visualizations for the Elliptic and Elliptic++ datasets.

Render with Manim Community Edition:
    manim -pqh elliptic_visualization.py EllipticScene
    manim -pqh elliptic_visualization.py EllipticPlusPlusScene

Facts grounded against the actual data files in this repo:
    transactions_data/txs_edgelist.csv  -> txId1, txId2  (tx -> tx, directed)
    transactions_data/txs_features.csv  -> txId, Time step, Local_feature_1..94, Aggregate_feature_1..72, ...
    transactions_data/txs_classes.csv   -> txId, class (1 illicit, 2 licit, 3 unknown)
    actors_data/AddrTx_edgelist.txt     -> input_address, txId  (wallet -> tx, "input" edge)
    actors_data/TxAddr_edgelist.txt     -> txId, output_address (tx -> wallet, "output" edge)
    actors_data/AddrAddr_edgelist.txt   -> input_address, output_address (wallet -> wallet)
    actors_data/wallets_features.txt    -> address, Time step, 55 wallet features
"""

from manim import *

# Manim's default Pango font on macOS renders body text with very loose
# tracking ("h o m o g e n e o u s"). Pin to Helvetica (a system font) so
# every Text in this file has consistent, tight kerning.
Text.set_default(font="Helvetica")


# ---------- shared helpers ----------------------------------------------------

TX_COLOR = BLUE_D
WALLET_COLOR = ORANGE
TX_TX_EDGE_COLOR = GREY_B
ADDR_TX_EDGE_COLOR = GOLD
TX_ADDR_EDGE_COLOR = TEAL
ADDR_ADDR_EDGE_COLOR = PURPLE_B
TIMESTEP_COLOR = YELLOW_E
ILLICIT = RED_D
LICIT = GREEN_D
UNKNOWN = GREY


def tx_node(point, radius=0.16, color=TX_COLOR):
    return Dot(point=point, radius=radius, color=color, stroke_width=2,
               stroke_color=WHITE)


def wallet_node(point, side=0.30, color=WALLET_COLOR):
    return Square(side_length=side, color=color, fill_color=color,
                  fill_opacity=0.85, stroke_color=WHITE, stroke_width=2).move_to(point)


def directed(a, b, color=TX_TX_EDGE_COLOR, stroke_width=3, buff=0.18):
    return Arrow(a.get_center(), b.get_center(), buff=buff,
                 stroke_width=stroke_width, color=color,
                 max_tip_length_to_length_ratio=0.12,
                 max_stroke_width_to_length_ratio=8)


# =============================================================================
# Scene 1: Elliptic
# =============================================================================

class EllipticScene(Scene):
    def construct(self):
        # ---- title ----
        title = Text("Elliptic", font_size=46, weight=BOLD)
        subtitle = Text("a homogeneous Bitcoin-transaction graph",
                        font_size=24, color=GREY_B)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.3)
        self.play(Write(title), FadeIn(subtitle, shift=0.2 * UP))
        self.wait(0.4)

        # ---- definitions panel ----
        node_legend = VGroup(
            tx_node(ORIGIN),
            Text("node = one Bitcoin transaction (a tx_id)",
                 font_size=20),
        ).arrange(RIGHT, buff=0.25)

        edge_legend = VGroup(
            Arrow(LEFT * 0.25, RIGHT * 0.25, buff=0,
                  stroke_width=3, color=TX_TX_EDGE_COLOR,
                  max_tip_length_to_length_ratio=0.35),
            Text("edge = BTC flow: output of tx_a becomes input of tx_b",
                 font_size=20),
        ).arrange(RIGHT, buff=0.25)

        defs = VGroup(node_legend, edge_legend).arrange(
            DOWN, aligned_edge=LEFT, buff=0.18
        ).next_to(header, DOWN, buff=0.35).to_edge(LEFT, buff=0.5)

        self.play(FadeIn(defs, shift=0.2 * UP))
        self.wait(0.6)

        # ---- three independent timestep components ----
        # Each cluster is a small DAG. Edges only exist WITHIN a cluster.
        ts_specs = [
            ("t = 14", LEFT * 4.6 + DOWN * 0.6),
            ("t = 15", DOWN * 0.6),
            ("t = 16", RIGHT * 4.6 + DOWN * 0.6),
        ]
        offsets = [
            UP * 0.95 + LEFT * 0.55,
            UP * 0.95 + RIGHT * 0.55,
            LEFT * 1.05,
            RIGHT * 1.05,
            DOWN * 0.55,
            DOWN * 1.40 + LEFT * 0.45,
            DOWN * 1.40 + RIGHT * 0.45,
        ]
        edge_pairs = [(0, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6)]

        clusters = []   # list of (label, box, nodes, edges)
        for label_text, center in ts_specs:
            nodes = [tx_node(center + o) for o in offsets]
            edges = [directed(nodes[i], nodes[j]) for i, j in edge_pairs]
            box = SurroundingRectangle(
                VGroup(*nodes, *edges), buff=0.30,
                color=TIMESTEP_COLOR, stroke_opacity=0.55, stroke_width=2,
            )
            label = Text(label_text, font_size=22, color=TIMESTEP_COLOR)
            label.next_to(box, UP, buff=0.08)
            clusters.append((label, box, nodes, edges))

        # build animation: draw clusters left-to-right
        for label, box, nodes, edges in clusters:
            self.play(
                Create(box),
                Write(label),
                run_time=0.5,
            )
            self.play(
                LaggedStart(*[GrowFromCenter(n) for n in nodes], lag_ratio=0.08),
                run_time=0.7,
            )
            self.play(
                LaggedStart(*[Create(e) for e in edges], lag_ratio=0.08),
                run_time=0.8,
            )
        self.wait(0.4)

        # ---- emphasize: NO edges between timestep components ----
        independence = Text(
            "Each timestep is its own connected component "
            "(no inter-timestep edges).",
            font_size=22, color=YELLOW_B,
        ).to_edge(DOWN, buff=0.3)

        # flash a "forbidden" edge between t=15 and t=16, then strike it out
        n_mid = clusters[1][2][3]   # right-side node of t=15 cluster
        n_right = clusters[2][2][2]  # left-side node of t=16 cluster
        forbidden = DashedLine(
            n_mid.get_center(), n_right.get_center(),
            color=RED, stroke_width=3, dash_length=0.12,
        )
        cross = Cross(forbidden, color=RED, stroke_width=4)
        self.play(Create(forbidden))
        self.play(Create(cross), Write(independence))
        self.wait(1.4)
        self.play(FadeOut(forbidden), FadeOut(cross))

        # ---- feature arrow: pick one node and show its 166 features ----
        focus_node = clusters[1][2][4]   # middle node of middle cluster
        focus_ring = Circle(radius=0.32, color=YELLOW, stroke_width=4).move_to(focus_node)

        feat_title = Text("features attached to this tx node",
                          font_size=20, color=WHITE, weight=BOLD)
        feat_lines = VGroup(
            Text("• tx_id  (identity)", font_size=18),
            Text("• Time step  (which of the 49 timesteps)", font_size=18),
            Text("• 94 local features  (per-tx anonymized)",
                 font_size=18, color=BLUE_B),
            Text("• 72 aggregate features  (1-hop neighbour stats)",
                 font_size=18, color=BLUE_B),
            Text("• class ∈ {illicit, licit, unknown}",
                 font_size=18, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        feat_box_inner = VGroup(feat_title, feat_lines).arrange(
            DOWN, aligned_edge=LEFT, buff=0.18
        )
        feat_panel = VGroup(
            SurroundingRectangle(feat_box_inner, buff=0.22,
                                 color=BLUE_B, stroke_opacity=0.9,
                                 fill_color=BLACK, fill_opacity=0.85),
            feat_box_inner,
        )
        feat_panel.scale(0.95).to_corner(DR, buff=0.4)

        feat_arrow = Arrow(
            focus_node.get_center() + DOWN * 0.35 + RIGHT * 0.25,
            feat_panel.get_top() + LEFT * 1.2,
            buff=0.15, stroke_width=4, color=YELLOW,
            max_tip_length_to_length_ratio=0.06,
        )

        # hide the bottom independence text temporarily so the panel reads cleanly
        self.play(FadeOut(independence))
        self.play(Create(focus_ring))
        self.play(GrowArrow(feat_arrow))
        self.play(FadeIn(feat_panel, shift=0.2 * UP))
        self.wait(2.0)

        # ---- closing summary ----
        summary = Text(
            "Elliptic = 49 disjoint per-timestep tx subgraphs, "
            "homogeneous nodes/edges.",
            font_size=22, color=YELLOW_B,
        ).to_edge(DOWN, buff=0.3)
        self.play(Write(summary))
        self.wait(2.0)


# =============================================================================
# Scene 2: Elliptic++
# =============================================================================

class EllipticPlusPlusScene(Scene):
    def construct(self):
        # ---- title ----
        title = Text("Elliptic++", font_size=46, weight=BOLD)
        subtitle = Text("a heterogeneous graph: transactions + wallets",
                        font_size=24, color=GREY_B)
        header = VGroup(title, subtitle).arrange(DOWN, buff=0.15).to_edge(UP, buff=0.3)
        self.play(Write(title), FadeIn(subtitle, shift=0.2 * UP))
        self.wait(0.4)

        # ---- legend: 2 node types + 4 edge types ----
        leg_node_tx = VGroup(
            tx_node(ORIGIN),
            Text("tx node  (one Bitcoin transaction)",
                 font_size=18),
        ).arrange(RIGHT, buff=0.20)
        leg_node_w = VGroup(
            wallet_node(ORIGIN),
            Text("wallet node  (one Bitcoin address)",
                 font_size=18),
        ).arrange(RIGHT, buff=0.20)

        def edge_swatch(color):
            return Arrow(LEFT * 0.22, RIGHT * 0.22, buff=0, stroke_width=3,
                         color=color, max_tip_length_to_length_ratio=0.35)

        leg_e1 = VGroup(edge_swatch(TX_TX_EDGE_COLOR),
                        Text("tx → tx   (BTC chain flow, same as Elliptic)",
                             font_size=18)).arrange(RIGHT, buff=0.20)
        leg_e2 = VGroup(edge_swatch(ADDR_TX_EDGE_COLOR),
                        Text("addr → tx   (wallet is an INPUT to a tx)",
                             font_size=18)).arrange(RIGHT, buff=0.20)
        leg_e3 = VGroup(edge_swatch(TX_ADDR_EDGE_COLOR),
                        Text("tx → addr   (tx pays an OUTPUT to a wallet)",
                             font_size=18)).arrange(RIGHT, buff=0.20)
        leg_e4 = VGroup(edge_swatch(ADDR_ADDR_EDGE_COLOR),
                        Text("addr → addr   (wallet ↔ wallet co-spend link)",
                             font_size=18)).arrange(RIGHT, buff=0.20)

        legend = VGroup(leg_node_tx, leg_node_w, leg_e1, leg_e2, leg_e3, leg_e4
                        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        legend_panel = VGroup(
            SurroundingRectangle(legend, buff=0.20, color=GREY_B,
                                 stroke_opacity=0.7,
                                 fill_color=BLACK, fill_opacity=0.6),
            legend,
        ).scale(0.85).to_corner(UL, buff=0.3).shift(DOWN * 0.55)

        self.play(FadeIn(legend_panel, shift=0.2 * UP))
        self.wait(0.6)

        # ---- build the graph: 3 timesteps, tx on top, wallets on bottom ----
        # tx nodes live INSIDE a timestep (they have a Time step column).
        # wallet nodes are SHARED across timesteps (an address persists in time);
        # we draw them on a single bottom row to make this explicit.
        ts_centers_x = [-3.7, 0.0, 3.7]
        tx_y = 1.05
        wallet_y = -1.85

        # tx nodes: 3 per timestep
        tx_offsets = [LEFT * 0.85, ORIGIN, RIGHT * 0.85]
        tx_by_ts = []   # list of lists
        ts_labels = []
        ts_boxes = []
        for cx, name in zip(ts_centers_x, ["t = 14", "t = 15", "t = 16"]):
            row = [tx_node(np.array([cx, tx_y, 0]) + o) for o in tx_offsets]
            tx_by_ts.append(row)
            box = SurroundingRectangle(
                VGroup(*row), buff=0.25, color=TIMESTEP_COLOR,
                stroke_opacity=0.55, stroke_width=2,
            )
            lbl = Text(name, font_size=20, color=TIMESTEP_COLOR
                       ).next_to(box, UP, buff=0.08)
            ts_boxes.append(box)
            ts_labels.append(lbl)

        # tx -> tx edges INSIDE each timestep
        tx_tx_edges = []
        for row in tx_by_ts:
            tx_tx_edges.append(directed(row[0], row[1]))
            tx_tx_edges.append(directed(row[1], row[2]))

        # wallets: shared bottom row, positioned so they bridge timesteps.
        # W0 is referenced by tx in t=14 AND t=15 -> creates cross-timestep path
        # W3 is referenced by tx in t=15 AND t=16 -> another cross-timestep path
        wallet_xs = [-4.2, -2.5, -0.9, 0.9, 2.5, 4.2]
        wallets = [wallet_node(np.array([x, wallet_y, 0])) for x in wallet_xs]

        # bipartite edges (addr<->tx). The KEY ones for cross-timestep are marked.
        # Each tuple: (wallet_idx, ts_idx, tx_idx_in_ts, direction)
        # direction = "in"  -> addr -> tx  (gold)
        # direction = "out" -> tx  -> addr (teal)
        bipartite = [
            (0, 0, 0, "in"),
            (1, 0, 1, "in"),
            (2, 0, 2, "out"),
            (0, 1, 0, "in"),    # <-- W0 reused in t=15  (cross-timestep)
            (3, 1, 1, "out"),
            (3, 2, 0, "in"),    # <-- W3 reused in t=16  (cross-timestep)
            (4, 2, 1, "in"),
            (5, 2, 2, "out"),
        ]
        bipartite_arrows = []
        cross_ts_arrows_idx = []
        for idx, (w_i, ts_i, tx_i, direction) in enumerate(bipartite):
            w = wallets[w_i]
            t = tx_by_ts[ts_i][tx_i]
            if direction == "in":
                arr = directed(w, t, color=ADDR_TX_EDGE_COLOR,
                               stroke_width=2.5, buff=0.14)
            else:
                arr = directed(t, w, color=TX_ADDR_EDGE_COLOR,
                               stroke_width=2.5, buff=0.14)
            bipartite_arrows.append(arr)

        # the four arrows incident to W0 and W3 are the cross-timestep ones
        for idx, (w_i, *_rest) in enumerate(bipartite):
            if w_i in (0, 3):
                cross_ts_arrows_idx.append(idx)

        # addr -> addr edges (a couple, to show they exist)
        aa_edges = [
            directed(wallets[1], wallets[2], color=ADDR_ADDR_EDGE_COLOR,
                     stroke_width=2.5, buff=0.16),
            directed(wallets[4], wallets[5], color=ADDR_ADDR_EDGE_COLOR,
                     stroke_width=2.5, buff=0.16),
        ]

        # ---- animate construction ----
        # 1) draw timestep boxes + tx nodes per timestep
        for box, lbl, row in zip(ts_boxes, ts_labels, tx_by_ts):
            self.play(Create(box), Write(lbl), run_time=0.4)
            self.play(LaggedStart(*[GrowFromCenter(n) for n in row],
                                  lag_ratio=0.1), run_time=0.5)
        # 2) tx -> tx edges
        self.play(LaggedStart(*[Create(e) for e in tx_tx_edges],
                              lag_ratio=0.08), run_time=0.9)

        # 3) wallets row (with a banner labelling them as shared-across-time)
        wallet_banner = Text(
            "wallet nodes are shared across all timesteps "
            "(an address persists through time)",
            font_size=20, color=ORANGE,
        ).to_edge(DOWN, buff=0.25)
        self.play(LaggedStart(*[GrowFromCenter(w) for w in wallets],
                              lag_ratio=0.1), Write(wallet_banner),
                  run_time=0.9)
        self.wait(0.3)

        # 4) bipartite edges
        self.play(LaggedStart(*[Create(a) for a in bipartite_arrows],
                              lag_ratio=0.05), run_time=1.2)
        # 5) addr -> addr edges
        self.play(LaggedStart(*[Create(a) for a in aa_edges],
                              lag_ratio=0.1), run_time=0.6)
        self.wait(0.4)

        # ---- highlight cross-timestep edges ----
        self.play(FadeOut(wallet_banner))
        cross_msg = Text(
            "cross-timestep paths arise from the bipartite Addr↔Tx structure: "
            "the same wallet appears in multiple timesteps.",
            font_size=20, color=YELLOW_B,
        ).to_edge(DOWN, buff=0.25)
        # rings around the two reused wallets
        ring_w0 = Circle(radius=0.30, color=YELLOW, stroke_width=4
                         ).move_to(wallets[0])
        ring_w3 = Circle(radius=0.30, color=YELLOW, stroke_width=4
                         ).move_to(wallets[3])
        # thicken + recolor the cross-timestep arrows
        cross_anims = [
            bipartite_arrows[i].animate.set_stroke(width=5, color=YELLOW)
            for i in cross_ts_arrows_idx
        ]
        self.play(Create(ring_w0), Create(ring_w3), Write(cross_msg))
        self.play(*cross_anims, run_time=0.8)
        self.wait(2.0)

        # restore original colors/widths so the legend stays truthful
        restore_anims = []
        for i in cross_ts_arrows_idx:
            _, _, _, direction = bipartite[i]
            target_color = ADDR_TX_EDGE_COLOR if direction == "in" else TX_ADDR_EDGE_COLOR
            restore_anims.append(
                bipartite_arrows[i].animate.set_stroke(width=2.5,
                                                       color=target_color)
            )
        self.play(*restore_anims, FadeOut(ring_w0), FadeOut(ring_w3),
                  FadeOut(cross_msg), run_time=0.5)

        # ---- feature arrows: one for tx, one for wallet ----
        # focus a tx node and a wallet node
        focus_tx = tx_by_ts[1][1]
        focus_w = wallets[3]
        focus_tx_ring = Circle(radius=0.26, color=YELLOW, stroke_width=4
                               ).move_to(focus_tx)
        focus_w_ring = Circle(radius=0.30, color=YELLOW, stroke_width=4
                              ).move_to(focus_w)

        # tx feature panel (right side, upper)
        tx_feat_inner = VGroup(
            Text("tx node features", font_size=18, weight=BOLD),
            Text("• tx_id, Time step", font_size=15),
            Text("• 94 local features", font_size=15, color=BLUE_B),
            Text("• 72 aggregate features", font_size=15, color=BLUE_B),
            Text("• class ∈ {illicit, licit, unknown}",
                 font_size=15, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        tx_feat_panel = VGroup(
            SurroundingRectangle(tx_feat_inner, buff=0.16, color=BLUE_B,
                                 stroke_opacity=0.9,
                                 fill_color=BLACK, fill_opacity=0.85),
            tx_feat_inner,
        ).scale(0.95)
        tx_feat_panel.to_corner(UR, buff=0.3).shift(DOWN * 0.4)

        # wallet feature panel (right side, lower)
        w_feat_inner = VGroup(
            Text("wallet node features", font_size=18, weight=BOLD),
            Text("• address, Time step", font_size=15),
            Text("• 55 wallet features", font_size=15, color=ORANGE),
            Text("  (lifetime, BTC sent/received,", font_size=14, color=GREY_B),
            Text("   tx counts, fees, block stats, …)",
                 font_size=14, color=GREY_B),
            Text("• class ∈ {illicit, licit, unknown}",
                 font_size=15, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.07)
        w_feat_panel = VGroup(
            SurroundingRectangle(w_feat_inner, buff=0.16, color=ORANGE,
                                 stroke_opacity=0.9,
                                 fill_color=BLACK, fill_opacity=0.85),
            w_feat_inner,
        ).scale(0.95)
        w_feat_panel.to_corner(DR, buff=0.3)

        tx_feat_arrow = Arrow(
            focus_tx.get_center() + UP * 0.25 + RIGHT * 0.15,
            tx_feat_panel.get_left() + LEFT * 0.05,
            buff=0.12, stroke_width=4, color=YELLOW,
            max_tip_length_to_length_ratio=0.05,
        )
        w_feat_arrow = Arrow(
            focus_w.get_center() + DOWN * 0.25 + RIGHT * 0.15,
            w_feat_panel.get_left() + LEFT * 0.05,
            buff=0.12, stroke_width=4, color=YELLOW,
            max_tip_length_to_length_ratio=0.05,
        )

        # fade legend out so the right side has room
        self.play(FadeOut(legend_panel))

        self.play(Create(focus_tx_ring))
        self.play(GrowArrow(tx_feat_arrow))
        self.play(FadeIn(tx_feat_panel, shift=0.2 * UP))
        self.wait(1.6)

        self.play(Create(focus_w_ring))
        self.play(GrowArrow(w_feat_arrow))
        self.play(FadeIn(w_feat_panel, shift=0.2 * UP))
        self.wait(2.0)

        # ---- closing summary ----
        summary = Text(
            "Elliptic++ = Elliptic's tx-tx graph + a wallet layer; "
            "wallets bridge timesteps via Addr↔Tx edges.",
            font_size=20, color=YELLOW_B,
        ).to_edge(DOWN, buff=0.25)
        self.play(Write(summary))
        self.wait(2.5)


# =============================================================================
# Static (single-frame) versions for slide deck use.
# Render with:  manim -sqh elliptic_visualization.py EllipticStatic
#               manim -sqh elliptic_visualization.py EllipticPlusPlusStatic
# (the -s flag tells Manim to save the last frame as a PNG instead of a video)
# =============================================================================

class EllipticStatic(Scene):
    def construct(self):
        # ---- title (top-left) ----
        title = Text("Elliptic", font_size=40, weight=BOLD)
        subtitle = Text(
            "homogeneous Bitcoin-tx graph  ·  49 timesteps  ·  ~2 weeks apart",
            font_size=18, color=GREY_B,
        )
        header = VGroup(title, subtitle).arrange(
            DOWN, buff=0.10, aligned_edge=LEFT
        ).to_corner(UL, buff=0.35)

        # ---- legend (top-right) ----
        leg_n = VGroup(
            tx_node(ORIGIN, radius=0.13),
            Text("node = 1 Bitcoin tx (a txId)", font_size=16),
        ).arrange(RIGHT, buff=0.18)
        leg_e = VGroup(
            Arrow(LEFT * 0.20, RIGHT * 0.20, buff=0, stroke_width=3,
                  color=TX_TX_EDGE_COLOR, max_tip_length_to_length_ratio=0.40),
            Text("edge = BTC flow  (output of tx_a → input of tx_b)",
                 font_size=16),
        ).arrange(RIGHT, buff=0.18)
        legend = VGroup(leg_n, leg_e).arrange(
            DOWN, aligned_edge=LEFT, buff=0.10
        )
        legend_panel = VGroup(
            SurroundingRectangle(legend, buff=0.16, color=GREY_B,
                                 fill_color=BLACK, fill_opacity=0.45,
                                 stroke_opacity=0.7, stroke_width=1.5),
            legend,
        ).to_corner(UR, buff=0.30)

        # ---- 3 timestep clusters (compressed so there's room below) ----
        ts_specs = [
            ("t = 14", LEFT * 4.4 + UP * 1.05),
            ("t = 15", UP * 1.05),
            ("t = 16", RIGHT * 4.4 + UP * 1.05),
        ]
        offsets = [
            UP * 0.65 + LEFT * 0.40,
            UP * 0.65 + RIGHT * 0.40,
            LEFT * 0.75,
            RIGHT * 0.75,
            DOWN * 0.30,
            DOWN * 0.95 + LEFT * 0.35,
            DOWN * 0.95 + RIGHT * 0.35,
        ]
        edge_pairs = [(0, 2), (1, 3), (2, 4), (3, 4), (4, 5), (4, 6)]

        cluster_nodes = []
        cluster_groups = []
        for label_text, center in ts_specs:
            nodes = [tx_node(center + o, radius=0.12) for o in offsets]
            cluster_nodes.append(nodes)
            edges = [directed(nodes[i], nodes[j]) for i, j in edge_pairs]
            box = SurroundingRectangle(
                VGroup(*nodes, *edges), buff=0.18,
                color=TIMESTEP_COLOR, stroke_opacity=0.6, stroke_width=2,
            )
            lbl = Text(label_text, font_size=18, color=TIMESTEP_COLOR
                       ).next_to(box, UP, buff=0.06)
            cluster_groups.append(VGroup(box, lbl, *nodes, *edges))

        # ---- forbidden inter-timestep edge: between t=15 and t=16, top row ----
        # use the TOP-RIGHT node of t=15 and TOP-LEFT node of t=16 so the
        # forbidden edge sits in the empty band above the lower cluster nodes.
        n_a = cluster_nodes[1][1]
        n_b = cluster_nodes[2][0]
        forbidden = DashedLine(
            n_a.get_center(), n_b.get_center(),
            color=RED, stroke_width=3, dash_length=0.12,
        )
        cross = Cross(forbidden, color=RED, stroke_width=5,
                      scale_factor=0.35)
        forbidden_lbl = Text("× no edge across timesteps",
                             font_size=15, color=RED_B
                             ).next_to(forbidden, UP, buff=0.06)

        # ---- bottom callout: independence ----
        indep_callout = Text(
            "Each timestep is its own connected component "
            "→ 49 disjoint subgraphs.",
            font_size=18, color=YELLOW_B,
        ).move_to(np.array([-2.5, -2.0, 0]))

        # ---- feature callout: arrow from a node to a feature panel ----
        focus = cluster_nodes[1][6]   # bottom-right node of t=15
        focus_ring = Circle(radius=0.22, color=YELLOW,
                            stroke_width=3).move_to(focus)

        feat_lines = VGroup(
            Text("features per tx node", font_size=16, weight=BOLD),
            Text("• txId, Time step  (identity)", font_size=13),
            Text("• 94 local features", font_size=13, color=BLUE_B),
            Text("• 72 aggregate features  (1-hop neighbour stats)",
                 font_size=13, color=BLUE_B),
            Text("• class ∈ {illicit, licit, unknown}",
                 font_size=13, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.06)
        feat_panel = VGroup(
            SurroundingRectangle(feat_lines, buff=0.16, color=BLUE_B,
                                 stroke_opacity=0.9,
                                 fill_color=BLACK, fill_opacity=0.85),
            feat_lines,
        ).to_corner(DR, buff=0.30)

        feat_arrow = Arrow(
            focus.get_center() + DOWN * 0.16 + RIGHT * 0.05,
            feat_panel.get_corner(UL) + RIGHT * 0.20 + DOWN * 0.05,
            buff=0.12, stroke_width=3, color=YELLOW,
            max_tip_length_to_length_ratio=0.05,
        )

        # ---- compose ----
        self.add(*cluster_groups)
        self.add(forbidden, cross, forbidden_lbl)
        self.add(focus_ring, feat_arrow, feat_panel)
        self.add(indep_callout)
        self.add(header, legend_panel)
        self.wait(0.05)


class EllipticPlusPlusStatic(Scene):
    def construct(self):
        # ---- title (top-left) ----
        title = Text("Elliptic++", font_size=40, weight=BOLD)
        subtitle = Text(
            "heterogeneous graph: tx + wallet nodes  ·  4 edge types",
            font_size=18, color=GREY_B,
        )
        header = VGroup(title, subtitle).arrange(
            DOWN, buff=0.10, aligned_edge=LEFT
        ).to_corner(UL, buff=0.30)

        # ---- legend (top-right) ----
        def edge_swatch(color):
            return Arrow(LEFT * 0.18, RIGHT * 0.18, buff=0, stroke_width=3,
                         color=color, max_tip_length_to_length_ratio=0.40)

        leg_ntx = VGroup(tx_node(ORIGIN, radius=0.11),
                         Text("tx node", font_size=14)
                         ).arrange(RIGHT, buff=0.15)
        leg_nw = VGroup(wallet_node(ORIGIN, side=0.22),
                        Text("wallet node", font_size=14)
                        ).arrange(RIGHT, buff=0.15)
        nodes_row = VGroup(leg_ntx, leg_nw).arrange(RIGHT, buff=0.40)

        leg_e1 = VGroup(edge_swatch(TX_TX_EDGE_COLOR),
                        Text("tx → tx   (BTC chain flow, same as Elliptic)",
                             font_size=14)).arrange(RIGHT, buff=0.15)
        leg_e2 = VGroup(edge_swatch(ADDR_TX_EDGE_COLOR),
                        Text("addr → tx   (wallet INPUT to a tx)",
                             font_size=14)).arrange(RIGHT, buff=0.15)
        leg_e3 = VGroup(edge_swatch(TX_ADDR_EDGE_COLOR),
                        Text("tx → addr   (tx OUTPUT to a wallet)",
                             font_size=14)).arrange(RIGHT, buff=0.15)
        leg_e4 = VGroup(edge_swatch(ADDR_ADDR_EDGE_COLOR),
                        Text("addr → addr   (wallet ↔ wallet co-spend)",
                             font_size=14)).arrange(RIGHT, buff=0.15)
        edges_col = VGroup(leg_e1, leg_e2, leg_e3, leg_e4).arrange(
            DOWN, aligned_edge=LEFT, buff=0.07
        )
        legend = VGroup(nodes_row, edges_col).arrange(
            DOWN, aligned_edge=LEFT, buff=0.12
        )
        legend_panel = VGroup(
            SurroundingRectangle(legend, buff=0.15, color=GREY_B,
                                 fill_color=BLACK, fill_opacity=0.55,
                                 stroke_opacity=0.7, stroke_width=1.5),
            legend,
        ).to_corner(UR, buff=0.25)

        # ---- main graph ----
        ts_centers_x = [-3.6, 0.0, 3.6]
        tx_y = 1.40
        wallet_y = -0.40
        tx_offs = [LEFT * 0.70, ORIGIN, RIGHT * 0.70]

        tx_by_ts = []
        ts_box_groups = []
        for cx, name in zip(ts_centers_x, ["t = 14", "t = 15", "t = 16"]):
            row = [tx_node(np.array([cx, tx_y, 0]) + o, radius=0.13)
                   for o in tx_offs]
            tx_by_ts.append(row)
            box = SurroundingRectangle(VGroup(*row), buff=0.22,
                                       color=TIMESTEP_COLOR,
                                       stroke_opacity=0.55, stroke_width=2)
            lbl = Text(name, font_size=16, color=TIMESTEP_COLOR
                       ).next_to(box, UP, buff=0.05)
            ts_box_groups.append(VGroup(box, lbl))

        tx_tx_edges = []
        for row in tx_by_ts:
            tx_tx_edges.append(directed(row[0], row[1]))
            tx_tx_edges.append(directed(row[1], row[2]))

        wallet_xs = [-4.2, -2.5, -0.9, 0.9, 2.5, 4.2]
        wallets = [wallet_node(np.array([x, wallet_y, 0]), side=0.28)
                   for x in wallet_xs]

        # bipartite edges. The four touching W0/W3 are CROSS-TIMESTEP.
        bipartite = [
            (0, 0, 0, "in"),
            (1, 0, 1, "in"),
            (2, 0, 2, "out"),
            (0, 1, 0, "in"),    # cross-timestep (W0)
            (3, 1, 1, "out"),
            (3, 2, 0, "in"),    # cross-timestep (W3)
            (4, 2, 1, "in"),
            (5, 2, 2, "out"),
        ]
        bipartite_arrows = []
        for idx, (w_i, ts_i, tx_i, direction) in enumerate(bipartite):
            w = wallets[w_i]
            t = tx_by_ts[ts_i][tx_i]
            color = ADDR_TX_EDGE_COLOR if direction == "in" else TX_ADDR_EDGE_COLOR
            src, dst = (w, t) if direction == "in" else (t, w)
            arr = directed(src, dst, color=color, stroke_width=2.2, buff=0.13)
            if w_i in (0, 3):
                arr.set_stroke(width=4, color=YELLOW)
            bipartite_arrows.append(arr)

        aa_edges = [
            directed(wallets[1], wallets[2], color=ADDR_ADDR_EDGE_COLOR,
                     stroke_width=2.2, buff=0.15),
            directed(wallets[4], wallets[5], color=ADDR_ADDR_EDGE_COLOR,
                     stroke_width=2.2, buff=0.15),
        ]

        ring_w0 = Circle(radius=0.27, color=YELLOW,
                         stroke_width=2.5).move_to(wallets[0])
        ring_w3 = Circle(radius=0.27, color=YELLOW,
                         stroke_width=2.5).move_to(wallets[3])

        # ---- cross-timestep callout ----
        cross_lbl = Text(
            "cross-timestep paths (yellow): the same wallet appears in multiple timesteps "
            "via Addr↔Tx edges",
            font_size=14, color=YELLOW_B,
        ).move_to(np.array([0, wallet_y - 0.85, 0]))

        # ---- feature panels ----
        # focus_tx is in t=15 (a within-timestep tx); focus_w is W4 (NOT one of
        # the cross-timestep wallets, to keep the rings unambiguous).
        focus_tx = tx_by_ts[1][1]
        focus_w = wallets[4]
        focus_tx_ring = Circle(radius=0.22, color=YELLOW,
                               stroke_width=3).move_to(focus_tx)
        focus_w_ring = Circle(radius=0.27, color=YELLOW,
                              stroke_width=3).move_to(focus_w)

        tx_feat_lines = VGroup(
            Text("tx node features", font_size=14, weight=BOLD),
            Text("• txId, Time step", font_size=11),
            Text("• 94 local features", font_size=11, color=BLUE_B),
            Text("• 72 aggregate features", font_size=11, color=BLUE_B),
            Text("• class ∈ {illicit, licit, unknown}",
                 font_size=11, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        tx_feat_panel = VGroup(
            SurroundingRectangle(tx_feat_lines, buff=0.12, color=BLUE_B,
                                 stroke_opacity=0.9,
                                 fill_color=BLACK, fill_opacity=0.85),
            tx_feat_lines,
        ).to_corner(DL, buff=0.25)

        w_feat_lines = VGroup(
            Text("wallet node features", font_size=14, weight=BOLD),
            Text("• address, Time step", font_size=11),
            Text("• 55 wallet features", font_size=11, color=ORANGE),
            Text("  (lifetime, BTC sent/recv, fees,", font_size=10,
                 color=GREY_B),
            Text("   #txs, block stats, …)", font_size=10, color=GREY_B),
            Text("• class ∈ {illicit, licit, unknown}",
                 font_size=11, color=GREY_A),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.04)
        w_feat_panel = VGroup(
            SurroundingRectangle(w_feat_lines, buff=0.12, color=ORANGE,
                                 stroke_opacity=0.9,
                                 fill_color=BLACK, fill_opacity=0.85),
            w_feat_lines,
        ).to_corner(DR, buff=0.25)

        tx_arrow = Arrow(
            focus_tx.get_center() + LEFT * 0.18 + DOWN * 0.10,
            tx_feat_panel.get_corner(UR) + LEFT * 0.40 + UP * 0.05,
            buff=0.12, stroke_width=2.5, color=YELLOW,
            max_tip_length_to_length_ratio=0.05,
        )
        w_arrow = Arrow(
            focus_w.get_center() + RIGHT * 0.20 + DOWN * 0.05,
            w_feat_panel.get_corner(UL) + RIGHT * 0.40 + UP * 0.05,
            buff=0.12, stroke_width=2.5, color=YELLOW,
            max_tip_length_to_length_ratio=0.05,
        )

        # ---- compose ----
        self.add(*ts_box_groups)
        for row in tx_by_ts:
            self.add(*row)
        self.add(*tx_tx_edges)
        self.add(*wallets)
        self.add(*aa_edges, *bipartite_arrows)
        self.add(ring_w0, ring_w3, focus_tx_ring, focus_w_ring)
        self.add(cross_lbl)
        self.add(tx_feat_panel, w_feat_panel, tx_arrow, w_arrow)
        self.add(header, legend_panel)
        self.wait(0.05)
