use std::{cmp::Ordering, sync::Mutex};

use std::thread;

use egui::epaint::QuadraticBezierShape;
use serde_json::Value;

use crate::graph::{
    AllSmallGraphs32, CycleSearchResult, DirectedPersistence, Edge, ExhaustiveCycleSearchResult,
    Graph, SimplexWiseSweepFiltration, SmallGraphView32, SmallGraphsWithEdgesSet32, SweepDir,
    Vertex, cycle_search, exhaustive_cycle_search,
};
use egui::{Color32, Pos2, Rect, Stroke, Ui, Vec2};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[derive(Clone)]
struct CycleSearchResSummary {
    maximal_minimal_cycle: Option<Vec<u16>>,
    has_non_part: bool,
}

fn world_to_screen(pos: Pos2, canvas_rect: Rect, grid_size: f32) -> Pos2 {
    let center = canvas_rect.center();
    Pos2::new(center.x + pos.x * grid_size, center.y - pos.y * grid_size)
}

fn screen_to_world(pos: Pos2, canvas_rect: Rect, grid_size: f32) -> Pos2 {
    let center = canvas_rect.center();
    Pos2::new(
        (pos.x - center.x) / grid_size,
        -(pos.y - center.y) / grid_size,
    )
}

fn lerp_color(start: Color32, end: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let lerp_channel = |a: u8, b: u8| -> u8 {
        let a = a as f32;
        let b = b as f32;
        (a + (b - a) * t).round().clamp(0.0, 255.0) as u8
    };
    Color32::from_rgba_unmultiplied(
        lerp_channel(start.r(), end.r()),
        lerp_channel(start.g(), end.g()),
        lerp_channel(start.b(), end.b()),
        lerp_channel(start.a(), end.a()),
    )
}

// TODO refactor common code with draw_graph_preview
fn graph_canvas_transform(rect: Rect, vertices: &[Vertex]) -> (f32, f64, f64) {
    if vertices.is_empty() {
        return (1.0, 0.0, 0.0);
    }

    let (min_x, min_y) = vertices
        .iter()
        .fold((f64::INFINITY, f64::INFINITY), |(min_x, min_y), v| {
            (min_x.min(v.x), min_y.min(v.y))
        });
    let (max_x, max_y) = vertices.iter().fold(
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
        |(max_x, max_y), v| (max_x.max(v.x), max_y.max(v.y)),
    );

    let graph_width = (max_x - min_x).max(1e-3);
    let graph_height = (max_y - min_y).max(1e-3);
    let scale = (rect.width() / graph_width as f32).min(rect.height() / graph_height as f32) * 0.85;
    let grid_size = if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        40.0
    };
    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;

    (grid_size, center_x, center_y)
}

// TODO refactor common code with draw_graph_preview
fn project_vertex(
    rect: Rect,
    grid_size: f32,
    center_x: f64,
    center_y: f64,
    vertex: &Vertex,
) -> Pos2 {
    world_to_screen(
        Pos2::new((vertex.x - center_x) as f32, (vertex.y - center_y) as f32),
        rect,
        grid_size,
    )
}

fn arc_edge_painter(
    a: Pos2,
    b: Pos2,
    arrow_head: bool,
    stroke: egui::Stroke,
    painter: &egui::Painter,
) {
    let ab = b - a;
    let c = a + 0.5 * ab + 0.3 * ab.rot90();
    let shape = QuadraticBezierShape::from_points_stroke(
        [a, c, b],
        false,
        egui::Color32::default(),
        stroke,
    );
    painter.add(shape);
    if !arrow_head {
        return;
    } // arrow head
    let dir = (b - c).normalized();
    let (c, s) = (35.0f32.cos(), 35.0f32.sin());
    let l_pt = b + 10. * Vec2::new(c * dir.x - s * dir.y, s * dir.x + c * dir.y);
    let r_pt = b + 10. * Vec2::new(c * dir.x + s * dir.y, -s * dir.x + c * dir.y);
    painter.line_segment([b, l_pt], stroke);
    painter.line_segment([b, r_pt], stroke);
}

#[derive(Default, Debug)]
struct CycleSearchPanel {
    cycle_data: CycleSearchResult,
    graphs: Vec<Graph>,
}

impl CycleSearchPanel {
    fn new(
        set: &Vec<SmallGraphView32>,
        include_common_edges: bool,
        base_graph: &Graph,
        dir: SweepDir,
    ) -> Self {
        let res = cycle_search(base_graph, set, dir, !include_common_edges, true);
        // let n_min = res.minimal_trafo();
        Self {
            cycle_data: res,
            // minimal_cycle_counts: n_min,
            graphs: set
                .iter()
                .map(|g| g.to_graph(base_graph.vertices.clone()))
                .collect(),
        }
    }

    fn edge_sets_for_pair(graph_a: &Graph, graph_b: &Graph) -> (Vec<Edge>, Vec<Edge>, Vec<Edge>) {
        let mut common = Vec::new();
        let mut unique_a = Vec::new();
        let mut unique_b = Vec::new();

        let n_vtx = graph_a.vertices.len();
        if n_vtx == 0 {
            return (common, unique_a, unique_b);
        }

        for a in 0..n_vtx {
            for b in (a + 1)..n_vtx {
                let a_has = graph_a.has_edge(Edge::new(a, b)); // (graph_a.edge_bits[div] & mask) != 0;
                let b_has = graph_b.has_edge(Edge::new(a, b)); // (graph_b.edge_bits[div] & mask) != 0;

                match (a_has, b_has) {
                    (true, true) => common.push(Edge::new(a, b)),
                    (true, false) => unique_a.push(Edge::new(a, b)),
                    (false, true) => unique_b.push(Edge::new(a, b)),
                    (false, false) => {}
                }
            }
        }

        (common, unique_a, unique_b)
    }

    fn draw_joint_cycle_graph(
        painter: &egui::Painter,
        rect: Rect,
        base_graph: &Graph,
        common_edges: &[Edge],
        cycles: &Vec<Vec<Edge>>,
    ) {
        if base_graph.vertices.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No vertices",
                egui::FontId::default(),
                Color32::GRAY,
            );
            return;
        }

        let (grid_size, center_x, center_y) = graph_canvas_transform(rect, &base_graph.vertices);

        let common_color = Color32::from_gray(150);
        // draw common edges in gray... will be overdrawn by cycles that include them..
        // if that setting is set
        for edge in common_edges {
            if edge.a >= base_graph.vertices.len() || edge.b >= base_graph.vertices.len() {
                continue;
            }
            let p1 = project_vertex(
                rect,
                grid_size,
                center_x,
                center_y,
                &base_graph.vertices[edge.a],
            );
            let p2 = project_vertex(
                rect,
                grid_size,
                center_x,
                center_y,
                &base_graph.vertices[edge.b],
            );
            arc_edge_painter(p1, p2, false, Stroke::new(2.0, common_color), painter);
            // painter.line_segment([p1, p2], Stroke::new(2.0, common_color));
        }

        let red_light = Color32::from_rgb(255, 190, 190);
        let red_dark = Color32::from_rgb(170, 30, 30);
        let blue_light = Color32::from_rgb(190, 215, 255);
        let blue_dark = Color32::from_rgb(30, 70, 185);
        let green_tint = Color32::from_rgb(39, 255, 0);

        let total = cycles.len();
        let denom = if total > 1 { (total - 1) as f32 } else { 1.0 };
        let mut indices = (0..cycles.len()).collect::<Vec<usize>>();
        // sort by cycle length descending
        indices.sort_by(|&a, &b| cycles[b].len().cmp(&cycles[a].len()));
        for (idx, cycle) in indices.iter().enumerate() {
            let cycle = &cycles[*cycle];
            let c_factor = idx as f32 / denom;
            let red_color = lerp_color(red_light, red_dark, c_factor);
            let red_color = lerp_color(red_color, green_tint, 0.4 * c_factor);
            let blue_color = lerp_color(blue_light, blue_dark, c_factor);
            let blue_color = lerp_color(blue_color, green_tint, 0.4 * c_factor);
            let mut is_i = true;
            for edg in cycle {
                let p1 = project_vertex(
                    rect,
                    grid_size,
                    center_x,
                    center_y,
                    &base_graph.vertices[edg.a],
                );
                let p2 = project_vertex(
                    rect,
                    grid_size,
                    center_x,
                    center_y,
                    &base_graph.vertices[edg.b],
                );
                let color = if is_i { red_color } else { blue_color };
                arc_edge_painter(p1, p2, true, Stroke::new(3.2 - c_factor, color), painter);
                is_i = !is_i;
            }
        }

        for (idx, vertex) in base_graph.vertices.iter().enumerate() {
            let pos = project_vertex(rect, grid_size, center_x, center_y, vertex);
            painter.circle_filled(pos, 5.0, Color32::WHITE);
            painter.circle_stroke(pos, 5.0, Stroke::new(1.5, Color32::BLACK));
            painter.text(
                pos + Vec2::new(0.0, -10.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{}", idx),
                egui::FontId::proportional(12.0),
                Color32::BLACK,
            );
        }
    }

    fn draw_single_graph_with_colors(
        painter: &egui::Painter,
        rect: Rect,
        base_graph: &Graph,
        unique_edges: &[Edge],
        common_edges: &[Edge],
        unique_color: Color32,
    ) {
        if base_graph.vertices.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No vertices",
                egui::FontId::default(),
                Color32::GRAY,
            );
            return;
        }

        let (grid_size, center_x, center_y) = graph_canvas_transform(rect, &base_graph.vertices);
        let common_color = Color32::from_gray(150);

        for edge in common_edges {
            let p1 = project_vertex(
                rect,
                grid_size,
                center_x,
                center_y,
                &base_graph.vertices[edge.a],
            );
            let p2 = project_vertex(
                rect,
                grid_size,
                center_x,
                center_y,
                &base_graph.vertices[edge.b],
            );
            arc_edge_painter(p1, p2, false, Stroke::new(2.0, common_color), painter);
            // painter.line_segment([p1, p2], Stroke::new(2.0, common_color));
        }

        for edge in unique_edges {
            let p1 = project_vertex(
                rect,
                grid_size,
                center_x,
                center_y,
                &base_graph.vertices[edge.a],
            );
            let p2 = project_vertex(
                rect,
                grid_size,
                center_x,
                center_y,
                &base_graph.vertices[edge.b],
            );
            arc_edge_painter(p1, p2, false, Stroke::new(2.8, unique_color), painter);
            // painter.line_segment([p1, p2], Stroke::new(2.8, unique_color));
        }

        for (idx, vertex) in base_graph.vertices.iter().enumerate() {
            let pos = project_vertex(rect, grid_size, center_x, center_y, vertex);
            painter.circle_filled(pos, 5.0, Color32::WHITE);
            painter.circle_stroke(pos, 5.0, Stroke::new(1.5, Color32::BLACK));
            painter.text(
                pos + Vec2::new(0.0, -10.0),
                egui::Align2::CENTER_BOTTOM,
                format!("{}", idx),
                egui::FontId::proportional(12.0),
                Color32::BLACK,
            );
        }
    }

    fn draw_partitionable_pair(
        graph_a: &Graph,
        graph_b: &Graph,
        cycles: &Vec<Vec<Edge>>,
        graph_a_lable: &str,
        graph_b_lable: &str,
        ui: &mut Ui,
    ) {
        // let cycles = &cycles[0];
        let (common_edges, _, _) = Self::edge_sets_for_pair(graph_a, graph_b);

        egui::Frame::group(ui.style()).show(ui, |ui| {
            // graph box
            ui.vertical(|ui| {
                ui.label(format!(
                    "Graph {} ↔ Graph {} ({} cycles)",
                    graph_a_lable,
                    graph_b_lable,
                    cycles.len()
                ));
                ui.add_space(6.0);

                let (response, painter) = ui
                    .allocate_painter(Vec2::new(ui.available_width(), 220.), egui::Sense::hover());
                Self::draw_joint_cycle_graph(
                    &painter,
                    response.rect,
                    graph_a,
                    &common_edges,
                    &cycles,
                );

                ui.add_space(4.0);
                if cycles.is_empty() {
                    ui.label("No alternating cycles were returned.");
                } else {
                    ui.horizontal_wrapped(|ui| {
                        for (cycle_idx, cycle) in cycles.iter().enumerate() {
                            ui.label(format!("Cycle {}: {} edges", cycle_idx + 1, cycle.len()));
                        }
                    });
                }
            });
        });
    }

    fn draw_non_partitionable_pair(
        graph_a: &Graph,
        graph_b: &Graph,
        graph_a_lable: &str,
        graph_b_lable: &str,
        ui: &mut Ui,
    ) {
        let (common_edges, unique_i, unique_j) = Self::edge_sets_for_pair(graph_a, graph_b);

        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.vertical(|ui| {
                ui.label(format!(
                    "Graph {} ↔ Graph {} (no partition)",
                    graph_a_lable, graph_b_lable
                ));
                ui.add_space(6.0);

                ui.horizontal(|ui| {
                    let graph_size = Vec2::new(ui.available_width() / 2., 220.0);
                    let red_light = Color32::from_rgb(255, 190, 190);
                    let red_dark = Color32::from_rgb(170, 30, 30);
                    let blue_light = Color32::from_rgb(190, 215, 255);
                    let blue_dark = Color32::from_rgb(30, 70, 185);

                    ui.vertical(|ui| {
                        ui.label(format!("Graph {}", graph_a_lable));
                        let (response, painter) =
                            ui.allocate_painter(graph_size, egui::Sense::hover());
                        Self::draw_single_graph_with_colors(
                            &painter,
                            response.rect,
                            graph_a,
                            &unique_i,
                            &common_edges,
                            lerp_color(red_light, red_dark, 0.5),
                        );
                    });

                    ui.add_space(12.0);

                    ui.vertical(|ui| {
                        ui.label(format!("Graph {}", graph_b_lable));
                        let (response, painter) =
                            ui.allocate_painter(graph_size, egui::Sense::hover());
                        Self::draw_single_graph_with_colors(
                            &painter,
                            response.rect,
                            graph_b,
                            &unique_j,
                            &common_edges,
                            lerp_color(blue_light, blue_dark, 0.5),
                        );
                    });
                });
            });
        });
    }

    fn draw(&self, ui: &mut Ui) {
        if self.graphs.is_empty() {
            ui.label("Collision set is empty.");
            return;
        }

        // ui.heading(format!("Collision set {}", set_idx + 1));
        ui.label(format!("Graphs in set: {}", self.graphs.len()));
        ui.add_space(8.0);

        ui.strong(format!(
            "Partitionable pairs: {}, Non partitionable pairs: {}",
            self.cycle_data.partitionable_pairs.len(),
            self.cycle_data.non_partitionable_pairs.len()
        ));

        ui.add_space(4.0);
        let n_pairs = self.cycle_data.non_partitionable_pairs.len()
            + self.cycle_data.partitionable_pairs.len();

        let min_width = 400.;
        let n_col = (ui.available_width() / min_width).floor() as usize;
        let r_width = ui.available_width() / n_col as f32 - 10.;
        let n_rows = (n_pairs as f32 / n_col as f32).ceil() as usize;

        egui::ScrollArea::vertical().show_rows(ui, 300., n_rows, |ui, row_range| {
            for row in row_range {
                ui.horizontal(|ui| {
                    // ui.set_width(ui.available_width());

                    for offset in 0..n_col {
                        ui.vertical(|ui| {
                            ui.set_width(r_width);
                            ui.vertical(|ui| {
                                let row = row * n_col + offset;
                                let n_part_pairs = self.cycle_data.partitionable_pairs.len();
                                if row < n_part_pairs {
                                    let p_pair = &self.cycle_data.partitionable_pairs[row];
                                    Self::draw_partitionable_pair(
                                        &self.graphs[p_pair.0],
                                        &self.graphs[p_pair.1],
                                        &p_pair.2,
                                        &format!("{}", p_pair.0 + 1),
                                        &format!("{}", p_pair.1 + 1),
                                        ui,
                                    );
                                } else if row - n_part_pairs
                                    < self.cycle_data.non_partitionable_pairs.len()
                                {
                                    let idx = row - n_part_pairs;
                                    let pair = self.cycle_data.non_partitionable_pairs[idx];
                                    Self::draw_non_partitionable_pair(
                                        &self.graphs[pair.0],
                                        &self.graphs[pair.1],
                                        &format!("{}", pair.0 + 1),
                                        &format!("{}", pair.1 + 1),
                                        ui,
                                    );
                                }
                            });
                        });
                    }
                });
            }
        });
    }
}

struct CollisionSetsPanel {
    sort_order: [bool; 6],
    collision_sets: Vec<Vec<SmallGraphView32>>,
    collision_sets_persistence: Vec<Vec<(DirectedPersistence, DirectedPersistence)>>,
    collision_sets_graph: Graph, // graph these were made from
    sweep_dir: SweepDir,
    first_second_homology_ratio: f64,
    // collision_sets_cycles: Vec<(ExhaustiveCycleSearchResult, Vec<usize>)>,
    collision_sets_cycle_summaries: Vec<CycleSearchResSummary>,
    // selected_collision_set_graphs: Vec<Vec<bool>>,
    exclude_common_edges: bool,
    help_button: bool,
    filter_non_part: bool,
    filter_n_graphs: String,
    filter_n_comp: String,
    filter_n_cyc: String,
    filter_part: String,
    filter_off_diag: String,
    filter_masks: [Vec<bool>; 6],
    filter_rebuild: bool,
    filter_indices: Vec<usize>,
    selected_set: Option<usize>,
}

fn draw_graph_preview(painter: &egui::Painter, graph: &Graph, rect: Rect) {
    // calculate bounds in
    let (min_x, min_y) = graph
        .vertices
        .iter()
        .fold((f64::INFINITY, f64::INFINITY), |(min_x, min_y), v| {
            (min_x.min(v.x), min_y.min(v.y))
        });
    let (max_x, max_y) = graph.vertices.iter().fold(
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
        |(max_x, max_y), v| (max_x.max(v.x), max_y.max(v.y)),
    );

    // Calculate offsets to center and scale the graph
    let graph_width = (max_x - min_x).max(1e-5);
    let graph_height = (max_y - min_y).max(1e-5);
    let scale = (rect.width() / graph_width as f32).min(rect.height() / graph_height as f32) * 0.8;
    let grid_size = scale;
    // bounds center in world coordinates
    let center_x = (min_x + max_x) / 2.0;
    let center_y = (min_y + max_y) / 2.0;

    // Draw edges
    for i in 0..graph.vertices.len() {
        for j in (i + 1)..graph.vertices.len() {
            if graph.has_edge(Edge::new(i, j)) {
                let v1 = &graph.vertices[i];
                let v2 = &graph.vertices[j];

                let p1 = world_to_screen(
                    Pos2::new((v1.x - center_x) as f32, (v1.y - center_y) as f32),
                    rect,
                    grid_size,
                );
                let p2 = world_to_screen(
                    Pos2::new((v2.x - center_x) as f32, (v2.y - center_y) as f32),
                    rect,
                    grid_size,
                );
                arc_edge_painter(p1, p2, false, Stroke::new(1.5, Color32::GRAY), painter);
                // painter.line_segment([p1, p2], Stroke::new(1.5, Color32::GRAY));
            }
        }
    }

    // Draw vertices
    for vertex in &graph.vertices {
        let pos = world_to_screen(
            Pos2::new((vertex.x - center_x) as f32, (vertex.y - center_y) as f32),
            rect,
            grid_size,
        );

        painter.circle_filled(pos, 4.0, Color32::WHITE);
        painter.circle_stroke(pos, 4.0, Stroke::new(1.5, Color32::BLACK));
    }
}

fn draw_small_persistence_diagram(
    ui: &mut egui::Ui,
    diagram: &crate::graph::PersistenceDiagram,
    color: Color32,
    width: f32,
    height: f32,
) {
    let (response, painter) = ui.allocate_painter(Vec2::new(width, height), egui::Sense::hover());

    let rect = response.rect;

    if diagram.points.is_empty() {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "No points",
            egui::FontId::proportional(10.0),
            Color32::GRAY,
        );
        return;
    }

    // Find bounds for finite points
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;

    for pt in diagram.points.iter() {
        min_val = min_val.min(pt.x);
        max_val = if pt.y.is_infinite() {
            max_val.max(pt.x)
        } else {
            max_val.max(pt.y)
        };
    }

    // Separate finite and infinite points
    let finite_points: Vec<_> = diagram.points.iter().filter(|p| p.y.is_finite()).collect();
    let infinite_points: Vec<_> = diagram
        .points
        .iter()
        .filter(|p| p.y.is_infinite())
        .collect();

    if max_val == min_val {
        max_val = min_val + 1.0;
    }

    let margin = 10.0;
    let plot_rect = rect.shrink(margin);

    // Draw diagonal
    painter.line_segment(
        [plot_rect.left_bottom(), plot_rect.right_top()],
        Stroke::new(0.5, Color32::from_gray(180)),
    );

    // Draw axes
    painter.line_segment(
        [
            Pos2::new(plot_rect.left(), plot_rect.bottom()),
            plot_rect.right_bottom(),
        ],
        Stroke::new(0.5, Color32::BLACK),
    );
    painter.line_segment(
        [
            Pos2::new(plot_rect.left(), plot_rect.bottom()),
            plot_rect.left_top(),
        ],
        Stroke::new(0.5, Color32::BLACK),
    );

    // Draw finite points
    for point in finite_points {
        let x_norm = ((point.x - min_val) / (max_val - min_val)) as f32;
        let y_norm = ((point.y - min_val) / (max_val - min_val)) as f32;

        let screen_pos = Pos2::new(
            plot_rect.left() + x_norm * plot_rect.width(),
            plot_rect.bottom() - y_norm * plot_rect.height(),
        );

        let radius = 2.0 + ((point.mult - 1) as f32).atan() + 1.;
        // let radius = radius * 1.5;
        painter.circle_filled(screen_pos, radius, color);
        painter.circle_stroke(screen_pos, radius, Stroke::new(1.0, Color32::BLACK));
        // small mult inside of point
        if point.mult > 1 {
            painter.text(
                screen_pos,
                egui::Align2::CENTER_CENTER,
                format!("{}", point.mult),
                egui::FontId::proportional(10.0),
                Color32::WHITE,
            );
        }
    }

    // Draw infinite points at the top
    for point in &infinite_points {
        let x_norm = ((point.x - min_val) / (max_val - min_val)) as f32;

        let screen_pos = Pos2::new(
            plot_rect.left() + x_norm * plot_rect.width(),
            plot_rect.top(),
        );

        let radius = 2.0 + ((point.mult - 1) as f32).atan() + 1.;
        // let radius = radius * 1.5;
        painter.circle_filled(screen_pos, radius, color);
        painter.circle_stroke(screen_pos, radius, Stroke::new(1.0, Color32::BLACK));
        // small mult inside of point
        if point.mult > 1 {
            painter.text(
                screen_pos,
                egui::Align2::CENTER_CENTER,
                format!("{}", point.mult),
                egui::FontId::proportional(10.0),
                Color32::WHITE,
            );
        }

        // Small arrow for infinity
        let arrow_start = screen_pos + Vec2::new(0.0, -radius - 1.0);
        let arrow_end = arrow_start + Vec2::new(0.0, -4.0);
        painter.line_segment([arrow_start, arrow_end], Stroke::new(0.8, Color32::BLACK));
        painter.line_segment(
            [arrow_end, arrow_end + Vec2::new(-1.5, 1.5)],
            Stroke::new(0.8, Color32::BLACK),
        );
        painter.line_segment(
            [arrow_end, arrow_end + Vec2::new(1.5, 1.5)],
            Stroke::new(0.8, Color32::BLACK),
        );
    }
}

fn draw_collision_set(
    ui: &mut Ui,
    collision_set: &Vec<SmallGraphView32>,
    persistence: &Vec<(DirectedPersistence, DirectedPersistence)>,
    mod_graph: &mut Graph,
    has_non_part: bool,
    display_cycle_sarch: &mut bool,
) {
    if collision_set.is_empty() {
        // ui.set_width(ui.available_width());
        ui.heading("No graphs availible");
        return;
    }
    let available_width = ui.available_width();
    let graph_width = 300.0;
    let diagram_width = 150.0;
    let item_width = graph_width + diagram_width + 10.0;
    let graph_height = 150.0;
    let spacing = 10.0;
    let graphs_per_row = ((available_width + spacing) / (item_width + spacing))
        .floor()
        .max(1.0) as usize;

    ui.vertical(|ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            // button to display pairwise cycles
            ui.strong(format!(
                "Has non partitionable: {}",
                if has_non_part { "Yes" } else { "No" }
            ));
            if ui.button("Show Cycles").clicked() {
                *display_cycle_sarch = true;
            }
            ui.add_space(5.0);
            ui.separator();
            ui.label("Selected graphs:");
            ui.add_space(5.0);

            // Draw selected graphs in grid
            let selected_indices: Vec<usize> = collision_set
                .iter()
                .enumerate()
                // .filter(|(idx, _)| self.selected_collision_set_graphs[set_idx][*idx])
                .map(|(idx, _)| idx)
                .collect();

            for chunk in selected_indices.chunks(graphs_per_row) {
                ui.horizontal(|ui| {
                    for &graph_idx in chunk {
                        ui.vertical(|ui| {
                            ui.label(format!("Graph {}", graph_idx + 1));
                            ui.horizontal(|ui| {
                                // Graph preview
                                let (response, painter) = ui.allocate_painter(
                                    Vec2::new(graph_width, graph_height),
                                    egui::Sense::hover(),
                                );
                                let rect = response.rect;
                                collision_set[graph_idx].write_to_graph_with_id_vtx(mod_graph);
                                draw_graph_preview(&painter, &mod_graph, rect);

                                ui.add_space(5.0);

                                ui.horizontal(|ui| {
                                    ui.vertical(|ui| {
                                        ui.label("forward:");
                                        ui.label("H₀"); // Compose key is great.
                                        draw_small_persistence_diagram(
                                            ui,
                                            &persistence[graph_idx].0.connected_comp,
                                            Color32::BLUE,
                                            diagram_width,
                                            70.0,
                                        );
                                        ui.add_space(3.0);
                                        ui.label("H₁");
                                        draw_small_persistence_diagram(
                                            ui,
                                            &persistence[graph_idx].0.cycles,
                                            Color32::RED,
                                            diagram_width,
                                            70.0,
                                        );
                                    });
                                    ui.vertical(|ui| {
                                        ui.label("backward:");
                                        ui.label("H₀"); // Compose key is great.
                                        draw_small_persistence_diagram(
                                            ui,
                                            &persistence[graph_idx].1.connected_comp,
                                            Color32::BLUE,
                                            diagram_width,
                                            70.0,
                                        );
                                        ui.add_space(3.0);
                                        ui.label("H₁");
                                        draw_small_persistence_diagram(
                                            ui,
                                            &persistence[graph_idx].1.cycles,
                                            Color32::RED,
                                            diagram_width,
                                            70.0,
                                        );
                                    });
                                });
                            });
                        });
                        ui.add_space(spacing);
                    }
                });
                ui.add_space(spacing);
            }
        });
    });
}

impl CollisionSetsPanel {
    fn new() -> Self {
        Self {
            sort_order: [false; 6],
            collision_sets: Vec::new(),
            collision_sets_persistence: Vec::new(),
            collision_sets_graph: Graph::new(0),
            first_second_homology_ratio: 0.5,
            exclude_common_edges: true,
            collision_sets_cycle_summaries: Vec::new(),
            sweep_dir: SweepDir::default(),
            help_button: false,
            filter_indices: Vec::new(),
            filter_n_comp: String::new(),
            filter_n_cyc: String::new(),
            filter_n_graphs: String::new(),
            filter_non_part: false,
            filter_off_diag: String::new(),
            filter_part: String::new(),
            filter_masks: [
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            ],
            filter_rebuild: false,
            selected_set: None,
        }
    }

    fn is_reversed(&self, key: CollisionSetSortOrder) -> bool {
        match key {
            CollisionSetSortOrder::Size => self.sort_order[0],
            CollisionSetSortOrder::Components => self.sort_order[1],
            CollisionSetSortOrder::Cycles => self.sort_order[2],
            CollisionSetSortOrder::OffDiagonalPoints(_) => self.sort_order[3],
            CollisionSetSortOrder::HasNonPartitionable => self.sort_order[4],
            CollisionSetSortOrder::MinimalPartition => self.sort_order[5],
        }
    }
    fn set_reversed(&mut self, key: CollisionSetSortOrder, value: bool) {
        match key {
            CollisionSetSortOrder::Size => self.sort_order[0] = value,
            CollisionSetSortOrder::Components => self.sort_order[1] = value,
            CollisionSetSortOrder::Cycles => self.sort_order[2] = value,
            CollisionSetSortOrder::OffDiagonalPoints(_) => self.sort_order[3] = value,
            CollisionSetSortOrder::HasNonPartitionable => self.sort_order[4] = value,
            CollisionSetSortOrder::MinimalPartition => self.sort_order[5] = value,
        }
    }

    fn sort_by(&mut self, sort_by: CollisionSetSortOrder, reverse: bool) {
        // Create indices and sort them
        let mut indices: Vec<usize> = (0..self.collision_sets.len()).collect();
        indices.sort_by(|&a, &b| {
            let ord = match sort_by {
                CollisionSetSortOrder::Components => {
                    let a_cc = self.collision_sets_persistence[a]
                        .first()
                        .map_or(0, |(fwd, _)| {
                            fwd.connected_comp
                                .points
                                .iter()
                                .map(|p| ((p.y == f64::INFINITY) as u32) * p.mult)
                                .sum::<u32>()
                            // should be the same we hope // .max(bkw.connected_comp.points.iter().map(|p| ((p.y == f64::INFINITY) as u32) * p.mult).sum::<u32>())
                            // TODO: test for same
                        });
                    let b_cc = self.collision_sets_persistence[b]
                        .first()
                        .map_or(0, |(fwd, _)| {
                            fwd.connected_comp
                                .points
                                .iter()
                                .map(|p| ((p.y == f64::INFINITY) as u32) * p.mult)
                                .sum::<u32>()
                        });
                    b_cc.cmp(&a_cc)
                }
                CollisionSetSortOrder::Cycles => {
                    let a_cyc = self.collision_sets_persistence[a]
                        .first()
                        .map_or(0, |(fwd, _)| {
                            fwd.cycles
                                .points
                                .iter()
                                .map(|p| ((p.y == f64::INFINITY) as u32) * p.mult)
                                .sum::<u32>()
                            // should be the same we hope // .max(bkw.connected_comp.points.iter().map(|p| ((p.y == f64::INFINITY) as u32) * p.mult).sum::<u32>())
                            // TODO: test for same
                        });
                    let b_cyc = self.collision_sets_persistence[b]
                        .first()
                        .map_or(0, |(fwd, _)| {
                            fwd.cycles
                                .points
                                .iter()
                                .map(|p| ((p.y == f64::INFINITY) as u32) * p.mult)
                                .sum::<u32>()
                        });
                    b_cyc.cmp(&a_cyc)
                }
                CollisionSetSortOrder::OffDiagonalPoints(x) if x >= 0. && x <= 1. => {
                    let a_cc = self.collision_sets_persistence[a].first().map_or(
                        (0., 0.),
                        |(fwd, bkw)| {
                            (
                                fwd.connected_comp
                                    .points
                                    .iter()
                                    .map(|p| !p.is_diagonal() as u32)
                                    .sum::<u32>() as f64,
                                bkw.connected_comp
                                    .points
                                    .iter()
                                    .map(|p| !p.is_diagonal() as u32)
                                    .sum::<u32>() as f64,
                            )
                        },
                    );
                    let b_cc = self.collision_sets_persistence[b].first().map_or(
                        (0., 0.),
                        |(fwd, bkw)| {
                            (
                                fwd.connected_comp
                                    .points
                                    .iter()
                                    .map(|p| !p.is_diagonal() as u32)
                                    .sum::<u32>() as f64,
                                bkw.connected_comp
                                    .points
                                    .iter()
                                    .map(|p| !p.is_diagonal() as u32)
                                    .sum::<u32>() as f64,
                            )
                        },
                    );
                    let a_cyc = self.collision_sets_persistence[a].first().map_or(
                        (0., 0.),
                        |(fwd, bkw)| {
                            (
                                fwd.cycles
                                    .points
                                    .iter()
                                    .map(|p| !p.is_diagonal() as u32)
                                    .sum::<u32>() as f64,
                                bkw.cycles
                                    .points
                                    .iter()
                                    .map(|p| !p.is_diagonal() as u32)
                                    .sum::<u32>() as f64,
                            )
                        },
                    );
                    let b_cyc = self.collision_sets_persistence[b].first().map_or(
                        (0., 0.),
                        |(fwd, bkw)| {
                            (
                                fwd.cycles
                                    .points
                                    .iter()
                                    .map(|p| !p.is_diagonal() as u32)
                                    .sum::<u32>() as f64,
                                bkw.cycles
                                    .points
                                    .iter()
                                    .map(|p| !p.is_diagonal() as u32)
                                    .sum::<u32>() as f64,
                            )
                        },
                    );
                    let a = (a_cyc.0 * x + a_cc.0 * (1. - x)).max(a_cyc.1 * x + a_cc.1 * (1. - x));
                    let b = (b_cyc.0 * x + b_cc.0 * (1. - x)).max(b_cyc.1 * x + b_cc.1 * (1. - x));
                    b.partial_cmp(&a).unwrap()
                }
                CollisionSetSortOrder::Size => self.collision_sets[b]
                    .len()
                    .cmp(&self.collision_sets[a].len()),
                CollisionSetSortOrder::HasNonPartitionable => {
                    let a_np = self.collision_sets_cycle_summaries[a].has_non_part;
                    let b_np = self.collision_sets_cycle_summaries[b].has_non_part;
                    b_np.cmp(&a_np)
                }
                CollisionSetSortOrder::MinimalPartition => {
                    let a_np = self.collision_sets_cycle_summaries[a].has_non_part;
                    let b_np = self.collision_sets_cycle_summaries[b].has_non_part;

                    a_np.cmp(&b_np).then({
                        let a_minimal_part =
                            &self.collision_sets_cycle_summaries[a].maximal_minimal_cycle;
                        let b_minimal_part =
                            &self.collision_sets_cycle_summaries[b].maximal_minimal_cycle;
                        if let (Some(a_min), Some(b_min)) = (a_minimal_part, b_minimal_part) {
                            ExhaustiveCycleSearchResult::compare_partitions_cycle(a_min, b_min)
                        } else {
                            Ordering::Equal
                        }
                    })
                }
                _ => panic!("Unsupported sort order"),
            };
            if reverse { ord.reverse() } else { ord }
        });

        // originally by https://stackoverflow.com/a/69774341
        // https://play.rust-lang.org/?version=nightly&mode=debug&edition=2021&gist=3c3f3f9ec2628b4a6b23c39f4a65a748
        for i in 0..indices.len() {
            if indices[i] == i {
                continue;
            }
            // found a cycle
            let mut current_idx = i;
            loop {
                let target_idx = indices[current_idx];
                indices[current_idx] = current_idx;
                if indices[target_idx] == target_idx {
                    break;
                }
                self.collision_sets.swap(current_idx, target_idx);
                self.collision_sets_persistence
                    .swap(current_idx, target_idx);
                self.collision_sets_cycle_summaries
                    .swap(current_idx, target_idx);
                for m in self.filter_masks.iter_mut() {
                    m.swap(current_idx, target_idx);
                }
                current_idx = target_idx;
            }
        }
        self.filter_rebuild = true;
    }

    fn compute_collision_sets_summary(&mut self) {
        let n = self.collision_sets.len();
        if n == 0 {
            return;
        }
        //println!("n: {}, max_t: {}", n, rayon::max_num_threads());
        #[cfg(not(target_arch = "wasm32"))]
        {
            let out_res: Mutex<Vec<CycleSearchResSummary>> = Mutex::new(vec![
                    CycleSearchResSummary {
                        has_non_part: false,
                        maximal_minimal_cycle: None
                    };
                    n
                ]);
            let mtx = Mutex::new(0..n);
            let n_thrds = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(1);
            println!("current number of availible threads: {}", n_thrds);
            thread::scope(|s| {
                for _ in 0..n_thrds {
                    s.spawn(|| {
                        println!("current tid: {:?}", thread::current().id());
                        let mut idx;
                        loop {
                            {
                                let res = mtx.lock().unwrap().next();
                                if let Some(res) = res {
                                    idx = res;
                                } else {
                                    return;
                                }
                            }
                            let set = &self.collision_sets[idx];
                            let res = cycle_search(
                                &self.collision_sets_graph,
                                set,
                                self.sweep_dir,
                                self.exclude_common_edges,
                                true,
                            );
                            // res.minimal_trafo();

                            let minimal_part =
                                res.partitionable_pairs
                                    .iter()
                                    .max_by(|(_, _, v), (_, _, w)| {
                                        ExhaustiveCycleSearchResult::compare_partitions(&v, &w)
                                    });
                            let out = CycleSearchResSummary {
                                maximal_minimal_cycle: minimal_part.map(|mp| {
                                    mp.2.iter().map(|v| v.len() as u16).collect::<Vec<u16>>()
                                }),
                                has_non_part: res.has_non_partitionable(),
                            };
                            std::mem::drop(res);
                            out_res.lock().unwrap()[idx] = out;
                        }
                    });
                }
            });
            self.collision_sets_cycle_summaries = out_res.into_inner().unwrap();
            println!("finished compute with threads threads: {}", n_thrds);
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.collision_sets_cycle_summaries = self
                .collision_sets
                .par_iter()
                .map(|set| {
                    let res = cycle_search(
                        &self.collision_sets_graph,
                        set,
                        self.sweep_dir,
                        self.exclude_common_edges,
                        true,
                    );
                    // res.minimal_trafo();

                    let minimal_part =
                        res.partitionable_pairs
                            .iter()
                            .max_by(|(_, _, v), (_, _, w)| {
                                ExhaustiveCycleSearchResult::compare_partitions(&v, &w)
                            });
                    CycleSearchResSummary {
                        maximal_minimal_cycle: minimal_part
                            .map(|mp| mp.2.iter().map(|v| v.len() as u16).collect::<Vec<u16>>()),
                        has_non_part: res.has_non_partitionable(),
                    }
                })
                .collect();
        }
    }

    fn recompute_collision_sets(
        &mut self,
        graph: &Graph,
        ignore_dangling_vertices: bool,
        exclude_common_edges: bool,
        dir: SweepDir,
    ) {
        self.exclude_common_edges = exclude_common_edges;
        self.sweep_dir = dir;
        // TODO: handle too many edges gracefully
        let g_iter = SmallGraphsWithEdgesSet32::new(&graph).unwrap();
        // let start = Instant::now();
        let sets = graph.find_all_remaining_edge_colliding_graphs(
            self.sweep_dir,
            ignore_dangling_vertices,
            g_iter,
        );
        // println!("Time to compute collision sets: {:?}", start.elapsed());
        self.collision_sets = sets;
        // Compute persistence for each graph in each set
        self.collision_sets_persistence = self
            .collision_sets
            .par_iter()
            .map_init(
                || graph.clone(),
                |mod_graph, set| {
                    set.iter()
                        .map(|gv| {
                            // println!("Collision: {:?}", gv);
                            gv.write_to_graph_with_id_vtx(mod_graph);
                            let filtration =
                                SimplexWiseSweepFiltration::from((&*mod_graph, self.sweep_dir));
                            let rev_filtration = SimplexWiseSweepFiltration::from((
                                &*mod_graph,
                                self.sweep_dir.flip(),
                            ));
                            (
                                DirectedPersistence::from(filtration),
                                DirectedPersistence::from(rev_filtration),
                            )
                        })
                        .collect()
                },
            )
            .collect();

        self.collision_sets_graph = graph.clone();

        // TODO: make it configurable whether to exclude common edges
        self.compute_collision_sets_summary();

        for m in self.filter_masks.iter_mut() {
            *m = vec![true; self.collision_sets.len()];
        }
        self.filter_indices = (0..self.collision_sets.len()).collect();
        self.filter_n_comp.clear();
        self.filter_n_cyc.clear();
        self.filter_n_graphs.clear();
        self.filter_non_part = false;
        self.filter_off_diag.clear();
        self.filter_part.clear();
        self.filter_rebuild = false;

        // self.sort_by(CollisionSetSortOrder::Size, false);
        // self.sort_by(
        //     CollisionSetSortOrder::OffDiagonalPoints(self.first_second_homology_ratio),
        //     false,
        // );
    }

    fn draw(&mut self, ui: &mut Ui) -> Option<CycleSearchPanel> {
        let mut c_cycle_summary: Option<CycleSearchPanel> = None;
        ui.horizontal( |ui| {
            ui.vertical(|ui| {
            ui.horizontal( |ui |{
                if ui.button("Help").clicked() {
                self.help_button = true;
                }
            });
            });
            ui.separator();
            if self.help_button {
            egui::Window::new("How to use the filter and sorting features")
            .open(&mut self.help_button)
            .default_size((400., 400.))
            .show(ui.ctx(), |ui| {
                ui.vertical_centered(|ui| {
                ui.heading("How to use the filter and sorting features");
                });
                ui.strong("Filter Range Notation");
                ui.label("You can filter parameters which support it (those with an ordering) by the following \"included range\" notation:\n\
                \ta, [c,d], [e,f], g\n\
                Will match all x such that x=a OR c<=x<=d OR e<=x<=f OR x=g.");
                ui.separator();
                ui.strong("Has Non-Partitionable");
                ui.label("Sorting this, determines if sets which contain non partitionable pairs are shown first or second.\n\
                Checking the checkbox below, restricts the results to those which do contain non partitionable pairs.");
                ui.separator();
                ui.strong("Largest Minimal Partition");
                ui.label("If we have a set of graphs G, E, F which all have the same VPHT, then we can attempt to split their pairwise union graphs \
                into red-blue up-down cycles. During this process we want to split into the shortest cycles possible. \
                If G, E can be split into two cycles of length 4 and 2 respectively, we denote their minimal partition [4, 2]. \
                If for graphs G, E, F their pairwise minimal partitions are [4, 2], [6], [2, 2, 2] for the respective pairs, \
                then their (the set's) largest minimal partition is [6]. \
                We arrive at this simply lexicographically sorting by cycle length and prefering more cycles, \
                that is:\n\t[2, 2] < [2, 2, 2, 2] < [4, 2, 2] < [4, 4] < [4, 4, 2] < [6] < [6, 2] < [6, 2, 2] \n\
                When sorted, the sets will then be sorted by the same ordering.\n\
                The filter window text-box follows filter range notation as described above, where the variables (x, a, b, ect.) must be replaced by minimal partitions. \
                Any sets which contain a pair of graphs matching the filter range will be included in the results");
                ui.separator();
                ui.strong("#Graphs");
                ui.label("Sorts by ascending or descending number of graphs in a partition.\n\
                Follows filter range notation as described above.");
                ui.separator();
                ui.strong("#Components");
                ui.label("Sorts by ascending or descending number of connected components of the graphs in a partition.\n\
                Follows filter range notation as described above.");
                ui.separator();
                ui.strong("#Cycles");
                ui.label("Sorts by ascending or descending number of cycles of the graphs in a partition.\n\
                Follows filter range notation as described above.");
                ui.separator();
                ui.strong("#Off-Diagonal Points");
                ui.label("Sorts by ascending or descending number of off diagonal points in the persistence diagrams. \
                maximum of the off-diagonal points in the forwards and backwards diagram is considered. The Slider determines how
                much zeroeth versus first homology off diagonal points are weighted. All the way towards H₀ and only off diagonal \
                connected components will be considered and vise versa. Lerp in between.\n\
                Follows filter range notation as described above.");
                ui.separator();
            });
            }
            ui.vertical(|ui| { ui.horizontal(|ui | {
            ui.strong("Has Non-Partitionable");
            let reversed = self.is_reversed(CollisionSetSortOrder::HasNonPartitionable);
            if ui.button(if reversed { "⬆" } else { "⬇" }).clicked() {
                self.set_reversed(CollisionSetSortOrder::HasNonPartitionable, !reversed);
                self.sort_by(CollisionSetSortOrder::HasNonPartitionable, !reversed);
            }});
            if ui.checkbox(&mut self.filter_non_part, "").changed() {
                self.filter_rebuild = true;
                let v = &mut self.filter_masks[0];
                for (b, s) in v.iter_mut().zip(self.collision_sets_cycle_summaries.iter()) {
                *b = (s.has_non_part && self.filter_non_part) || !self.filter_non_part;
                }
            }
            });

            // miminum part
            ui.separator();
            let mut temp_str = String::new();
            ui.vertical(|ui| { ui.horizontal(|ui | {
                ui.strong("Largest Minimal Partition");
                let reversed = self.is_reversed(CollisionSetSortOrder::MinimalPartition);
                if ui.button(if reversed { "⬆" } else { "⬇" }).clicked() {
                    self.set_reversed(CollisionSetSortOrder::MinimalPartition, !reversed);
                    self.sort_by(CollisionSetSortOrder::MinimalPartition, !reversed);
                }});
                if !ui.add(egui::TextEdit::singleline(&mut self.filter_part).desired_width(0.).clip_text(false)).changed() {return;}
                if !self.filter_part.is_ascii() {
                    return;
                }
                temp_str.clear();
                temp_str.push('[');
                temp_str.push_str(&self.filter_part);
                temp_str.push(']');
                let mut exact_matches: Vec<Vec<u16>> = Vec::new();
                let mut ranges: Vec<(Vec<u16>, Vec<u16>)> = Vec::new();
                let mut correct = true;
                if let Ok(v) = serde_json::from_str::<Value>(&temp_str) {
                    // valid json
                    println!("parsed text for partitions: {:?}", v);
                    if let Some(a) = v.as_array() {
                    // valid format outer most
                    for v in a.iter() {
                        if let Some(p) = v.as_array() {
                        // each element is either a minimal partitoin, or an interval
                        if p.is_empty() {return;} // can be neither, invalid
                        if p.iter().all(|v| v.is_u64()) {
                            exact_matches.push(p.iter().map(|v: &Value| v.as_u64().unwrap() as u16).collect());
                            continue;
                        }
                        if p.len() == 2 && p.iter().all(|v|
                            v.as_array().and_then(|a|
                            if a.iter().all(|v| v.is_u64()) {Some(0u64)} else {None}
                            ).is_some()) {
                            ranges.push(
                            (p[0].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as u16).collect(),
                            p[1].as_array().unwrap().iter().map(|v|v.as_u64().unwrap() as u16).collect())
                            );
                            continue;
                        }
                        // matched neither of the above patterns, invalid
                        // correct = false;
                        }
                        // did not match one of the continue branches
                        correct = false;
                    }
                    }
                } else {correct = false; }
                println!("is correct: {}, is empty: {}", correct, self.filter_part.is_empty());
                self.filter_rebuild = true;
                let v = &mut self.filter_masks[1];
                if !correct || self.filter_part.is_empty() {
                    for b in v.iter_mut() {
                        *b = true;
                    }
                    return;
                }
                for b in v.iter_mut() {
                    *b = false;
                }
                println!("exact: {:?}, ranges: {:?}", exact_matches, ranges);
                for em in exact_matches {
                    for (b, s) in v.iter_mut().zip(self.collision_sets_cycle_summaries.iter()) {
                    let mimc = &s.maximal_minimal_cycle;
                    if let Some(mimc) = mimc {
                        *b |= mimc.len() == em.len() && mimc.iter().zip(em.iter()).all(|(a,b)| a == b);
                    }
                    }
                }
                for (lo, hi) in ranges.iter() {
                    for (b, s) in v.iter_mut().zip(self.collision_sets_cycle_summaries.iter()) {
                    let mimc = &s.maximal_minimal_cycle;
                    if let Some(mimc) = mimc {
                        *b |= ExhaustiveCycleSearchResult::compare_partitions_cycle(lo, mimc).is_le() &&
                        ExhaustiveCycleSearchResult::compare_partitions_cycle(mimc, hi).is_le();
                    }
                    }
                }
            });

            // Graphs
            ui.separator();
            ui.vertical(|ui| { ui.horizontal(|ui | {
            ui.strong("#Graphs");
            let reversed = self.is_reversed(CollisionSetSortOrder::Size);
            if ui.button(if reversed { "⬆" } else { "⬇" }).clicked() {
                self.set_reversed(CollisionSetSortOrder::Size, !reversed);
                self.sort_by(CollisionSetSortOrder::Size, !reversed);
            }});
            if !ui.add(egui::TextEdit::singleline(&mut self.filter_n_graphs).desired_width(0.).clip_text(false)).changed() {return;}
            if !self.filter_n_graphs.is_ascii() {
                return;
            }
            temp_str.clear();
            temp_str.push('[');
            temp_str.push_str(&self.filter_n_graphs);
            temp_str.push(']');
            let mut correct = true;
            let mut exact_matches: Vec<usize> = Vec::new();
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            if let Ok(v) = serde_json::from_str::<Value>(&temp_str) {
                if let Some(a) = v.as_array() {
                for v in a.iter() {
                    if let Some(n) = v.as_u64() {
                    exact_matches.push(n as usize);
                    continue;
                    }
                    if let Some(arr) = v.as_array() {
                    if arr.len() == 2 && arr.iter().all(|v| v.is_u64()) {
                        ranges.push((arr[0].as_u64().unwrap() as usize, arr[1].as_u64().unwrap() as usize));
                        continue;
                    }
                    }
                    correct = false;
                }
                }
            } else {correct = false;}
            self.filter_rebuild = true;
            let v = &mut self.filter_masks[2];
            if !correct || self.filter_n_graphs.is_empty() {
                for b in v.iter_mut() {
                    *b = true;
                }
                return;
            }

            for b in v.iter_mut() {
                *b = false;
            }
            for em in exact_matches {
                for (b, s) in v.iter_mut().zip(self.collision_sets.iter()) {
                *b |= s.len() == em;
                }
            }
            for (lo, hi) in ranges.iter() {
                for (b, s) in v.iter_mut().zip(self.collision_sets.iter()) {
                *b |= s.len() >= *lo && s.len() <= *hi;
                }
            }
            });

            // Components
            ui.separator();
            ui.vertical(|ui| { ui.horizontal(|ui | {
            ui.strong("#Components");
            let reversed = self.is_reversed(CollisionSetSortOrder::Components);
            if ui.button(if reversed { "⬆" } else { "⬇" }).clicked() {
                self.set_reversed(CollisionSetSortOrder::Components, !reversed);
                self.sort_by(CollisionSetSortOrder::Components, !reversed);
            }});
            if !ui.add(egui::TextEdit::singleline(&mut self.filter_n_comp).desired_width(0.).clip_text(false)).changed() {return;}
            if !self.filter_n_comp.is_ascii() {
                return;
            }
            temp_str.clear();
            temp_str.push('[');
            temp_str.push_str(&self.filter_n_comp);
            temp_str.push(']');
            let mut correct = true;
            let mut exact_matches: Vec<u32> = Vec::new();
            let mut ranges: Vec<(u32, u32)> = Vec::new();
            if let Ok(v) = serde_json::from_str::<Value>(&temp_str) {
                if let Some(a) = v.as_array() {
                for v in a.iter() {
                    if let Some(n) = v.as_u64() {
                    exact_matches.push(n as u32);
                    continue;
                    }
                    if let Some(arr) = v.as_array() {
                    if arr.len() == 2 && arr.iter().all(|v| v.is_u64()) {
                        ranges.push((arr[0].as_u64().unwrap() as u32, arr[1].as_u64().unwrap() as u32));
                        continue;
                    }
                    }
                    correct = false;
                }
                }
            } else {correct = false;}
            self.filter_rebuild = true;
            let v = &mut self.filter_masks[3];
            if !correct || self.filter_n_comp.is_empty() {
                for b in v.iter_mut() {
                    *b = true;
                }
                return;
            }
            for b in v.iter_mut() {
                *b = false;
            }
            for em in exact_matches {
                for (b, s) in v.iter_mut().zip(self.collision_sets_persistence.iter()) {
                let cc = s.first().map_or(0, |(fwd, _)| {
                    fwd.connected_comp.points.iter().map(|p| ((p.y == f64::INFINITY) as u32) * p.mult).sum::<u32>()
                });
                *b |= cc == em;
                }
            }
            for (lo, hi) in ranges.iter() {
                for (b, s) in v.iter_mut().zip(self.collision_sets_persistence.iter()) {
                let cc = s.first().map_or(0, |(fwd, _)| {
                    fwd.connected_comp.points.iter().map(|p| ((p.y == f64::INFINITY) as u32) * p.mult).sum::<u32>()
                });
                *b |= cc >= *lo && cc <= *hi;
                }
            }
            });

            //cycles
            ui.separator();
            ui.vertical(|ui| { ui.horizontal(|ui | {
            ui.strong("#Cycles");
            let reversed = self.is_reversed(CollisionSetSortOrder::Cycles);
            if ui.button(if reversed { "⬆" } else { "⬇" }).clicked() {
                self.set_reversed(CollisionSetSortOrder::Cycles, !reversed);
                self.sort_by(CollisionSetSortOrder::Cycles, !reversed);
            }});
            if !ui.add(egui::TextEdit::singleline(&mut self.filter_n_cyc).desired_width(0.).clip_text(false)).changed() {return;}
            if !self.filter_n_cyc.is_ascii() {
                return;
            }
            temp_str.clear();
            temp_str.push('[');
            temp_str.push_str(&self.filter_n_cyc);
            temp_str.push(']');
            let mut correct = true;
            let mut exact_matches: Vec<u32> = Vec::new();
            let mut ranges: Vec<(u32, u32)> = Vec::new();
            if let Ok(v) = serde_json::from_str::<Value>(&temp_str) {
                if let Some(a) = v.as_array() {
                for v in a.iter() {
                    if let Some(n) = v.as_u64() {
                    exact_matches.push(n as u32);
                    continue;
                    }
                    if let Some(arr) = v.as_array() {
                    if arr.len() == 2 && arr.iter().all(|v| v.is_u64()) {
                        ranges.push((arr[0].as_u64().unwrap() as u32, arr[1].as_u64().unwrap() as u32));
                        continue;
                    }
                    }
                    correct = false;
                }
                }
            } else {correct = false;}
            self.filter_rebuild = true;
            let v = &mut self.filter_masks[4];
            if !correct || self.filter_n_cyc.is_empty() {
                for b in v.iter_mut() {
                    *b = true;
                }
                return;
            }
            for b in v.iter_mut() {
                *b = false;
            }
            for em in exact_matches {
                for (b, s) in v.iter_mut().zip(self.collision_sets_persistence.iter()) {
                let cyc = s.first().map_or(0, |(fwd, _)| {
                    fwd.cycles.points.iter().map(|p| ((p.y == f64::INFINITY) as u32) * p.mult).sum::<u32>()
                });
                *b |= cyc == em;
                }
            }
            for (lo, hi) in ranges.iter() {
                for (b, s) in v.iter_mut().zip(self.collision_sets_persistence.iter()) {
                let cyc = s.first().map_or(0, |(fwd, _)| {
                    fwd.cycles.points.iter().map(|p| ((p.y == f64::INFINITY) as u32) * p.mult).sum::<u32>()
                });
                *b |= cyc >= *lo && cyc <= *hi;
                }
            }
            });

            // off diag points
            ui.separator();
            ui.vertical(|ui| { ui.horizontal(|ui | {
            ui.strong("#Off-Diagonal Points");
            ui.add(egui::Slider::new(&mut self.first_second_homology_ratio, 0.0..=1.0).text("H₀ vs H₁ weight"));
            let reversed = self.is_reversed(CollisionSetSortOrder::OffDiagonalPoints(self.first_second_homology_ratio));
            if ui.button(if reversed { "⬆" } else { "⬇" }).clicked() {
                self.set_reversed(CollisionSetSortOrder::OffDiagonalPoints(self.first_second_homology_ratio), !reversed);
                self.sort_by(CollisionSetSortOrder::OffDiagonalPoints(self.first_second_homology_ratio), !reversed);
            }});
            if !ui.add(egui::TextEdit::singleline(&mut self.filter_off_diag).desired_width(0.).clip_text(false)).changed() {return;}
            if !self.filter_off_diag.is_ascii() {
                return;
            }
            temp_str.clear();
            temp_str.push('[');
            temp_str.push_str(&self.filter_off_diag);
            temp_str.push(']');
            let mut correct = true;
            let mut exact_matches: Vec<f64> = Vec::new();
            let mut ranges: Vec<(f64, f64)> = Vec::new();
            if let Ok(v) = serde_json::from_str::<Value>(&temp_str) {
                if let Some(a) = v.as_array() {
                for v in a.iter() {
                    if let Some(n) = v.as_f64() {
                    exact_matches.push(n);
                    continue;
                    }
                    if let Some(arr) = v.as_array() {
                    if arr.len() == 2 && arr.iter().all(|v| v.is_f64()) {
                        ranges.push((arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap()));
                        continue;
                    }
                    }
                    correct = false;
                }
                }
            } else {correct = false;}

            self.filter_rebuild = true;
            let v = &mut self.filter_masks[5];
            if !correct || self.filter_off_diag.is_empty() {
                for b in v.iter_mut() {
                    *b = true;
                }
                return;
            }
            for b in v.iter_mut() {
                *b = false;
            }
            let x = self.first_second_homology_ratio;
            for em in exact_matches {
                for (b, s) in v.iter_mut().zip(self.collision_sets_persistence.iter()) {
                let (a_cc, a_cyc) = s.first().map_or((0., 0.), |(fwd, bkw)| {
                    ((fwd.connected_comp.points.iter().map(|p| !p.is_diagonal() as u32).sum::<u32>() as f64)
                    .max(bkw.connected_comp.points.iter().map(|p| !p.is_diagonal() as u32).sum::<u32>() as f64),
                     (fwd.cycles.points.iter().map(|p| !p.is_diagonal() as u32).sum::<u32>() as f64)
                    .max(bkw.cycles.points.iter().map(|p| !p.is_diagonal() as u32).sum::<u32>() as f64))
                });
                let val = a_cyc * x + a_cc * (1. - x);
                *b |= (val - em).abs() < 1e-6;
                }
            }
            for (lo, hi) in ranges.iter() {
                for (b, s) in v.iter_mut().zip(self.collision_sets_persistence.iter()) {
                let (a_cc, a_cyc) = s.first().map_or((0., 0.), |(fwd, bkw)| {
                    ((fwd.connected_comp.points.iter().map(|p| !p.is_diagonal() as u32).sum::<u32>() as f64)
                    .max(bkw.connected_comp.points.iter().map(|p| !p.is_diagonal() as u32).sum::<u32>() as f64),
                     (fwd.cycles.points.iter().map(|p| !p.is_diagonal() as u32).sum::<u32>() as f64)
                    .max(bkw.cycles.points.iter().map(|p| !p.is_diagonal() as u32).sum::<u32>() as f64))
                });
                let val = a_cyc * x + a_cc * (1. - x);
                *b |= val >= *lo && val <= *hi;
                }
            }
            });
        });

        // rebuild the filter
        if self.filter_rebuild {
            self.filter_rebuild = false;
            self.filter_indices = (0..self.collision_sets.len())
                .filter(|idx| self.filter_masks.iter().all(|m| m[*idx]))
                .collect();
            self.selected_set = None;
        }
        ui.separator();
        ui.with_layout(
            egui::Layout::left_to_right(egui::Align::Center).with_cross_justify(true),
            |ui| {
                ui.with_layout(
                    egui::Layout::top_down(egui::Align::Min).with_cross_justify(true),
                    |ui| {
                        ui.set_width(ui.available_width() / 5.);
                        if self.filter_indices.is_empty() {
                            self.selected_set = None;
                            ui.centered_and_justified(|ui| {
                                egui::Frame::group(ui.style()).show(ui, |ui| {
                                    ui.set_width(ui.available_width());
                                    ui.set_height(ui.available_height());
                                    ui.heading("No collision sets availible");
                                });
                            });
                            return;
                        } else {
                            egui::ScrollArea::vertical()
                                .max_width(ui.available_width())
                                .show_rows(ui, 50., self.filter_indices.len(), |ui, vis_rows| {
                                    for row_idx in vis_rows {
                                        let set_idx = self.filter_indices[row_idx];
                                        let collision_set = &self.collision_sets[set_idx];
                                        // if set_idx >= self.selected_collision_set_graphs.len() {
                                        //     self.selected_collision_set_graphs
                                        //         .resize(set_idx + 1, vec![true; collision_set.len()]);
                                        // }
                                        // for graph_idx in 0..collision_set.len() {
                                        //     if graph_idx >= self.selected_collision_set_graphs[set_idx].len() {
                                        //         self.selected_collision_set_graphs[set_idx]
                                        //             .resize(graph_idx + 1, true);
                                        //     }
                                        // }

                                        let minimal_part = &self.collision_sets_cycle_summaries
                                            [set_idx]
                                            .maximal_minimal_cycle;
                                        let minimal_partition = if let Some(mp) = minimal_part {
                                            format!("min. part. cycles: {:?}", mp)
                                        } else {
                                            String::from("no partitionable pairs")
                                        };

                                        let fr = egui::Frame::group(ui.style());
                                        fr.show(ui, |ui| {
                                            let mar = fr.total_margin();
                                            ui.set_height(50. - mar.top - mar.bottom);
                                            ui.set_width(ui.available_width());
                                            ui.vertical(|ui| {
                                                ui.label(format!(
                                                    "Set {} ({} graphs, {})",
                                                    set_idx + 1,
                                                    collision_set.len(),
                                                    minimal_partition
                                                ));
                                                if ui.button("Show").clicked() {
                                                    self.selected_set = Some(set_idx);
                                                }
                                            });
                                        });
                                    }
                                });
                        }
                    },
                );
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    self.draw_selected_graphs(&mut c_cycle_summary, self.selected_set, ui);
                });
            },
        );
        c_cycle_summary
    }

    fn draw_selected_graphs(
        &self,
        c_cycle_summary: &mut Option<CycleSearchPanel>,
        set_idx: Option<usize>,
        ui: &mut Ui,
    ) {
        if self.collision_sets.is_empty() || set_idx.is_none() {
            ui.centered_and_justified(|ui| {
                ui.heading("No collision sets selected");
            });
            return;
        }
        let set_idx = set_idx
            .expect("you should have an is_none check before this which you may have removed");
        let mut mod_graph = self.collision_sets_graph.clone();
        let mut display_cycle_search = false;
        draw_collision_set(
            ui,
            &self.collision_sets[set_idx],
            &self.collision_sets_persistence[set_idx],
            &mut mod_graph,
            self.collision_sets_cycle_summaries[set_idx].has_non_part,
            &mut display_cycle_search,
        );
        if display_cycle_search {
            *c_cycle_summary = Some(CycleSearchPanel::new(
                &self.collision_sets[set_idx],
                !self.exclude_common_edges,
                &self.collision_sets_graph,
                self.sweep_dir,
            ))
        }
    }
}

#[derive(Default)]
struct CollidingGraphsPanel {
    sweep_dir: SweepDir,
    base_graph: Graph,
    collision_set: Vec<SmallGraphView32>,
    cycle_search_res: ExhaustiveCycleSearchResult,
    include_common_edges: bool,
    help_button: bool,
    persistence: Vec<(DirectedPersistence, DirectedPersistence)>,
}

impl CollidingGraphsPanel {
    fn new() -> Self {
        Self::default()
    }

    fn draw(&self, ui: &mut Ui, c_cycle_summary: &mut Option<CycleSearchPanel>) {
        let mut mod_graph = self.base_graph.clone();
        let mut display_cycle_search = false;
        // TODO: add help button
        draw_collision_set(
            ui,
            &self.collision_set,
            &self.persistence,
            &mut mod_graph,
            self.cycle_search_res.has_non_partitionable(),
            &mut display_cycle_search,
        );
        if display_cycle_search {
            *c_cycle_summary = Some(CycleSearchPanel::new(
                &self.collision_set,
                self.include_common_edges,
                &self.base_graph,
                self.sweep_dir,
            ))
        }
    }

    fn recompute(
        &mut self,
        graph: &Graph,
        ignore_dangling_vertices: bool,
        exclude_common_edges: bool,
        sweep_dir: SweepDir,
    ) {
        self.include_common_edges = !exclude_common_edges;
        // TODO: handle too many vtx gracefully
        let g_iter = AllSmallGraphs32::new(graph).unwrap();
        self.base_graph = graph.clone();
        self.sweep_dir = sweep_dir;
        let collisions =
            graph.find_colliding_graphs(self.sweep_dir, g_iter, ignore_dangling_vertices);
        self.collision_set = collisions;
        // Compute persistence for each colliding graph
        self.persistence = self
            .collision_set
            .par_iter()
            .map_init(
                || self.base_graph.clone(),
                |mod_graph, gv| {
                    // println!("Collision: {:?}", gv);
                    gv.write_to_graph_with_id_vtx(mod_graph);
                    let filtration =
                        SimplexWiseSweepFiltration::from((&*mod_graph, self.sweep_dir));
                    let rev_filtration =
                        SimplexWiseSweepFiltration::from((&*mod_graph, self.sweep_dir.flip()));
                    (
                        DirectedPersistence::from(filtration),
                        DirectedPersistence::from(rev_filtration),
                    )
                },
            )
            .collect();

        self.cycle_search_res = exhaustive_cycle_search(
            &self.base_graph,
            &self.collision_set,
            self.sweep_dir,
            exclude_common_edges,
        )
    }
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct TemplateApp {
    #[serde(skip)]
    graph: Graph,

    sweep_dir: SweepDir,

    #[serde(skip)]
    persistence: Option<DirectedPersistence>,

    #[serde(skip)]
    selected_vertex: Option<usize>,

    #[serde(skip)]
    edge_start: Option<usize>,

    mode: InteractionMode,

    #[serde(skip)]
    colliding_graphs_data: CollidingGraphsPanel,

    #[serde(skip)]
    collision_sets_data: CollisionSetsPanel,

    show_collisions_panel: bool,
    show_collision_sets_panel: bool,

    ignore_dangling_vertices: bool,
    exclude_common_edges: bool,

    #[serde(skip)]
    selected_collisions: Vec<bool>,

    show_cycle_search: bool,
    #[serde(skip)]
    cycle_search_panel: CycleSearchPanel,
}

#[derive(serde::Deserialize, serde::Serialize, PartialEq, Clone, Copy)]
enum InteractionMode {
    AddVertex,
    AddEdge,
    DeleteEdge,
    DeleteVertex,
    SetSweepDir,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            graph: Graph::new(0),
            sweep_dir: SweepDir::new(1.0, 0.0),
            persistence: None,
            selected_vertex: None,
            edge_start: None,
            mode: InteractionMode::AddVertex,
            colliding_graphs_data: CollidingGraphsPanel::new(),
            collision_sets_data: CollisionSetsPanel::new(),
            show_collisions_panel: false,
            show_collision_sets_panel: false,
            selected_collisions: Vec::new(),
            ignore_dangling_vertices: true,
            exclude_common_edges: true,
            show_cycle_search: false,
            cycle_search_panel: CycleSearchPanel::default(),
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            let mut app: Self = eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            app.recompute_persistence();
            app
        } else {
            Default::default()
        }
    }

    fn recompute_persistence(&mut self) {
        if self.graph.vertices.is_empty() {
            self.persistence = None;
            return;
        }

        let filtration = SimplexWiseSweepFiltration::from((&self.graph, self.sweep_dir));
        self.persistence = Some(DirectedPersistence::from(filtration));
    }

    fn find_vertex_at(&self, screen_pos: Pos2, canvas_rect: Rect, grid_size: f32) -> Option<usize> {
        let threshold = 10.0;

        for (idx, vertex) in self.graph.vertices.iter().enumerate() {
            let vtx_pos = world_to_screen(
                Pos2::new(vertex.x as f32, vertex.y as f32),
                canvas_rect,
                grid_size,
            );

            if vtx_pos.distance(screen_pos) < threshold {
                return Some(idx);
            }
        }
        None
    }

    fn find_edge_at(&self, screen_pos: Pos2, canvas_rect: Rect, grid_size: f32) -> Option<Edge> {
        // TODO: select edges not vertices...
        let threshold = 10.0;

        for i in 0..self.graph.vertices.len() {
            for j in (i + 1)..self.graph.vertices.len() {
                if self.graph.has_edge(Edge::new(i, j)) {
                    let v1 = &self.graph.vertices[i];
                    let v2 = &self.graph.vertices[j];

                    let p1 = world_to_screen(
                        Pos2::new(v1.x as f32, v1.y as f32),
                        canvas_rect,
                        grid_size,
                    );
                    let p2 = world_to_screen(
                        Pos2::new(v2.x as f32, v2.y as f32),
                        canvas_rect,
                        grid_size,
                    );

                    // Calculate distance from point to line segment
                    let line_vec = p2 - p1;
                    let point_vec = screen_pos - p1;
                    let line_len = line_vec.length();

                    if line_len > 0.0 {
                        let t = (point_vec.dot(line_vec) / (line_len * line_len)).clamp(0.0, 1.0);
                        let projection = p1 + line_vec * t;
                        let dist = screen_pos.distance(projection);

                        if dist < threshold {
                            return Some(Edge::new(i, j));
                        }
                    }
                }
            }
        }
        None
    }

    fn draw_main_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:

            egui::MenuBar::new().ui(ui, |ui| {
                egui::widgets::global_theme_preference_buttons(ui);
                ui.add_space(16.0);
                if ui.button("Clear Graph").clicked() {
                    self.graph = Graph::new(0);
                    self.persistence = None;
                    self.selected_vertex = None;
                    self.edge_start = None;
                }
            });
        });
    }

    fn draw_control_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");

        ui.separator();
        self.select_graph_interaction_mode(ui);

        ui.separator();
        ui.label("Sweep Direction:");

        self.update_sweep_direction(ui);

        ui.separator();
        ui.label(format!("Vertices: {}", self.graph.vertices.len()));

        let edge_count = self.graph.edges.iter().filter(|&&e| e).count() / 2;
        ui.label(format!("Edges: {}", edge_count));

        if ui.button("Recompute Persistence").clicked() {
            self.recompute_persistence();
        }

        // --- SECTION: find graphs that collide with the current graph ---
        ui.separator();

        self.draw_collision_controls(ui);
    }

    fn draw_collision_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Collision Detection");
        ui.checkbox(
            &mut self.ignore_dangling_vertices,
            "Ignore Dangling Vertices",
        );
        if ui
            .checkbox(
                &mut self.exclude_common_edges,
                "Exclude Common Edges from Cycle Search",
            )
            .changed()
        {
            // TODO: dedupe code
            // self.collision_sets_data.compute_collision_sets_summary();
        };
        ui.separator();

        self.draw_colliding_graphs_controls(ui);
        ui.separator();

        // --- SECTION: find graphs with the current edges and vertices that collide amongst themselves ---
        self.draw_collision_sets_controls(ui);
    }

    fn draw_collision_sets_controls(&mut self, ui: &mut egui::Ui) {
        if ui.button("Compute Collision Sets").clicked() {
            self.collision_sets_data.recompute_collision_sets(
                &self.graph,
                self.ignore_dangling_vertices,
                self.exclude_common_edges,
                self.sweep_dir,
            );
            self.show_collision_sets_panel = true;
        }

        if ui.button("Show Colliding Sets").clicked() {
            self.show_collision_sets_panel = true;
        }

        ui.label(format!(
            "Collision sets found: {}",
            self.collision_sets_data.collision_sets.len()
        ));
    }

    fn draw_colliding_graphs_controls(&mut self, ui: &mut egui::Ui) {
        if ui.button("Compute Colliding Graphs").clicked() {
            self.colliding_graphs_data.recompute(
                &self.graph,
                self.ignore_dangling_vertices,
                self.exclude_common_edges,
                self.sweep_dir,
            );
            self.show_collisions_panel = true;
        }

        if ui.button("Show Colliding Graphs").clicked() {
            self.show_collisions_panel = true;
        }
        ui.label(format!(
            "Collisions found: {}",
            self.colliding_graphs_data.collision_set.len()
        ));
    }

    fn update_sweep_direction(&mut self, ui: &mut egui::Ui) {
        let dir_x = self.sweep_dir.x as f32;
        let dir_y = self.sweep_dir.y as f32;
        let mut changed = false;
        let mut theta = dir_y.atan2(dir_x).to_degrees();

        ui.horizontal(|ui| {
            ui.label("Angle:");
            changed |= ui
                .add(egui::DragValue::new(&mut theta).speed(0.5))
                .changed();
        });
        theta = theta.to_radians();

        let (dir_x, dir_y) = (theta.cos(), theta.sin());
        if changed {
            self.sweep_dir = SweepDir::new(dir_x as f64, dir_y as f64);
            self.recompute_persistence();
        }

        if ui.button("Flip Direction").clicked() {
            self.sweep_dir = self.sweep_dir.flip();
            self.recompute_persistence();
        }
    }

    fn select_graph_interaction_mode(&mut self, ui: &mut egui::Ui) {
        ui.label("Mode:");
        ui.radio_value(&mut self.mode, InteractionMode::AddVertex, "Add Vertex");
        ui.radio_value(&mut self.mode, InteractionMode::AddEdge, "Add Edge");
        ui.radio_value(&mut self.mode, InteractionMode::DeleteEdge, "Delete Edge");
        ui.radio_value(
            &mut self.mode,
            InteractionMode::DeleteVertex,
            "Delete Vertex",
        );
        ui.radio_value(
            &mut self.mode,
            InteractionMode::SetSweepDir,
            "Set Sweep Direction",
        );
    }

    fn draw_collision_sets_window(&mut self, ctx: &egui::Context) {
        let mut open = self.show_collision_sets_panel;
        let mut cycle_panel = None;
        egui::Window::new("Collision Sets")
            .open(&mut open)
            .default_size([500.0, 600.0])
            .min_width(500.0)
            //.max_width(1000.)
            .show(ctx, |ui| {
                cycle_panel = self.collision_sets_data.draw(ui);
            });
        self.show_collision_sets_panel = open;
        if let Some(cycle_panel) = cycle_panel {
            self.cycle_search_panel = cycle_panel;
            self.show_cycle_search = true;
        }
    }
}

impl eframe::App for TemplateApp {
    /// Called by the framework to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    /// Called each time the UI needs repainting, which may be many times per second.
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Put your widgets into a `SidePanel`, `TopBottomPanel`, `CentralPanel`, `Window` or `Area`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        self.draw_main_top_bar(ctx);

        egui::SidePanel::left("control_panel")
            .min_width(200.0)
            .show(ctx, |ui| {
                self.draw_control_panel(ui);
            });

        // --- SECTION: current graph pd:
        // TODO: add reverse
        egui::SidePanel::right("diagram_panel")
            .min_width(300.0)
            .show(ctx, |ui| {
                ui.heading("Persistence Diagrams");

                if let Some(ref pers) = self.persistence {
                    ui.separator();
                    ui.label("Connected Components (H₀):");
                    self.draw_persistence_diagram(ui, &pers.connected_comp, Color32::BLUE);

                    ui.separator();
                    ui.label("Cycles (H₁):");
                    self.draw_persistence_diagram(ui, &pers.cycles, Color32::RED);
                } else {
                    ui.label("No persistence data. Add vertices to the graph.");
                }
            });

        // --- current graph view in center
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Graph Editor");

            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::click_and_drag());

            let canvas_rect = response.rect;
            let grid_size = 40.0;

            // Draw grid
            self.draw_grid(&painter, canvas_rect, grid_size);

            // Draw edges
            for i in 0..self.graph.vertices.len() {
                for j in (i + 1)..self.graph.vertices.len() {
                    if self.graph.has_edge(Edge::new(i, j)) {
                        let v1 = &self.graph.vertices[i];
                        let v2 = &self.graph.vertices[j];

                        let p1 = world_to_screen(
                            Pos2::new(v1.x as f32, v1.y as f32),
                            canvas_rect,
                            grid_size,
                        );
                        let p2 = world_to_screen(
                            Pos2::new(v2.x as f32, v2.y as f32),
                            canvas_rect,
                            grid_size,
                        );
                        arc_edge_painter(p1, p2, false, Stroke::new(2.0, Color32::GRAY), &painter);
                        // painter.line_segment([p1, p2], Stroke::new(2.0, Color32::GRAY));
                    }
                }
            }

            // Draw sweep direction
            let center = canvas_rect.center();
            let sweep_end = Pos2::new(
                center.x + self.sweep_dir.x as f32 * 50.0,
                center.y - self.sweep_dir.y as f32 * 50.0,
            );
            painter.arrow(center, sweep_end - center, Stroke::new(2.0, Color32::GREEN));

            // Draw vertices
            for (idx, vertex) in self.graph.vertices.iter().enumerate() {
                let pos = world_to_screen(
                    Pos2::new(vertex.x as f32, vertex.y as f32),
                    canvas_rect,
                    grid_size,
                );

                let color = if Some(idx) == self.selected_vertex || Some(idx) == self.edge_start {
                    Color32::YELLOW
                } else {
                    Color32::WHITE
                };

                painter.circle_filled(pos, 6.0, color);
                painter.circle_stroke(pos, 6.0, Stroke::new(2.0, Color32::BLACK));

                painter.text(
                    pos + Vec2::new(10.0, -10.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("{:.2}", self.sweep_dir.height(vertex)),
                    egui::FontId::default(),
                    Color32::BLACK,
                );
            }

            // Handle interactions
            if response.clicked() {
                if let Some(interact_pos) = response.interact_pointer_pos() {
                    match self.mode {
                        InteractionMode::AddVertex => {
                            let world_pos = screen_to_world(interact_pos, canvas_rect, grid_size);

                            let new_idx = self.graph.vertices.len();
                            let mut new_graph = Graph::new(new_idx + 1);

                            // Copy existing vertices and edges
                            for (i, v) in self.graph.vertices.iter().enumerate() {
                                new_graph.vertices.push(*v);
                                for j in 0..self.graph.vertices.len() {
                                    if self.graph.has_edge(Edge::new(i, j)) {
                                        new_graph.add_edge(Edge::new(i, j));
                                    }
                                }
                            }

                            new_graph.vertices.push(Vertex {
                                x: world_pos.x as f64,
                                y: world_pos.y as f64,
                            });

                            self.graph = new_graph;
                            self.recompute_persistence();
                        }
                        InteractionMode::AddEdge => {
                            if let Some(clicked_vtx) =
                                self.find_vertex_at(interact_pos, canvas_rect, grid_size)
                            {
                                if let Some(start) = self.edge_start {
                                    if start != clicked_vtx {
                                        self.graph.add_edge(Edge::new(start, clicked_vtx));
                                        self.recompute_persistence();
                                    }
                                    self.edge_start = None;
                                } else {
                                    self.edge_start = Some(clicked_vtx);
                                }
                            } else {
                                self.edge_start = None;
                            }
                        }
                        InteractionMode::DeleteEdge => {
                            if let Some(edge) =
                                self.find_edge_at(interact_pos, canvas_rect, grid_size)
                            {
                                self.graph.remove_edge(edge);
                                self.recompute_persistence();
                            }
                        }
                        InteractionMode::SetSweepDir => {
                            let center = canvas_rect.center();
                            let dir = interact_pos - center;
                            self.sweep_dir = SweepDir::new(dir.x as f64, -dir.y as f64);
                            self.recompute_persistence();
                        }
                        InteractionMode::DeleteVertex => {
                            if let Some(vtx_to_delete) =
                                self.find_vertex_at(interact_pos, canvas_rect, grid_size)
                            {
                                let new_n = self.graph.vertices.len() - 1;
                                let mut new_graph = Graph::new(new_n);

                                for (i, v) in self.graph.vertices.iter().enumerate() {
                                    if i == vtx_to_delete {
                                        continue;
                                    }
                                    let new_i = if i > vtx_to_delete { i - 1 } else { i };
                                    new_graph.vertices.push(*v);

                                    for j in 0..self.graph.vertices.len() {
                                        if j == vtx_to_delete {
                                            continue;
                                        }
                                        let new_j = if j > vtx_to_delete { j - 1 } else { j };

                                        if self.graph.has_edge(Edge::new(i, j)) {
                                            new_graph.add_edge(Edge::new(new_i, new_j));
                                        }
                                    }
                                }

                                self.graph = new_graph;
                                self.selected_vertex = None;
                                self.edge_start = None;
                                self.recompute_persistence();
                            }
                        }
                    }
                }
            }

            if response.hovered() {
                if let Some(hover_pos) = response.hover_pos() {
                    if let Some(idx) = self.find_vertex_at(hover_pos, canvas_rect, grid_size) {
                        self.selected_vertex = Some(idx);
                    } else {
                        if self.mode != InteractionMode::AddEdge {
                            self.selected_vertex = None;
                        }
                    }
                }
            }
        });

        // Colliding graphs window
        if self.show_collisions_panel {
            let mut mod_panel = None;
            let mut open = self.show_collisions_panel;
            egui::Window::new("Colliding Graphs")
                .open(&mut open)
                .movable(true)
                .default_size([800.0, 600.0])
                .min_width(500.0)
                .show(ctx, |ui| {
                    self.colliding_graphs_data.draw(ui, &mut mod_panel);
                });
            self.show_collisions_panel = open;
            if let Some(panel) = mod_panel {
                self.cycle_search_panel = panel;
                self.show_cycle_search = true;
            }
        }

        // Collision sets panel
        if self.show_collision_sets_panel {
            self.draw_collision_sets_window(ctx);
        }
        let mut open = self.show_cycle_search;
        if open {
            egui::Window::new("Cycle Search in Collision Set")
                .open(&mut open)
                .default_size([600.0, 400.0])
                .min_width(400.0)
                .show(ctx, |ui| self.cycle_search_panel.draw(ui));
            self.show_cycle_search = open;
        }
    }
}

#[derive(Clone, Copy)]
enum CollisionSetSortOrder {
    Size,
    Components,
    Cycles,
    OffDiagonalPoints(f64),
    HasNonPartitionable,
    MinimalPartition,
}

impl TemplateApp {
    fn draw_grid(&self, painter: &egui::Painter, rect: Rect, grid_size: f32) {
        let center = rect.center();
        let light_gray = Color32::from_gray(200);
        let gray = Color32::from_gray(150);

        // Vertical lines
        let mut x = center.x;
        while x < rect.right() {
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(
                    if (x - center.x).abs() < 1.0 { 1.0 } else { 0.5 },
                    if (x - center.x).abs() < 1.0 {
                        gray
                    } else {
                        light_gray
                    },
                ),
            );
            x += grid_size;
        }

        x = center.x - grid_size;
        while x > rect.left() {
            painter.line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(0.5, light_gray),
            );
            x -= grid_size;
        }

        // Horizontal lines
        let mut y = center.y;
        while y < rect.bottom() {
            painter.line_segment(
                [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                Stroke::new(
                    if (y - center.y).abs() < 1.0 { 1.0 } else { 0.5 },
                    if (y - center.y).abs() < 1.0 {
                        gray
                    } else {
                        light_gray
                    },
                ),
            );
            y += grid_size;
        }

        y = center.y - grid_size;
        while y > rect.top() {
            painter.line_segment(
                [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                Stroke::new(0.5, light_gray),
            );
            y -= grid_size;
        }
    }

    fn draw_persistence_diagram(
        &self,
        ui: &mut egui::Ui,
        diagram: &crate::graph::PersistenceDiagram,
        color: Color32,
    ) {
        let (response, painter) =
            ui.allocate_painter(Vec2::new(ui.available_width(), 200.0), egui::Sense::hover());

        let rect = response.rect;

        if diagram.points.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No points",
                egui::FontId::default(),
                Color32::GRAY,
            );
            return;
        }

        // Find bounds for finite points
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for pt in diagram.points.iter() {
            min_val = min_val.min(pt.x);
            max_val = if pt.y.is_infinite() {
                max_val.max(pt.x)
            } else {
                max_val.max(pt.y)
            };
        }

        // Separate finite and infinite points
        let finite_points: Vec<_> = diagram.points.iter().filter(|p| p.y.is_finite()).collect();
        let infinite_points: Vec<_> = diagram
            .points
            .iter()
            .filter(|p| p.y.is_infinite())
            .collect();

        // If we still don't have a valid upper bound (only points at infinity)
        if max_val == min_val {
            max_val = min_val + 1.0;
        }

        let margin = 20.0;
        let plot_rect = rect.shrink(margin);

        // Draw diagonal
        painter.line_segment(
            [plot_rect.left_bottom(), plot_rect.right_top()],
            Stroke::new(1.0, Color32::from_gray(180)),
        );

        // Draw axes
        painter.line_segment(
            [
                Pos2::new(plot_rect.left(), plot_rect.bottom()),
                plot_rect.right_bottom(),
            ],
            Stroke::new(1.0, Color32::BLACK),
        );
        painter.line_segment(
            [
                Pos2::new(plot_rect.left(), plot_rect.bottom()),
                plot_rect.left_top(),
            ],
            Stroke::new(1.0, Color32::BLACK),
        );

        // Draw finite points
        for point in finite_points {
            let x_norm = ((point.x - min_val) / (max_val - min_val)) as f32;
            let y_norm = ((point.y - min_val) / (max_val - min_val)) as f32;

            let screen_pos = Pos2::new(
                plot_rect.left() + x_norm * plot_rect.width(),
                plot_rect.bottom() - y_norm * plot_rect.height(),
            );

            let radius = 3.0 + (point.mult as f32).sqrt();
            painter.circle_filled(screen_pos, radius, color);
            painter.circle_stroke(screen_pos, radius, Stroke::new(1.0, Color32::BLACK));
        }

        // Draw infinite points at the top of the diagram
        for point in &infinite_points {
            let x_norm = ((point.x - min_val) / (max_val - min_val)) as f32;

            let screen_pos = Pos2::new(
                plot_rect.left() + x_norm * plot_rect.width(),
                plot_rect.top(),
            );

            let radius = 3.0 + (point.mult as f32).sqrt();

            // Draw with a different style to indicate infinity
            painter.circle_filled(screen_pos, radius, color);
            painter.circle_stroke(screen_pos, radius, Stroke::new(2.0, Color32::BLACK));

            // Draw a small upward arrow or symbol to indicate infinity
            let arrow_start = screen_pos + Vec2::new(0.0, -radius - 2.0);
            let arrow_end = arrow_start + Vec2::new(0.0, -8.0);
            painter.line_segment([arrow_start, arrow_end], Stroke::new(1.5, Color32::BLACK));
            painter.line_segment(
                [arrow_end, arrow_end + Vec2::new(-3.0, 3.0)],
                Stroke::new(1.5, Color32::BLACK),
            );
            painter.line_segment(
                [arrow_end, arrow_end + Vec2::new(3.0, 3.0)],
                Stroke::new(1.5, Color32::BLACK),
            );
        }

        // Labels
        painter.text(
            Pos2::new(rect.center().x, rect.bottom() - 5.0),
            egui::Align2::CENTER_BOTTOM,
            "Birth",
            egui::FontId::default(),
            Color32::BLACK,
        );

        painter.text(
            Pos2::new(plot_rect.right(), plot_rect.bottom() - 1.0),
            egui::Align2::CENTER_TOP,
            format!("{:.2}", max_val),
            egui::FontId::default(),
            Color32::BLACK,
        );

        painter.text(
            Pos2::new(plot_rect.left(), plot_rect.bottom() - 1.0),
            egui::Align2::CENTER_TOP,
            format!("{:.2}", min_val),
            egui::FontId::default(),
            Color32::BLACK,
        );

        painter.text(
            Pos2::new(rect.left() + 5.0, rect.center().y),
            egui::Align2::LEFT_CENTER,
            "Death",
            egui::FontId::default(),
            Color32::BLACK,
        );

        /*painter.text(
            Pos2::new(plot_rect.left(), plot_rect.top()),
            egui::Align2::RIGHT_CENTER,
            format!("{:.2}", max_val),
            egui::FontId::default(),
            Color32::BLACK,
        );*/

        // Add infinity symbol at top
        if !infinite_points.is_empty() {
            painter.text(
                Pos2::new(rect.left() + 5.0, plot_rect.top()),
                egui::Align2::LEFT_CENTER,
                "∞",
                egui::FontId::proportional(14.0),
                Color32::BLACK,
            );
        }
    }

    fn draw_graph_preview(&self, painter: &egui::Painter, graph: &Graph, rect: Rect) {
        // calculate bounds in
        let (min_x, min_y) = graph
            .vertices
            .iter()
            .fold((f64::INFINITY, f64::INFINITY), |(min_x, min_y), v| {
                (min_x.min(v.x), min_y.min(v.y))
            });
        let (max_x, max_y) = graph.vertices.iter().fold(
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
            |(max_x, max_y), v| (max_x.max(v.x), max_y.max(v.y)),
        );

        // Calculate offsets to center and scale the graph
        let graph_width = (max_x - min_x).max(1e-5);
        let graph_height = (max_y - min_y).max(1e-5);
        let scale =
            (rect.width() / graph_width as f32).min(rect.height() / graph_height as f32) * 0.8;
        let grid_size = scale;
        // bounds center in world coordinates
        let center_x = (min_x + max_x) / 2.0;
        let center_y = (min_y + max_y) / 2.0;

        // Draw edges
        for i in 0..graph.vertices.len() {
            for j in (i + 1)..graph.vertices.len() {
                if graph.has_edge(Edge::new(i, j)) {
                    let v1 = &graph.vertices[i];
                    let v2 = &graph.vertices[j];

                    let p1 = world_to_screen(
                        Pos2::new((v1.x - center_x) as f32, (v1.y - center_y) as f32),
                        rect,
                        grid_size,
                    );
                    let p2 = world_to_screen(
                        Pos2::new((v2.x - center_x) as f32, (v2.y - center_y) as f32),
                        rect,
                        grid_size,
                    );
                    arc_edge_painter(p1, p2, false, Stroke::new(1.5, Color32::GRAY), painter);
                    // painter.line_segment([p1, p2], Stroke::new(1.5, Color32::GRAY));
                }
            }
        }

        // Draw vertices
        for vertex in &graph.vertices {
            let pos = world_to_screen(
                Pos2::new((vertex.x - center_x) as f32, (vertex.y - center_y) as f32),
                rect,
                grid_size,
            );

            painter.circle_filled(pos, 4.0, Color32::WHITE);
            painter.circle_stroke(pos, 4.0, Stroke::new(1.5, Color32::BLACK));
        }
    }

    fn draw_small_persistence_diagram(
        &self,
        ui: &mut egui::Ui,
        diagram: &crate::graph::PersistenceDiagram,
        color: Color32,
        width: f32,
        height: f32,
    ) {
        let (response, painter) =
            ui.allocate_painter(Vec2::new(width, height), egui::Sense::hover());

        let rect = response.rect;

        if diagram.points.is_empty() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No points",
                egui::FontId::proportional(10.0),
                Color32::GRAY,
            );
            return;
        }

        // Find bounds for finite points
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for pt in diagram.points.iter() {
            min_val = min_val.min(pt.x);
            max_val = if pt.y.is_infinite() {
                max_val.max(pt.x)
            } else {
                max_val.max(pt.y)
            };
        }

        // Separate finite and infinite points
        let finite_points: Vec<_> = diagram.points.iter().filter(|p| p.y.is_finite()).collect();
        let infinite_points: Vec<_> = diagram
            .points
            .iter()
            .filter(|p| p.y.is_infinite())
            .collect();

        if max_val == min_val {
            max_val = min_val + 1.0;
        }

        let margin = 10.0;
        let plot_rect = rect.shrink(margin);

        // Draw diagonal
        painter.line_segment(
            [plot_rect.left_bottom(), plot_rect.right_top()],
            Stroke::new(0.5, Color32::from_gray(180)),
        );

        // Draw axes
        painter.line_segment(
            [
                Pos2::new(plot_rect.left(), plot_rect.bottom()),
                plot_rect.right_bottom(),
            ],
            Stroke::new(0.5, Color32::BLACK),
        );
        painter.line_segment(
            [
                Pos2::new(plot_rect.left(), plot_rect.bottom()),
                plot_rect.left_top(),
            ],
            Stroke::new(0.5, Color32::BLACK),
        );

        // Draw finite points
        for point in finite_points {
            let x_norm = ((point.x - min_val) / (max_val - min_val)) as f32;
            let y_norm = ((point.y - min_val) / (max_val - min_val)) as f32;

            let screen_pos = Pos2::new(
                plot_rect.left() + x_norm * plot_rect.width(),
                plot_rect.bottom() - y_norm * plot_rect.height(),
            );

            let radius = 2.0 + ((point.mult - 1) as f32).atan() + 1.;
            // let radius = radius * 1.5;
            painter.circle_filled(screen_pos, radius, color);
            painter.circle_stroke(screen_pos, radius, Stroke::new(1.0, Color32::BLACK));
            // small mult inside of point
            if point.mult > 1 {
                painter.text(
                    screen_pos,
                    egui::Align2::CENTER_CENTER,
                    format!("{}", point.mult),
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
                );
            }
        }

        // Draw infinite points at the top
        for point in &infinite_points {
            let x_norm = ((point.x - min_val) / (max_val - min_val)) as f32;

            let screen_pos = Pos2::new(
                plot_rect.left() + x_norm * plot_rect.width(),
                plot_rect.top(),
            );

            let radius = 2.0 + ((point.mult - 1) as f32).atan() + 1.;
            // let radius = radius * 1.5;
            painter.circle_filled(screen_pos, radius, color);
            painter.circle_stroke(screen_pos, radius, Stroke::new(1.0, Color32::BLACK));
            // small mult inside of point
            if point.mult > 1 {
                painter.text(
                    screen_pos,
                    egui::Align2::CENTER_CENTER,
                    format!("{}", point.mult),
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
                );
            }

            // Small arrow for infinity
            let arrow_start = screen_pos + Vec2::new(0.0, -radius - 1.0);
            let arrow_end = arrow_start + Vec2::new(0.0, -4.0);
            painter.line_segment([arrow_start, arrow_end], Stroke::new(0.8, Color32::BLACK));
            painter.line_segment(
                [arrow_end, arrow_end + Vec2::new(-1.5, 1.5)],
                Stroke::new(0.8, Color32::BLACK),
            );
            painter.line_segment(
                [arrow_end, arrow_end + Vec2::new(1.5, 1.5)],
                Stroke::new(0.8, Color32::BLACK),
            );
        }
    }
}
